mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT;
use candle::cuda_backend::cudarc::driver::DevicePtrMut;
use candle::cuda_backend::WrapErr;
use candle::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor};
use half::{bf16, f16};
use std::ptr;

fn layer_norm_internal_type(dtype: DType) -> Result<u32> {
    let internal_type = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        dtype => candle::bail!("dtype {dtype:?} is not supported"),
    };
    Ok(internal_type)
}

pub struct LayerNorm {
    pub epsilon: f32,
    pub is_rms_norm: bool,
    pub output_only: bool,
    pub gamma: Tensor,
    pub beta: Option<Tensor>,
}

fn round_multiple(x: usize, m: usize) -> usize {
    (x + m - 1) / m * m
}

impl LayerNorm {
    fn fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
        r: Option<&candle::CudaStorage>,
        r_l: Option<&Layout>,
    ) -> Result<(candle::CudaStorage, Shape)> {
        // Assume all tensors are on the same device and take device of x
        let dev = x.device();

        // Get internal layer norm type id for the given dtype
        let layer_norm_type = layer_norm_internal_type(x.dtype())?;

        // Make sure that gamma is a CUDA tensor and get the underlying storage
        let (g, g_l) = self.gamma.storage_and_layout();
        let g = match &*g {
            Storage::Cuda(g) => g,
            _ => candle::bail!("gamma must be a cuda tensor"),
        };

        // Get cuda slices for all tensors
        let x = x.as_cuda_slice::<T>()?;
        let g = g.as_cuda_slice::<T>()?;

        // Input matrix layout
        let (rows, cols) = x_l.shape().dims2()?;

        if !(cols % 8 == 0 && cols <= 8192) {
            candle::bail!("hidden size must be % 8 and <= 8192")
        }
        if g_l.shape().elem_count() != cols {
            candle::bail!("gamma length must match hidden size {cols}")
        }

        let x_stride = x_l.stride();
        let g_stride = g_l.stride();

        let x_rank = x_stride.len();
        let g_rank = g_stride.len();

        if x_rank != 2 {
            candle::bail!("layer-norm expects input tensors of rank 2. Found: {x_rank}")
        }
        if g_rank == 0 {
            candle::bail!("gamma must have at least one dimension")
        }
        if x_stride[x_rank - 1] != 1 {
            candle::bail!("the last dim of x must be contiguous {x_stride:?}")
        }
        if g_stride[g_rank - 1] != 1 {
            candle::bail!("the last dim of g must be contiguous {g_stride:?}")
        }

        // Get cuda views for all tensors
        let x = x.slice(x_l.start_offset()..);
        let g = g.slice(g_l.start_offset()..);

        // Round cols to match with the correct kernel
        let cols_rounded = if cols <= 1536 {
            round_multiple(cols, 256)
        } else if cols <= 3072 {
            round_multiple(cols, 512)
        } else {
            round_multiple(cols, 1024)
        };

        let is_rms_norm = if self.is_rms_norm { 1 } else { 0 };
        let stream = dev.cuda_stream();
        let beta_storage = self.beta.as_ref().map(Tensor::storage_and_layout);

        // If beta is et, get ids device pointer
        let (b_ptr, b_sync) = if let Some((b, b_l)) = beta_storage.as_ref() {
            // Make sure that beta is a CUDA tensor and get the underlying storage
            let b = match &**b {
                Storage::Cuda(b) => b,
                _ => candle::bail!("beta must be a cuda tensor"),
            };

            if b_l.shape().elem_count() != cols {
                candle::bail!("beta length must match hidden size {cols}")
            }
            let b = b.as_cuda_slice::<T>()?;
            let b = b.slice(b_l.start_offset()..);

            let b_stride = b_l.stride();
            let b_rank = b_stride.len();

            if b_rank == 0 {
                candle::bail!("beta must have at least one dimension")
            }
            if b_stride[b_rank - 1] != 1 {
                candle::bail!("the last dim of b must be contiguous {b_stride:?}")
            }
            let (b_ptr, b_sync) = b.view_ptr(&stream);
            (b_ptr as *const core::ffi::c_void, Some(b_sync))
        } else {
            (ptr::null() as *const std::ffi::c_void, None)
        };

        // If residual is set, get its device pointer
        let (r_ptr, r_sync) = if let (Some(r), Some(r_l)) = (r, r_l) {
            // Check shape
            let expected_shape = x_l.shape().dims2()?;
            if r_l.shape().dims2()? != expected_shape {
                candle::bail!("shape mismatch x {:?} and r {:?}", x_l.shape(), r_l.shape());
            }

            let r = r.as_cuda_slice::<T>()?;
            let r = r.slice(r_l.start_offset()..);

            let r_stride = r_l.stride();
            let r_rank = r_stride.len();

            if r_rank != 2 {
                candle::bail!("layer-norm expects input tensors of rank 2. Found: {r_rank}")
            }

            if r_stride[r_rank - 1] != 1 {
                candle::bail!("the last dim of r must be contiguous {r_stride:?}")
            }
            let (r_ptr, r_sync) = r.view_ptr(&stream);
            (r_ptr as *const std::ffi::c_void, Some(r_sync))
        } else {
            (ptr::null() as *const std::ffi::c_void, None)
        };

        let output_rows = if self.output_only { rows } else { rows * 2 };
        let out_shape = Shape::from((output_rows, cols));

        let mut out = unsafe { dev.alloc::<T>(out_shape.elem_count()) }?;
        let (mut dst, mut dst_add) = if self.output_only {
            (out.slice_mut(..rows * cols), None)
        } else {
            let (dst, dst_add) = out.split_at_mut(rows * cols);
            (dst, Some(dst_add))
        };

        // The output-only path does not expose the statistics, so avoid allocating them.
        let mut mu = if self.output_only {
            None
        } else {
            Some(unsafe { dev.alloc::<f32>(rows) }?)
        };
        let mut rsigma = if self.output_only {
            None
        } else {
            Some(unsafe { dev.alloc::<f32>(rows) }?)
        };

        // Get cuda device pointers from cuda slices
        let (x_ptr, x_sync) = x.view_ptr(&stream);
        let (g_ptr, g_sync) = g.view_ptr(&stream);
        let (dst_add_ptr, dst_add_sync) = if let Some(dst_add) = dst_add.as_mut() {
            let (dst_add_ptr, dst_add_sync) = dst_add.device_ptr_mut(&stream);
            (dst_add_ptr as *mut core::ffi::c_void, Some(dst_add_sync))
        } else {
            (ptr::null_mut(), None)
        };
        let (dst_ptr, dst_sync) = dst.device_ptr_mut(&stream);
        let (mu_ptr, mu_sync) = if let Some(mu) = mu.as_mut() {
            let (mu_ptr, mu_sync) = mu.device_ptr_mut(&stream);
            (mu_ptr as *mut core::ffi::c_void, Some(mu_sync))
        } else {
            (ptr::null_mut(), None)
        };
        let (rsigma_ptr, rsigma_sync) = if let Some(rsigma) = rsigma.as_mut() {
            let (rsigma_ptr, rsigma_sync) = rsigma.device_ptr_mut(&stream);
            (rsigma_ptr as *mut core::ffi::c_void, Some(rsigma_sync))
        } else {
            (ptr::null_mut(), None)
        };
        let x_ptr = x_ptr as *const core::ffi::c_void;
        let g_ptr = g_ptr as *const core::ffi::c_void;
        let dst_ptr = dst_ptr as *mut core::ffi::c_void;

        let multi_processors_count = stream
            .context()
            .attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .w()?;

        let status = unsafe {
            // Launch Kernel
            ffi::run_ln(
                x_ptr,
                r_ptr,
                g_ptr,
                b_ptr,
                dst_add_ptr,
                dst_ptr,
                mu_ptr,
                rsigma_ptr,
                self.epsilon,
                cols_rounded as u32,
                rows as u32,
                cols as u32,
                multi_processors_count,
                layer_norm_type,
                layer_norm_type,
                layer_norm_type,
                layer_norm_type,
                2,
                is_rms_norm,
                stream.cu_stream().cast(),
            )
        };
        if status != 0 {
            candle::bail!("layer-norm CUDA kernel failed with status {status}");
        }
        drop((
            b_sync,
            r_sync,
            x_sync,
            g_sync,
            dst_add_sync,
            dst_sync,
            mu_sync,
            rsigma_sync,
        ));

        let out = candle::CudaStorage::wrap_cuda_slice(out, dev.clone());

        Ok((out, out_shape))
    }
}

impl candle::CustomOp1 for LayerNorm {
    fn name(&self) -> &'static str {
        "fused-layer-norm"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for fused-layer-norm")
    }

    fn cuda_fwd(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.fwd::<f16>(x, x_l, None, None),
            DType::BF16 => self.fwd::<bf16>(x, x_l, None, None),
            DType::F32 => self.fwd::<f32>(x, x_l, None, None),
            dt => {
                candle::bail!("fused-layer-norm is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }
}

impl candle::CustomOp2 for LayerNorm {
    fn name(&self) -> &'static str {
        "fused-layer-norm"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for fused-layer-norm")
    }

    fn cuda_fwd(
        &self,
        x: &candle::CudaStorage,
        x_l: &Layout,
        r: &candle::CudaStorage,
        r_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.fwd::<f16>(x, x_l, Some(r), Some(r_l)),
            DType::BF16 => self.fwd::<bf16>(x, x_l, Some(r), Some(r_l)),
            DType::F32 => self.fwd::<f32>(x, x_l, Some(r), Some(r_l)),
            dt => {
                candle::bail!("fused-layer-norm is only supported for f32, f16 and bf16 ({dt:?})")
            }
        }
    }
}

/// Layer Normalization Layer
///
/// # Arguments
///
/// * `x` - Input tensor of rank 2
/// * `gamma` - Channel scale
/// * `beta` - Channel bias
/// * `epsilon` - A value added to the denominator for numerical stability
///
/// The resulting tensor has the same dimensions as `x`
pub fn layer_norm(
    x: &Tensor,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    epsilon: f32,
) -> Result<Tensor> {
    let op = LayerNorm {
        epsilon,
        gamma: gamma.clone(),
        beta: beta.cloned(),
        is_rms_norm: false,
        output_only: false,
    };
    let results = x.apply_op1(op)?;
    let rows = x.dims()[0];
    results.narrow(0, 0, rows)
}

/// Layer Normalization Layer that only returns the normalized output.
///
/// Unlike [`layer_norm`], this avoids allocating the unused residual-add and statistics buffers.
pub fn layer_norm_output_only(
    x: &Tensor,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    epsilon: f32,
) -> Result<Tensor> {
    let op = LayerNorm {
        epsilon,
        gamma: gamma.clone(),
        beta: beta.cloned(),
        is_rms_norm: false,
        output_only: true,
    };
    x.apply_op1(op)
}

/// Fused Add Layer Normalization Layer
///
/// # Arguments
///
/// * `x` - Input tensor of rank 2
/// * `res` - Residual tensor of rank 2. Will be added to `x` before normalization. Must have
/// the same shape as `x`.
/// * `gamma` - Channel scale
/// * `beta` - Channel bias
/// * `epsilon` - A value added to the denominator for numerical stability
///
/// The resulting tensors have the same dimensions as `x`
/// First tensor is the result of the normalization, second is the result of the residual add
pub fn fused_add_layer_norm(
    x: &Tensor,
    res: &Tensor,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    epsilon: f32,
) -> Result<(Tensor, Tensor)> {
    let op = LayerNorm {
        epsilon,
        gamma: gamma.clone(),
        beta: beta.cloned(),
        is_rms_norm: false,
        output_only: false,
    };
    let results = x.apply_op2(&res, op)?;
    let rows = x.dims()[0];
    Ok((results.narrow(0, 0, rows)?, results.narrow(0, rows, rows)?))
}

/// Fused Add Layer Normalization Layer that only returns the normalized output.
///
/// The residual is still added before normalization, but the intermediate sum is not stored.
pub fn fused_add_layer_norm_output_only(
    x: &Tensor,
    res: &Tensor,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    epsilon: f32,
) -> Result<Tensor> {
    let op = LayerNorm {
        epsilon,
        gamma: gamma.clone(),
        beta: beta.cloned(),
        is_rms_norm: false,
        output_only: true,
    };
    x.apply_op2(res, op)
}

/// Layer RMS Normalization Layer
///
/// # Arguments
///
/// * `x` - Input tensor of rank 2
/// * `gamma` - Channel scale
/// * `beta` - Channel bias
/// * `epsilon` - A value added to the denominator for numerical stability
///
/// The resulting tensor has the same dimensions as `x`
pub fn rms_norm(x: &Tensor, gamma: &Tensor, beta: Option<&Tensor>, epsilon: f32) -> Result<Tensor> {
    let op = LayerNorm {
        epsilon,
        gamma: gamma.clone(),
        beta: beta.cloned(),
        is_rms_norm: true,
        output_only: false,
    };
    let results = x.apply_op1(op)?;
    let rows = x.dims()[0];
    results.narrow(0, 0, rows)
}

/// Fused Add RMS Normalization Layer
///
/// # Arguments
///
/// * `x` - Input tensor of rank 2
/// * `res` - Residual tensor of rank 2. Will be added to `x` before normalization. Must have
/// the same shape as `x`.
/// * `gamma` - Channel scale
/// * `beta` - Channel bias
/// * `epsilon` - A value added to the denominator for numerical stability
///
/// The resulting tensors have the same dimensions as `x`
/// First tensor is the result of the normalization, second is the result of the residual add
pub fn fused_add_rms_norm(
    x: &Tensor,
    res: &Tensor,
    gamma: &Tensor,
    beta: Option<&Tensor>,
    epsilon: f32,
) -> Result<(Tensor, Tensor)> {
    let op = LayerNorm {
        epsilon,
        gamma: gamma.clone(),
        beta: beta.cloned(),
        is_rms_norm: true,
        output_only: false,
    };
    let results = x.apply_op2(&res, op)?;
    let rows = x.dims()[0];
    Ok((results.narrow(0, 0, rows)?, results.narrow(0, rows, rows)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    fn layer_norm_truth(
        x: &Tensor,
        gamma: &Tensor,
        beta: Option<&Tensor>,
        epsilon: f64,
        rms: bool,
    ) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };

        let (_seq_len, hidden_size) = x.shape().dims2()?;
        let x = x.to_dtype(internal_dtype)?;

        let x = if !rms {
            let mean_x = (x.sum_keepdim(1)? / hidden_size as f64)?;
            x.broadcast_sub(&mean_x)?
        } else {
            x
        };

        let norm_x = (x.sqr()?.sum_keepdim(1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + epsilon)?.sqrt()?)?;

        let mut x = x_normed.to_dtype(x_dtype)?.broadcast_mul(gamma)?;
        if let Some(beta) = beta {
            x = x.broadcast_add(beta)?;
        }
        Ok(x)
    }

    fn to_vec2_round(t: Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
        let b = 10f32.powi(digits);
        let t = t.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let t = t
            .iter()
            .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
            .collect();
        Ok(t)
    }

    #[test]
    fn test_layer_norm() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let res = layer_norm(&x, &g, Some(&b), 1e-12)?;
        let truth = layer_norm_truth(&x, &g, Some(&b), 1e-12, false)?;

        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }

    #[test]
    fn test_layer_norm_no_bias() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let res = layer_norm(&x, &g, None, 1e-12)?;
        let truth = layer_norm_truth(&x, &g, None, 1e-12, false)?;

        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }

    #[test]
    fn test_rms_norm() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let res = rms_norm(&x, &g, Some(&b), 1e-12)?;
        let truth = layer_norm_truth(&x, &g, Some(&b), 1e-12, true)?;
        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }

    #[test]
    fn test_rms_norm_no_bias() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let res = rms_norm(&x, &g, None, 1e-12)?;
        let truth = layer_norm_truth(&x, &g, None, 1e-12, true)?;

        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }

    #[test]
    fn test_layer_norm_add() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let r = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let (res, res_add) = fused_add_layer_norm(&x, &r, &g, Some(&b), 1e-12)?;
        let truth_add = (x + r)?;
        let truth = layer_norm_truth(&truth_add, &g, Some(&b), 1e-12, false)?;
        assert_eq!(to_vec2_round(res_add, 3)?, to_vec2_round(truth_add, 3)?);
        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }

    #[test]
    fn test_rms_norm_add() -> Result<()> {
        let device = Device::new_cuda(0)?;

        let x = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let r = Tensor::randn(0., 1., (4, 8), &device)?.to_dtype(DType::F32)?;
        let g = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;
        let b = Tensor::randn(0., 1., 8, &device)?.to_dtype(DType::F32)?;

        let (res, res_add) = fused_add_rms_norm(&x, &r, &g, Some(&b), 1e-12)?;
        let truth_add = (x + r)?;
        let truth = layer_norm_truth(&truth_add, &g, Some(&b), 1e-12, true)?;
        assert_eq!(to_vec2_round(res_add, 3)?, to_vec2_round(truth_add, 3)?);
        assert_eq!(to_vec2_round(res, 3)?, to_vec2_round(truth, 3)?);
        Ok(())
    }
}
