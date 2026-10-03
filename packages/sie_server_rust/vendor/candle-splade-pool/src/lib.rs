mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtrMut;
use candle::{CpuStorage, DType, Layout, Result, Shape, Tensor};
use half::f16;

const MAX_GRID_Y: usize = 65_535;

#[derive(Debug)]
struct SegmentedMaxLog1p;

fn validate_layouts(
    input_layout: &Layout,
    offsets_layout: &Layout,
) -> Result<(usize, usize, usize, usize, Shape)> {
    let input_dims = input_layout.dims();
    if input_dims.len() != 2 {
        candle::bail!(
            "SPLADE segmented pool expects input shape [tokens, vocab], got rank {}",
            input_dims.len()
        )
    }
    let offsets_dims = offsets_layout.dims();
    if offsets_dims.len() != 1 {
        candle::bail!(
            "SPLADE segmented pool expects offsets shape [batch + 1], got rank {}",
            offsets_dims.len()
        )
    }
    if input_layout.contiguous_offsets().is_none() {
        candle::bail!("SPLADE segmented pool expects contiguous input")
    }
    if offsets_layout.contiguous_offsets().is_none() {
        candle::bail!("SPLADE segmented pool expects contiguous offsets")
    }

    let total_tokens = input_dims[0];
    let vocab_size = input_dims[1];
    let offset_count = offsets_dims[0];
    if total_tokens == 0 {
        candle::bail!("SPLADE segmented pool requires at least one token")
    }
    if vocab_size == 0 {
        candle::bail!("SPLADE segmented pool requires a non-empty vocabulary")
    }
    if offset_count < 2 {
        candle::bail!("SPLADE segmented pool requires at least one sequence")
    }

    let batch_size = offset_count - 1;
    if batch_size > MAX_GRID_Y {
        candle::bail!(
            "SPLADE segmented pool batch {batch_size} exceeds CUDA grid.y limit {MAX_GRID_Y}"
        )
    }
    if total_tokens > u32::MAX as usize {
        candle::bail!("SPLADE segmented pool token count exceeds U32 offsets")
    }
    if vocab_size > i32::MAX as usize {
        candle::bail!("SPLADE segmented pool vocabulary exceeds i32 kernel ABI")
    }
    if batch_size > i32::MAX as usize {
        candle::bail!("SPLADE segmented pool batch exceeds i32 kernel ABI")
    }
    let output_elems = batch_size
        .checked_mul(vocab_size)
        .ok_or_else(|| candle::Error::msg("SPLADE segmented pool output size overflow"))?;

    Ok((
        total_tokens,
        vocab_size,
        batch_size,
        output_elems,
        Shape::from((batch_size, vocab_size)),
    ))
}

impl candle::CustomOp2 for SegmentedMaxLog1p {
    fn name(&self) -> &'static str {
        "splade-segmented-max-log1p"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("SPLADE segmented pool is only supported on CUDA")
    }

    fn cuda_fwd(
        &self,
        input: &candle::CudaStorage,
        input_layout: &Layout,
        offsets: &candle::CudaStorage,
        offsets_layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        if input.dtype() != DType::F16 {
            candle::bail!(
                "SPLADE segmented pool only supports F16 input, got {:?}",
                input.dtype()
            )
        }
        if offsets.dtype() != DType::U32 {
            candle::bail!(
                "SPLADE segmented pool requires U32 offsets, got {:?}",
                offsets.dtype()
            )
        }
        let (total_tokens, vocab_size, batch_size, output_elems, output_shape) =
            validate_layouts(input_layout, offsets_layout)?;
        let (input_start, input_end) = input_layout
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::msg("SPLADE segmented pool expects contiguous input"))?;
        let (offsets_start, offsets_end) =
            offsets_layout.contiguous_offsets().ok_or_else(|| {
                candle::Error::msg("SPLADE segmented pool expects contiguous offsets")
            })?;

        let device = input.device();
        let stream = device.cuda_stream();
        let input = input.as_cuda_slice::<f16>()?.slice(input_start..input_end);
        let offsets = offsets
            .as_cuda_slice::<u32>()?
            .slice(offsets_start..offsets_end);
        let mut output = unsafe { device.alloc::<f16>(output_elems)? };

        let (input_ptr, input_sync) = input.view_ptr(&stream);
        let (offsets_ptr, offsets_sync) = offsets.view_ptr(&stream);
        let (output_ptr, output_sync) = output.device_ptr_mut(&stream);
        let status = unsafe {
            ffi::splade_segmented_max_log1p_f16(
                input_ptr as *const core::ffi::c_void,
                offsets_ptr as *const core::ffi::c_void,
                output_ptr as *mut core::ffi::c_void,
                total_tokens as i64,
                vocab_size as i32,
                batch_size as i32,
                stream.cu_stream().cast(),
            )
        };
        drop((input_sync, offsets_sync, output_sync));
        if status != 0 {
            candle::bail!("SPLADE segmented pool CUDA kernel failed with status {status}")
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, device.clone());
        Ok((output, output_shape))
    }
}

/// Max-pool packed non-negative F16 SPLADE activations by sequence and apply
/// `log1p` to each pooled vocabulary weight in the same CUDA kernel.
pub fn segmented_max_log1p(input: &Tensor, offsets: &Tensor) -> Result<Tensor> {
    input.apply_op2_no_bwd(offsets, &SegmentedMaxLog1p)
}
