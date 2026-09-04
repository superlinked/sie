mod ffi;

use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};
use candle::cuda_backend::{CudaDevice, DeviceId};
use candle::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor};
use half::{bf16, f16};
use std::collections::HashMap;
use std::ffi::{c_int, c_long};
use std::sync::{Arc, Mutex};

const BF16_GELU_ERF_LUT_SIZE: usize = 1 << 16;

#[derive(Default)]
struct Bf16GeluErfLutCache {
    luts: Mutex<HashMap<DeviceId, Arc<CudaSlice<bf16>>>>,
}

/// Reusable exact-erf GELU gate whose CUDA lookup tables live as long as its owner.
///
/// Loaded models should retain and clone this value across layers so unloading the
/// model also releases the cached device allocation.
#[derive(Clone)]
pub struct GeluErfGate {
    intermediate_size: usize,
    bf16_luts: Arc<Bf16GeluErfLutCache>,
}

impl GeluErfGate {
    pub fn new(intermediate_size: usize) -> Result<Self> {
        validate_intermediate_size(intermediate_size)?;
        Ok(Self {
            intermediate_size,
            bf16_luts: Arc::new(Bf16GeluErfLutCache::default()),
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        gelu_gate_with_kind(
            input,
            self.intermediate_size,
            GeluGateKind::ErfFirst,
            Some(self.bf16_luts.clone()),
        )
    }
}

#[derive(Clone, Copy)]
enum GeluGateKind {
    TanhSecond,
    ErfFirst,
}

struct GeluGate {
    intermediate_size: usize,
    kind: GeluGateKind,
    bf16_luts: Option<Arc<Bf16GeluErfLutCache>>,
}

impl GeluGate {
    fn output_shape(&self, layout: &Layout) -> Result<(Shape, usize)> {
        let mut dims = layout.shape().dims().to_vec();
        let Some(last_dim) = dims.last_mut() else {
            candle::bail!("gelu-gate expects rank >= 1")
        };
        validate_intermediate_size(self.intermediate_size)?;
        let expected = self
            .intermediate_size
            .checked_mul(2)
            .ok_or_else(|| candle::Error::msg("gelu-gate input width overflow"))?;
        if *last_dim != expected {
            candle::bail!(
                "gelu-gate expected last dim {}, got {}",
                expected,
                *last_dim
            );
        }
        if layout.contiguous_offsets().is_none() {
            candle::bail!("gelu-gate expects a contiguous input tensor");
        }

        *last_dim = self.intermediate_size;
        let rows = layout.shape().elem_count() / expected;
        Ok((Shape::from(dims), rows))
    }
}

impl candle::CustomOp1 for GeluGate {
    fn name(&self) -> &'static str {
        match self.kind {
            GeluGateKind::TanhSecond => "gelu-gate",
            GeluGateKind::ErfFirst => "gelu-erf-gate",
        }
    }

    fn cpu_fwd(&self, _storage: &CpuStorage, _layout: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("gelu-gate is only supported on CUDA")
    }

    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let (out_shape, rows) = self.output_shape(layout)?;
        let elem_count = out_shape.elem_count();
        let dev = storage.device();
        let stream = dev.cuda_stream();
        let (start, end) = layout
            .contiguous_offsets()
            .ok_or_else(|| candle::Error::msg("gelu-gate expects a contiguous input tensor"))?;

        match storage.dtype() {
            DType::F16 => launch::<f16>(
                storage,
                start,
                end,
                rows,
                elem_count,
                self.intermediate_size,
                0,
                self.kind,
                self.bf16_luts.as_deref(),
                out_shape,
                &stream,
            ),
            DType::BF16 => launch::<bf16>(
                storage,
                start,
                end,
                rows,
                elem_count,
                self.intermediate_size,
                1,
                self.kind,
                self.bf16_luts.as_deref(),
                out_shape,
                &stream,
            ),
            DType::F32 => launch::<f32>(
                storage,
                start,
                end,
                rows,
                elem_count,
                self.intermediate_size,
                2,
                self.kind,
                self.bf16_luts.as_deref(),
                out_shape,
                &stream,
            ),
            dtype => candle::bail!("gelu-gate only supports f16/bf16/f32, got {dtype:?}"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn launch<T>(
    storage: &candle::CudaStorage,
    start: usize,
    end: usize,
    rows: usize,
    elem_count: usize,
    intermediate_size: usize,
    dtype: u32,
    kind: GeluGateKind,
    bf16_luts: Option<&Bf16GeluErfLutCache>,
    out_shape: Shape,
    stream: &std::sync::Arc<candle::cuda_backend::cudarc::driver::CudaStream>,
) -> Result<(candle::CudaStorage, Shape)>
where
    T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
{
    let dev = storage.device();
    let input = storage.as_cuda_slice::<T>()?.slice(start..end);
    let mut output = unsafe { dev.alloc::<T>(elem_count)? };
    if elem_count > 0 {
        let lut = if matches!(kind, GeluGateKind::ErfFirst) && dtype == 1 {
            let cache = bf16_luts.ok_or_else(|| {
                candle::Error::msg("BF16 exact-GELU gate requires an owner-scoped lookup cache")
            })?;
            Some(bf16_gelu_erf_lut(dev, stream, cache)?)
        } else {
            None
        };
        let (input_ptr, input_sync) = input.view_ptr(stream);
        let (output_ptr, output_sync) = output.device_ptr_mut(stream);
        let (lut_ptr, lut_sync) = match lut.as_ref() {
            Some(lut) => {
                let (ptr, sync) = lut.device_ptr(stream);
                (Some(ptr), Some(sync))
            }
            None => (None, None),
        };
        let status = unsafe {
            let input = input_ptr as *const core::ffi::c_void;
            let output = output_ptr as *mut core::ffi::c_void;
            match kind {
                GeluGateKind::TanhSecond => ffi::gelu_gate(
                    input,
                    output,
                    rows as c_long,
                    intermediate_size as c_int,
                    dtype,
                    stream.cu_stream().cast(),
                ),
                GeluGateKind::ErfFirst if dtype == 1 => ffi::gelu_erf_gate_bf16_lut(
                    input,
                    output,
                    lut_ptr.expect("BF16 exact-GELU lookup table must be initialized")
                        as *const core::ffi::c_void,
                    rows as c_long,
                    intermediate_size as c_int,
                    stream.cu_stream().cast(),
                ),
                GeluGateKind::ErfFirst => ffi::gelu_erf_gate(
                    input,
                    output,
                    rows as c_long,
                    intermediate_size as c_int,
                    dtype,
                    stream.cu_stream().cast(),
                ),
            }
        };
        drop((input_sync, output_sync, lut_sync));
        if status != 0 {
            candle::bail!("gelu-gate CUDA kernel failed with status {status}");
        }
    }
    let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok((output, out_shape))
}

fn bf16_gelu_erf_lut(
    dev: &CudaDevice,
    stream: &Arc<candle::cuda_backend::cudarc::driver::CudaStream>,
    cache: &Bf16GeluErfLutCache,
) -> Result<Arc<CudaSlice<bf16>>> {
    let mut luts = cache
        .luts
        .lock()
        .map_err(|_| candle::Error::msg("BF16 exact-GELU lookup-table cache lock poisoned"))?;
    if let Some(lut) = luts.get(&dev.id()) {
        return Ok(lut.clone());
    }

    let mut lut = unsafe { dev.alloc::<bf16>(BF16_GELU_ERF_LUT_SIZE)? };
    let (lut_ptr, lut_sync) = lut.device_ptr_mut(stream);
    let status = unsafe {
        ffi::init_gelu_erf_bf16_lut(lut_ptr as *mut core::ffi::c_void, stream.cu_stream().cast())
    };
    drop(lut_sync);
    if status != 0 {
        candle::bail!("BF16 exact-GELU lookup-table initialization failed with status {status}");
    }

    let lut = Arc::new(lut);
    luts.insert(dev.id(), lut.clone());
    Ok(lut)
}

pub fn gelu_gate(input: &Tensor, intermediate_size: usize) -> Result<Tensor> {
    gelu_gate_with_kind(input, intermediate_size, GeluGateKind::TanhSecond, None)
}

fn gelu_gate_with_kind(
    input: &Tensor,
    intermediate_size: usize,
    kind: GeluGateKind,
    bf16_luts: Option<Arc<Bf16GeluErfLutCache>>,
) -> Result<Tensor> {
    let (storage, _) = input.storage_and_layout();
    match &*storage {
        Storage::Cuda(_) => {}
        _ => candle::bail!("gelu-gate expects a CUDA tensor"),
    }
    input.apply_op1_no_bwd(&GeluGate {
        intermediate_size,
        kind,
        bf16_luts,
    })
}

fn validate_intermediate_size(intermediate_size: usize) -> Result<()> {
    if intermediate_size == 0 || intermediate_size > i32::MAX as usize {
        candle::bail!("gelu-gate intermediate size must be in 1..=i32::MAX");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    fn bf16_bits(tensor: &Tensor) -> Result<Vec<u16>> {
        Ok(tensor
            .flatten_all()?
            .to_vec1::<bf16>()?
            .into_iter()
            .map(bf16::to_bits)
            .collect())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn bf16_lut_handles_tail_shape_and_repeated_calls() -> Result<()> {
        const ROWS: usize = 9;
        const INTERMEDIATE_SIZE: usize = 33;

        let device = Device::new_cuda(0)?;
        let gates = (0..ROWS * INTERMEDIATE_SIZE)
            .map(|index| bf16::from_f32((index % 29) as f32 / 4.0 - 3.5))
            .collect::<Vec<_>>();
        let ups = (0..ROWS * INTERMEDIATE_SIZE)
            .map(|index| bf16::from_f32((index % 17) as f32 / 5.0 - 1.5))
            .collect::<Vec<_>>();
        let mut input = Vec::with_capacity(ROWS * INTERMEDIATE_SIZE * 2);
        for row in 0..ROWS {
            let start = row * INTERMEDIATE_SIZE;
            let end = start + INTERMEDIATE_SIZE;
            input.extend_from_slice(&gates[start..end]);
            input.extend_from_slice(&ups[start..end]);
        }

        let input = Tensor::from_vec(input, (ROWS, INTERMEDIATE_SIZE * 2), &device)?;
        let gates = Tensor::from_vec(gates, (ROWS, INTERMEDIATE_SIZE), &device)?;
        let ups = Tensor::from_vec(ups, (ROWS, INTERMEDIATE_SIZE), &device)?;
        let expected = (&gates.gelu_erf()? * &ups)?;
        let gate = GeluErfGate::new(INTERMEDIATE_SIZE)?;
        let first = gate.forward(&input)?;
        let second = gate.forward(&input)?;

        let expected = bf16_bits(&expected)?;
        assert_eq!(bf16_bits(&first)?, expected);
        assert_eq!(bf16_bits(&second)?, expected);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn bf16_lut_matches_candle_for_all_raw_inputs() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let gates = (u16::MIN..=u16::MAX)
            .map(bf16::from_bits)
            .collect::<Vec<_>>();
        let ups = vec![bf16::from_f32(1.0); BF16_GELU_ERF_LUT_SIZE];
        let input = Tensor::from_vec(
            [gates.as_slice(), ups.as_slice()].concat(),
            (1, BF16_GELU_ERF_LUT_SIZE * 2),
            &device,
        )?;
        let gate_tensor = Tensor::from_vec(gates.clone(), BF16_GELU_ERF_LUT_SIZE, &device)?;
        let up_tensor = Tensor::from_vec(ups, BF16_GELU_ERF_LUT_SIZE, &device)?;
        let expected = (&gate_tensor.gelu_erf()? * &up_tensor)?.to_vec1::<bf16>()?;
        let actual = GeluErfGate::new(BF16_GELU_ERF_LUT_SIZE)?
            .forward(&input)?
            .to_vec2::<bf16>()?
            .remove(0);

        let mut finite_inputs = 0usize;
        let mut nan_inputs = 0usize;
        let mut infinite_inputs = 0usize;
        for (raw_bits, ((gate, actual), expected)) in
            (u16::MIN..=u16::MAX).zip(gates.iter().zip(actual.iter()).zip(expected.iter()))
        {
            if gate.is_finite() {
                finite_inputs += 1;
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "finite BF16 input 0x{raw_bits:04x}"
                );
            } else if gate.is_nan() {
                nan_inputs += 1;
                assert!(
                    actual.is_nan() && expected.is_nan(),
                    "NaN BF16 input 0x{raw_bits:04x}"
                );
            } else {
                infinite_inputs += 1;
                if expected.is_nan() {
                    assert!(actual.is_nan(), "infinite BF16 input 0x{raw_bits:04x}");
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "infinite BF16 input 0x{raw_bits:04x}"
                    );
                }
            }
        }

        assert_eq!(finite_inputs, 65_280);
        assert_eq!(nan_inputs, 254);
        assert_eq!(infinite_inputs, 2);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn f16_and_f32_exact_erf_paths_match_candle_for_multiple_rows() -> Result<()> {
        const ROWS: usize = 5;
        const INTERMEDIATE_SIZE: usize = 34;

        let device = Device::new_cuda(0)?;
        let values = (0..ROWS * INTERMEDIATE_SIZE * 2)
            .map(|index| (index % 41) as f32 / 7.0 - 2.5)
            .collect::<Vec<_>>();
        for (dtype, tolerance) in [(DType::F16, 0.004f32), (DType::F32, 1e-6f32)] {
            let input = Tensor::from_vec(values.clone(), (ROWS, INTERMEDIATE_SIZE * 2), &device)?
                .to_dtype(dtype)?;
            let halves = input.chunk(2, candle::D::Minus1)?;
            let expected = (&halves[0].gelu_erf()? * &halves[1])?.to_dtype(DType::F32)?;
            let actual = GeluErfGate::new(INTERMEDIATE_SIZE)?
                .forward(&input)?
                .to_dtype(DType::F32)?;

            for (actual, expected) in actual
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .zip(expected.flatten_all()?.to_vec1::<f32>()?)
            {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{dtype:?} exact-erf mismatch: actual={actual} expected={expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn bf16_lut_scalar_fallback_handles_misaligned_even_width_input() -> Result<()> {
        const ROWS: usize = 3;
        const INTERMEDIATE_SIZE: usize = 32;
        const ELEMENTS: usize = ROWS * INTERMEDIATE_SIZE * 2;

        let device = Device::new_cuda(0)?;
        let values = (0..=ELEMENTS)
            .map(|index| bf16::from_f32((index % 37) as f32 / 8.0 - 2.0))
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(values, ELEMENTS + 1, &device)?
            .narrow(0, 1, ELEMENTS)?
            .reshape((ROWS, INTERMEDIATE_SIZE * 2))?;
        let halves = input.chunk(2, candle::D::Minus1)?;
        let expected = (&halves[0].gelu_erf()? * &halves[1])?;
        let actual = GeluErfGate::new(INTERMEDIATE_SIZE)?.forward(&input)?;

        assert_eq!(bf16_bits(&actual)?, bf16_bits(&expected)?);
        Ok(())
    }

    #[test]
    fn bf16_lut_cache_lifetime_is_bound_to_gate_owners() -> Result<()> {
        let gate = GeluErfGate::new(32)?;
        let cache = Arc::downgrade(&gate.bf16_luts);
        let cloned = gate.clone();

        drop(gate);
        assert!(cache.upgrade().is_some());
        drop(cloned);
        assert!(cache.upgrade().is_none());
        Ok(())
    }

    #[test]
    #[ignore = "requires CUDA"]
    fn populated_bf16_lut_cache_drops_with_all_gate_owners() -> Result<()> {
        const INTERMEDIATE_SIZE: usize = 32;

        let device = Device::new_cuda(0)?;
        let input = Tensor::from_vec(
            vec![bf16::from_f32(1.0); INTERMEDIATE_SIZE * 2],
            (1, INTERMEDIATE_SIZE * 2),
            &device,
        )?;
        let gate = GeluErfGate::new(INTERMEDIATE_SIZE)?;
        let cloned = gate.clone();
        let cache = Arc::downgrade(&gate.bf16_luts);

        let output = gate.forward(&input)?;
        output.to_vec2::<bf16>()?;
        assert_eq!(
            gate.bf16_luts
                .luts
                .lock()
                .map_err(|_| candle::Error::msg("test cache lock poisoned"))?
                .len(),
            1
        );

        drop(gate);
        assert!(cache.upgrade().is_some());
        drop(cloned);
        assert!(cache.upgrade().is_none());
        Ok(())
    }
}
