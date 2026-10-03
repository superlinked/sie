//! Small optimized layer wrappers used by the custom Candle XLM-R path.

#[cfg(feature = "cuda")]
use self::cublaslt::CublasLtExt;
#[cfg(feature = "cuda")]
use std::sync::{Once, OnceLock};

use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{init, ops, Init, VarBuilder};
use serde::Deserialize;
#[cfg(feature = "cuda")]
use tracing::warn;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum HiddenAct {
    #[serde(alias = "gelu_new", alias = "gelu_pytorch_tanh")]
    Gelu,
    Relu,
    Silu,
    Swiglu,
    Tanh,
}

impl HiddenAct {
    pub(crate) fn forward(self, x: &Tensor) -> Result<Tensor> {
        match self {
            // Matches TEI's fast path: cuBLASLt supports the tanh GELU epilogue.
            Self::Gelu => x.gelu(),
            Self::Relu => x.relu(),
            Self::Silu => x.silu(),
            Self::Swiglu => ops::swiglu(x),
            Self::Tanh => x.tanh(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FastLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    act: Option<HiddenAct>,
}

impl FastLinear {
    pub(crate) fn new(weight: Tensor, bias: Option<Tensor>, act: Option<HiddenAct>) -> Self {
        Self { weight, bias, act }
    }

    pub(crate) fn load(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        act: Option<HiddenAct>,
    ) -> Result<Self> {
        let weight =
            vb.get_with_hints((out_dim, in_dim), "weight", init::DEFAULT_KAIMING_NORMAL)?;
        let bound = 1. / (in_dim as f64).sqrt();
        let bias = vb.get_with_hints(
            out_dim,
            "bias",
            Init::Uniform {
                lo: -bound,
                up: bound,
            },
        )?;
        Ok(Self {
            weight,
            bias: Some(bias),
            act,
        })
    }

    pub(crate) fn load_no_bias(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        act: Option<HiddenAct>,
    ) -> Result<Self> {
        let weight =
            vb.get_with_hints((out_dim, in_dim), "weight", init::DEFAULT_KAIMING_NORMAL)?;
        Ok(Self::new(weight, None, act))
    }

    pub(crate) fn load_qkv(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let query = Self::load(in_dim, out_dim, vb.pp("query"), None)?;
        let key = Self::load(in_dim, out_dim, vb.pp("key"), None)?;
        let value = Self::load(in_dim, out_dim, vb.pp("value"), None)?;
        let weight = Tensor::cat(&[&query.weight, &key.weight, &value.weight], 0)?;
        let bias = match (&query.bias, &key.bias, &value.bias) {
            (Some(query), Some(key), Some(value)) => Some(Tensor::cat(&[query, key, value], 0)?),
            _ => None,
        };
        Ok(Self::new(weight, bias, None))
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(output) = self.forward_cublaslt(x)? {
            return Ok(output);
        }

        let x = match *x.dims() {
            [b1, b2, m, k] if x.is_contiguous() => {
                let w = self.weight.t()?;
                x.reshape((b1 * b2 * m, k))?
                    .matmul(&w)?
                    .reshape((b1, b2, m, ()))?
            }
            [b1, b2, _, _] => {
                let w = self.weight.broadcast_left((b1, b2))?.t()?;
                x.matmul(&w)?
            }
            [bsize, m, k] if x.is_contiguous() => {
                let w = self.weight.t()?;
                x.reshape((bsize * m, k))?
                    .matmul(&w)?
                    .reshape((bsize, m, ()))?
            }
            [bsize, _, _] => {
                let w = self.weight.broadcast_left(bsize)?.t()?;
                x.matmul(&w)?
            }
            _ => {
                let w = self.weight.t()?;
                x.matmul(&w)?
            }
        };
        let x = match &self.bias {
            Some(bias) => x.broadcast_add(bias)?,
            None => x,
        };
        self.forward_activation(x)
    }

    fn forward_activation(&self, x: Tensor) -> Result<Tensor> {
        match self.act {
            Some(act) => act.forward(&x),
            None => Ok(x),
        }
    }

    #[cfg(feature = "cuda")]
    fn forward_cublaslt(&self, x: &Tensor) -> Result<Option<Tensor>> {
        if !matches!(x.device(), Device::Cuda(_)) {
            return Ok(None);
        }
        if !x.is_contiguous() {
            return Ok(None);
        }
        if !cublaslt_enabled() {
            return Ok(None);
        }
        let Some(cublaslt) = cublaslt::get(x.device()) else {
            return Ok(None);
        };
        let (fused_act, activation_fused) = cublaslt_activation(self.act);
        let mut output = match *x.dims() {
            // Broadcasting the weight creates a stride-zero view that cuBLASLt rejects.
            // Flatten the batch into the row dimension and reuse the contiguous rank-two path.
            [bsize, rows, columns] => {
                let flattened_rows = bsize.checked_mul(rows).ok_or_else(|| {
                    candle::Error::msg("FastLinear rank-three row count overflow")
                })?;
                cublaslt
                    .matmul(
                        &self.weight,
                        &x.reshape((flattened_rows, columns))?,
                        self.bias.as_ref(),
                        fused_act,
                    )?
                    .reshape((bsize, rows, ()))?
            }
            [_, _] => cublaslt.matmul(&self.weight, x, self.bias.as_ref(), fused_act)?,
            _ => return Ok(None),
        };
        if !activation_fused {
            output = self.forward_activation(output)?;
        }
        Ok(Some(output))
    }

    #[cfg(not(feature = "cuda"))]
    fn forward_cublaslt(&self, _x: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }

    pub(crate) fn backend(&self) -> &'static str {
        if self.uses_cublaslt() {
            "cuda_cublaslt"
        } else {
            "candle_matmul"
        }
    }

    pub(crate) fn uses_cublaslt(&self) -> bool {
        cublaslt_available(self.weight.device())
    }

    pub(crate) fn activation_kernel_fused(&self) -> bool {
        self.uses_cublaslt() && matches!(self.act, Some(HiddenAct::Gelu | HiddenAct::Relu))
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FastLayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    epsilon: f32,
}

impl FastLayerNorm {
    pub(crate) fn load(vb: VarBuilder, hidden_size: usize, epsilon: f64) -> Result<Self> {
        let weight = vb.get_with_hints(hidden_size, "weight", Init::Const(1.))?;
        let bias = vb.get_with_hints(hidden_size, "bias", Init::Const(0.))?;
        Ok(Self {
            weight,
            bias: Some(bias),
            epsilon: epsilon as f32,
        })
    }

    pub(crate) fn load_no_bias(vb: VarBuilder, hidden_size: usize, epsilon: f64) -> Result<Self> {
        let weight = vb.get_with_hints(hidden_size, "weight", Init::Const(1.))?;
        Ok(Self {
            weight,
            bias: None,
            epsilon: epsilon as f32,
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
    ) -> Result<Tensor> {
        if let Some(output) = self.forward_cuda(hidden_states, residual)? {
            return Ok(output);
        }

        let hidden_states = match residual {
            Some(residual) => hidden_states.add(residual)?,
            None => hidden_states.clone(),
        };
        self.forward_fallback(&hidden_states)
    }

    fn forward_fallback(&self, hidden_states: &Tensor) -> Result<Tensor> {
        if hidden_states.is_contiguous() {
            if let Some(bias) = &self.bias {
                return ops::layer_norm(hidden_states, &self.weight, bias, self.epsilon);
            }
        }

        let hidden_states_dtype = hidden_states.dtype();
        let internal_dtype = match hidden_states_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            dtype => dtype,
        };
        let hidden_size = hidden_states.dim(D::Minus1)?;
        let hidden_states = hidden_states.to_dtype(internal_dtype)?;
        let mean_hidden_states = (hidden_states.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let hidden_states = hidden_states.broadcast_sub(&mean_hidden_states)?;
        let norm_hidden_states =
            (hidden_states.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let hidden_states = hidden_states
            .broadcast_div(&(norm_hidden_states + self.epsilon as f64)?.sqrt()?)?
            .to_dtype(hidden_states_dtype)?
            .broadcast_mul(&self.weight)?;
        match &self.bias {
            Some(bias) => hidden_states.broadcast_add(bias),
            None => Ok(hidden_states),
        }
    }

    #[cfg(feature = "cuda")]
    fn forward_cuda(
        &self,
        hidden_states: &Tensor,
        residual: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        if !matches!(hidden_states.device(), Device::Cuda(_)) {
            return Ok(None);
        }
        let original_shape = hidden_states.shape();
        let hidden_size = hidden_states.dim(D::Minus1)?;
        if hidden_size % 8 != 0 || hidden_size > 8192 {
            return Ok(None);
        }
        let hidden_states = hidden_states.flatten_to(D::Minus2)?;
        let output_only = layer_norm_output_only_enabled();
        let output = match residual {
            Some(residual) => {
                let residual = residual.flatten_to(D::Minus2)?;
                if output_only {
                    candle_layer_norm::fused_add_layer_norm_output_only(
                        &hidden_states,
                        &residual,
                        &self.weight,
                        self.bias.as_ref(),
                        self.epsilon,
                    )?
                } else {
                    candle_layer_norm::fused_add_layer_norm(
                        &hidden_states,
                        &residual,
                        &self.weight,
                        self.bias.as_ref(),
                        self.epsilon,
                    )?
                    .0
                }
            }
            None => {
                if output_only {
                    candle_layer_norm::layer_norm_output_only(
                        &hidden_states,
                        &self.weight,
                        self.bias.as_ref(),
                        self.epsilon,
                    )?
                } else {
                    candle_layer_norm::layer_norm(
                        &hidden_states,
                        &self.weight,
                        self.bias.as_ref(),
                        self.epsilon,
                    )?
                }
            }
        };
        Ok(Some(output.reshape(original_shape)?))
    }

    #[cfg(not(feature = "cuda"))]
    fn forward_cuda(
        &self,
        _hidden_states: &Tensor,
        _residual: Option<&Tensor>,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }

    pub(crate) fn backend(&self) -> &'static str {
        #[cfg(feature = "cuda")]
        if matches!(self.weight.device(), Device::Cuda(_)) {
            return if layer_norm_output_only_enabled() {
                "cuda_fused_layer_norm_output_only"
            } else {
                "cuda_fused_layer_norm"
            };
        }
        "candle_layer_norm"
    }
}

pub(crate) fn cublaslt_available(device: &Device) -> bool {
    matches!(device, Device::Cuda(_)) && cublaslt_available_inner(device)
}

pub(crate) fn index_select(tensor: &Tensor, indices: &Tensor, dim: usize) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if matches!(tensor.device(), Device::Cuda(_)) && matches!(indices.device(), Device::Cuda(_))
        {
            candle_index_select_cu::index_select(tensor, indices, dim)
        } else {
            tensor.index_select(indices, dim)
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        tensor.index_select(indices, dim)
    }
}

#[cfg(feature = "cuda")]
fn cublaslt_available_inner(device: &Device) -> bool {
    cublaslt_enabled() && cublaslt::get(device).is_some()
}

#[cfg(not(feature = "cuda"))]
fn cublaslt_available_inner(_device: &Device) -> bool {
    false
}

#[cfg(feature = "cuda")]
fn cublaslt_activation(act: Option<HiddenAct>) -> (Option<candle_cublaslt::Activation>, bool) {
    match act {
        Some(HiddenAct::Gelu) => (Some(candle_cublaslt::Activation::Gelu), true),
        Some(HiddenAct::Relu) => (Some(candle_cublaslt::Activation::Relu), true),
        _ => (None, act.is_none()),
    }
}

#[cfg(feature = "cuda")]
fn cublaslt_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let disabled = env_bool_any(&["SIE_CANDLE_DISABLE_CUBLASLT"], false);
        let enabled = env_bool_any(&["SIE_CANDLE_ENABLE_CUBLASLT"], true);
        enabled && !disabled
    })
}

#[cfg(feature = "cuda")]
fn layer_norm_output_only_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let disabled = std::env::var("SIE_CANDLE_DISABLE_LAYER_NORM_OUTPUT_ONLY").ok();
        layer_norm_output_only_enabled_from(disabled.as_deref())
    })
}

#[cfg(any(feature = "cuda", test))]
fn layer_norm_output_only_enabled_from(disabled: Option<&str>) -> bool {
    !env_bool_value(disabled, false)
}

#[cfg(feature = "cuda")]
fn env_bool_any(names: &[&str], default: bool) -> bool {
    let value = names.iter().find_map(|name| std::env::var(name).ok());
    env_bool_value(value.as_deref(), default)
}

#[cfg(any(feature = "cuda", test))]
fn env_bool_value(value: Option<&str>, default: bool) -> bool {
    value
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

#[cfg(feature = "cuda")]
mod cublaslt {
    use super::*;
    use candle_cublaslt::{fused_matmul, Activation, CublasLt};

    static INIT: Once = Once::new();
    static mut CUBLASLT: Option<CublasLt> = None;

    pub(super) fn get(device: &Device) -> Option<CublasLt> {
        unsafe {
            INIT.call_once(|| {
                CUBLASLT = match CublasLt::new(device) {
                    Ok(cublaslt) => Some(cublaslt),
                    Err(error) => {
                        warn!(?error, "failed to initialize Candle cuBLASLt wrapper");
                        None
                    }
                };
            });
            #[allow(static_mut_refs)]
            CUBLASLT.clone()
        }
    }

    pub(super) trait CublasLtExt {
        fn matmul(
            &self,
            weight: &Tensor,
            input: &Tensor,
            bias: Option<&Tensor>,
            act: Option<Activation>,
        ) -> Result<Tensor>;
    }

    impl CublasLtExt for CublasLt {
        fn matmul(
            &self,
            weight: &Tensor,
            input: &Tensor,
            bias: Option<&Tensor>,
            act: Option<Activation>,
        ) -> Result<Tensor> {
            fused_matmul(weight, input, None, None, None, bias, act, self.clone())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn hidden_act_gelu_uses_tanh_approximation() -> Result<()> {
        let xs = Tensor::new(&[-1f32, 0., 1.], &Device::Cpu)?;
        let fast = HiddenAct::Gelu.forward(&xs)?;
        let expected = xs.gelu()?;
        assert_eq!(fast.to_vec1::<f32>()?, expected.to_vec1::<f32>()?);
        Ok(())
    }

    #[test]
    fn fast_linear_matches_stock_linear_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device)?;
        let bias = Tensor::new(&[0.5f32, -0.5, 1.], &device)?;
        let layer = FastLinear {
            weight: weight.clone(),
            bias: Some(bias.clone()),
            act: None,
        };
        let xs = Tensor::new(&[[10f32, 100.], [1., 2.]], &device)?;
        let actual = layer.forward(&xs)?;
        let expected = xs.matmul(&weight.t()?)?.broadcast_add(&bias)?;
        assert_close_2d(&actual.to_vec2::<f32>()?, &expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn fast_linear_rank_three_matches_flattened_projection_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device)?;
        let bias = Tensor::new(&[0.5f32, -0.5, 1.], &device)?;
        let layer = FastLinear::new(weight.clone(), Some(bias.clone()), None);
        let xs = Tensor::new(
            &[
                [[1f32, 2.], [3., 4.], [5., 6.]],
                [[7., 8.], [9., 10.], [11., 12.]],
            ],
            &device,
        )?;

        let actual = layer.forward(&xs)?;
        let expected = xs
            .reshape((6, 2))?
            .matmul(&weight.t()?)?
            .broadcast_add(&bias)?
            .reshape((2, 3, 3))?;

        assert_eq!(actual.dims(), &[2, 3, 3]);
        assert_close_3d(&actual.to_vec3::<f32>()?, &expected.to_vec3::<f32>()?);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn fast_linear_rank_three_cublaslt_matches_flattened_projection_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let weight_values = [[1f32, 2.], [3., 4.], [5., 6.]];
        let bias_values = [0.5f32, -0.5, 1.];
        let input_values = [
            [[1f32, 2.], [3., 4.], [5., 6.]],
            [[7., 8.], [9., 10.], [11., 12.]],
        ];

        for (dtype, tolerance) in [
            (DType::F32, 1e-5f32),
            (DType::F16, 0.02f32),
            (DType::BF16, 0.15f32),
        ] {
            let weight = Tensor::new(&weight_values, &device)?.to_dtype(dtype)?;
            let bias = Tensor::new(&bias_values, &device)?.to_dtype(dtype)?;
            let xs = Tensor::new(&input_values, &device)?.to_dtype(dtype)?;
            let layer = FastLinear::new(weight.clone(), Some(bias.clone()), Some(HiddenAct::Silu));

            let actual = layer
                .forward_cublaslt(&xs)?
                .expect("cuBLASLt should handle contiguous rank-three CUDA input");
            let expected = xs
                .reshape((6, 2))?
                .matmul(&weight.t()?)?
                .broadcast_add(&bias)?
                .silu()?
                .reshape((2, 3, 3))?;

            assert_eq!(actual.dims(), &[2, 3, 3]);
            let actual = actual.to_dtype(DType::F32)?.to_vec3::<f32>()?;
            let expected = expected.to_dtype(DType::F32)?.to_vec3::<f32>()?;
            assert_close_3d_with_tolerance(&actual, &expected, tolerance);
        }
        Ok(())
    }

    #[test]
    fn fast_linear_qkv_projection_preserves_projection_order() -> Result<()> {
        let device = Device::Cpu;
        let query = Tensor::new(&[[1f32, 2.]], &device)?;
        let key = Tensor::new(&[[3f32, 4.]], &device)?;
        let value = Tensor::new(&[[5f32, 6.]], &device)?;
        let qkv_weight = Tensor::cat(&[&query, &key, &value], 0)?;
        let qkv_bias = Tensor::new(&[0f32, 10., 100.], &device)?;
        let layer = FastLinear::new(qkv_weight, Some(qkv_bias), None);
        let output = layer.forward(&Tensor::new(&[[7f32, 11.]], &device)?)?;
        assert_eq!(output.to_vec2::<f32>()?, vec![vec![29., 75., 201.]]);
        Ok(())
    }

    #[test]
    fn fast_layer_norm_fuses_residual_semantics_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let hidden_states = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?;
        let residual = Tensor::new(&[[0.5f32, -1., 2.], [1., 0., -2.]], &device)?;
        let weight = Tensor::new(&[1f32, 0.5, 2.], &device)?;
        let bias = Tensor::new(&[0.1f32, -0.2, 0.3], &device)?;
        let layer_norm = FastLayerNorm {
            weight: weight.clone(),
            bias: Some(bias.clone()),
            epsilon: 1e-5,
        };
        let actual = layer_norm.forward(&hidden_states, Some(&residual))?;
        let expected = ops::layer_norm(&(hidden_states + residual)?, &weight, &bias, 1e-5)?;
        assert_close_2d(&actual.to_vec2::<f32>()?, &expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn layer_norm_output_only_kill_switch_parses_boolean_values() {
        assert!(layer_norm_output_only_enabled_from(None));
        assert!(!layer_norm_output_only_enabled_from(Some("1")));
        assert!(!layer_norm_output_only_enabled_from(Some(" YeS ")));
        assert!(layer_norm_output_only_enabled_from(Some("0")));
        assert!(layer_norm_output_only_enabled_from(Some("false")));
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn layer_norm_output_only_matches_reference_on_default_cuda_stream() -> Result<()> {
        assert_layer_norm_output_only_matches_reference(&Device::new_cuda(0)?)
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn layer_norm_output_only_matches_reference_on_non_default_cuda_stream() -> Result<()> {
        assert_layer_norm_output_only_matches_reference(&Device::new_cuda_with_stream(0)?)
    }

    #[cfg(feature = "cuda")]
    fn assert_layer_norm_output_only_matches_reference(device: &Device) -> Result<()> {
        let input_values = (0..24)
            .map(|value| (value as f32 - 12.) / 7.)
            .collect::<Vec<_>>();
        let residual_values = (0..24)
            .map(|value| (11. - value as f32) / 13.)
            .collect::<Vec<_>>();
        let gamma_values = (0..8)
            .map(|value| 0.75 + value as f32 / 16.)
            .collect::<Vec<_>>();
        let beta_values = (0..8)
            .map(|value| (value as f32 - 4.) / 20.)
            .collect::<Vec<_>>();

        for (dtype, tolerance) in [
            (DType::F32, 1e-3f32),
            (DType::F16, 0.02f32),
            (DType::BF16, 0.15f32),
        ] {
            let input = Tensor::from_vec(input_values.clone(), (3, 8), device)?.to_dtype(dtype)?;
            let residual =
                Tensor::from_vec(residual_values.clone(), (3, 8), device)?.to_dtype(dtype)?;
            let gamma = Tensor::from_vec(gamma_values.clone(), 8, device)?.to_dtype(dtype)?;
            let beta = Tensor::from_vec(beta_values.clone(), 8, device)?.to_dtype(dtype)?;

            let expected = ops::layer_norm(&input, &gamma, &beta, 1e-5)?;
            let output_only =
                candle_layer_norm::layer_norm_output_only(&input, &gamma, Some(&beta), 1e-5)?;
            assert_close_2d_with_tolerance(
                &output_only.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                &expected.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                tolerance,
            );

            let expected =
                ops::layer_norm(&(input.clone() + residual.clone())?, &gamma, &beta, 1e-5)?;
            let output_only = candle_layer_norm::fused_add_layer_norm_output_only(
                &input,
                &residual,
                &gamma,
                Some(&beta),
                1e-5,
            )?;
            assert_close_2d_with_tolerance(
                &output_only.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                &expected.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                tolerance,
            );
        }
        Ok(())
    }

    #[test]
    fn fast_linear_no_bias_loader_matches_stock_linear_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device)?;
        let vb = VarBuilder::from_tensors(
            std::collections::HashMap::from([("weight".to_string(), weight.clone())]),
            DType::F32,
            &device,
        );
        let layer = FastLinear::load_no_bias(2, 3, vb, None)?;
        let xs = Tensor::new(&[[10f32, 100.], [1., 2.]], &device)?;

        let actual = layer.forward(&xs)?;
        let expected = xs.matmul(&weight.t()?)?;

        assert!(layer.bias.is_none());
        assert_close_2d(&actual.to_vec2::<f32>()?, &expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn fast_layer_norm_no_bias_loader_matches_stock_layer_norm_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let weight = Tensor::new(&[1f32, 0.5, 2.], &device)?;
        let vb = VarBuilder::from_tensors(
            std::collections::HashMap::from([("weight".to_string(), weight.clone())]),
            DType::F32,
            &device,
        );
        let layer_norm = FastLayerNorm::load_no_bias(vb, 3, 1e-5)?;
        let xs = Tensor::new(&[[1f32, 2., 3.], [4., 5., 7.]], &device)?;

        let actual = layer_norm.forward(&xs, None)?;
        let expected = xs.apply(&candle_nn::LayerNorm::new_no_bias(weight, 1e-5))?;

        assert!(layer_norm.bias.is_none());
        assert_eq!(layer_norm.backend(), "candle_layer_norm");
        assert_close_2d(&actual.to_vec2::<f32>()?, &expected.to_vec2::<f32>()?);
        Ok(())
    }

    fn assert_close_2d(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
        assert_close_2d_with_tolerance(actual, expected, 1e-5);
    }

    fn assert_close_2d_with_tolerance(actual: &[Vec<f32>], expected: &[Vec<f32>], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (&actual, &expected) in actual_row.iter().zip(expected_row) {
                assert!(
                    (actual - expected).abs() < tolerance,
                    "actual={actual} expected={expected} tolerance={tolerance}"
                );
            }
        }
    }

    fn assert_close_3d(actual: &[Vec<Vec<f32>>], expected: &[Vec<Vec<f32>>]) {
        assert_close_3d_with_tolerance(actual, expected, 1e-5);
    }

    fn assert_close_3d_with_tolerance(
        actual: &[Vec<Vec<f32>>],
        expected: &[Vec<Vec<f32>>],
        tolerance: f32,
    ) {
        assert_eq!(actual.len(), expected.len());
        for (actual_matrix, expected_matrix) in actual.iter().zip(expected) {
            assert_eq!(actual_matrix.len(), expected_matrix.len());
            for (actual_row, expected_row) in actual_matrix.iter().zip(expected_matrix) {
                assert_eq!(actual_row.len(), expected_row.len());
                for (&actual, &expected) in actual_row.iter().zip(expected_row) {
                    assert!(
                        (actual - expected).abs() <= tolerance,
                        "actual={actual} expected={expected} tolerance={tolerance}"
                    );
                }
            }
        }
    }
}
