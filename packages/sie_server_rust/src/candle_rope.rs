//! Shared RoPE helpers for Candle model kernels.

use candle::{DType, Device, Result, Tensor};
#[cfg(feature = "cuda")]
use candle_rotary::apply_rotary_inplace;
use serde::{de, Deserialize, Deserializer};

#[derive(Debug, Clone)]
pub enum RopeScaling {
    Llama3 {
        #[allow(dead_code)]
        rope_type: String,
        factor: f32,
        high_freq_factor: f32,
        low_freq_factor: f32,
        original_max_position_embeddings: usize,
    },
    Ntk {
        #[allow(dead_code)]
        rope_type: String,
        factor: f32,
    },
}

#[derive(Deserialize)]
struct RawRopeScaling {
    #[serde(alias = "type")]
    rope_type: String,
    factor: f32,
    #[serde(default)]
    high_freq_factor: Option<f32>,
    #[serde(default)]
    low_freq_factor: Option<f32>,
    #[serde(default)]
    original_max_position_embeddings: Option<usize>,
}

impl<'de> Deserialize<'de> for RopeScaling {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawRopeScaling::deserialize(deserializer)?;
        let rope_type = raw.rope_type;
        if rope_type == "llama3" {
            Ok(Self::Llama3 {
                rope_type,
                factor: raw.factor,
                high_freq_factor: raw
                    .high_freq_factor
                    .ok_or_else(|| de::Error::missing_field("high_freq_factor"))?,
                low_freq_factor: raw
                    .low_freq_factor
                    .ok_or_else(|| de::Error::missing_field("low_freq_factor"))?,
                original_max_position_embeddings: raw
                    .original_max_position_embeddings
                    .ok_or_else(|| de::Error::missing_field("original_max_position_embeddings"))?,
            })
        } else if rope_type == "ntk" {
            Ok(Self::Ntk {
                rope_type,
                factor: raw.factor,
            })
        } else {
            Err(de::Error::unknown_variant(&rope_type, &["llama3", "ntk"]))
        }
    }
}

pub fn inv_freqs(
    dim: usize,
    base: f32,
    device: &Device,
    rope_scaling: Option<&RopeScaling>,
) -> Result<Tensor> {
    if let Some(rope_scaling) = rope_scaling {
        match rope_scaling {
            RopeScaling::Llama3 {
                factor,
                high_freq_factor,
                low_freq_factor,
                original_max_position_embeddings,
                ..
            } => {
                let old_context_len = *original_max_position_embeddings as f32;
                let low_freq_wavelen = old_context_len / *low_freq_factor;
                let high_freq_wavelen = old_context_len / *high_freq_factor;
                let inv_freq = (0..dim)
                    .step_by(2)
                    .map(|idx| {
                        let freq_idx = idx as f32 / dim as f32;
                        let inv_freq_base = 1.0 / base.powf(freq_idx);
                        let wavelen = 2.0 * std::f32::consts::PI / inv_freq_base;
                        if wavelen < high_freq_wavelen {
                            inv_freq_base
                        } else if wavelen > low_freq_wavelen {
                            inv_freq_base / *factor
                        } else {
                            let smooth_factor = (old_context_len / wavelen - *low_freq_factor)
                                / (*high_freq_factor - *low_freq_factor);
                            let inv_freq_llama = inv_freq_base / *factor;
                            (1.0 - smooth_factor) * inv_freq_llama + smooth_factor * inv_freq_base
                        }
                    })
                    .collect::<Vec<_>>();
                return Tensor::from_vec(inv_freq, (1, dim / 2), device);
            }
            RopeScaling::Ntk { factor, .. } => {
                let inv_freqs = base_inv_freqs(dim, base * *factor, device)?;
                let scale = factor.powf(2.0 / dim as f32) as f64;
                return inv_freqs / scale;
            }
        }
    }
    base_inv_freqs(dim, base, device)
}

fn base_inv_freqs(dim: usize, base: f32, device: &Device) -> Result<Tensor> {
    let inv_freq = (0..dim)
        .step_by(2)
        .map(|idx| 1f32 / base.powf(idx as f32 / dim as f32))
        .collect::<Vec<_>>();
    Tensor::from_vec(inv_freq, (1, dim / 2), device)
}

pub fn cos_sin(
    length: usize,
    inv_freqs: &Tensor,
    dtype: DType,
    repeat_freqs: bool,
) -> Result<(Tensor, Tensor)> {
    let positions = Tensor::arange(0u32, length as u32, inv_freqs.device())?
        .to_dtype(DType::F32)?
        .reshape((length, 1))?;
    let mut freqs = positions.matmul(inv_freqs)?;
    if repeat_freqs {
        freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
    }
    Ok((freqs.cos()?.to_dtype(dtype)?, freqs.sin()?.to_dtype(dtype)?))
}

#[cfg(feature = "cuda")]
pub fn apply_rotary_packed_inplace(
    query: &Tensor,
    key: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<()> {
    apply_rotary_inplace(query, key, cos, sin, true)
}

#[cfg(not(feature = "cuda"))]
pub fn apply_rotary_packed_inplace(
    _query: &Tensor,
    _key: &Tensor,
    _cos: &Tensor,
    _sin: &Tensor,
) -> Result<()> {
    candle::bail!("packed RoPE rotary requires the cuda feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_scaling_accepts_supported_ntk_type() {
        let scaling: RopeScaling = serde_json::from_str(r#"{"type":"ntk","factor":8.0}"#).unwrap();
        match scaling {
            RopeScaling::Ntk { factor, .. } => assert_eq!(factor, 8.0),
            other => panic!("expected NTK RoPE scaling, got {other:?}"),
        }
    }

    #[test]
    fn rope_scaling_accepts_supported_llama3_type() {
        let scaling: RopeScaling = serde_json::from_str(
            r#"{
                "rope_type":"llama3",
                "factor":8.0,
                "high_freq_factor":4.0,
                "low_freq_factor":1.0,
                "original_max_position_embeddings":8192
            }"#,
        )
        .unwrap();
        match scaling {
            RopeScaling::Llama3 {
                factor,
                high_freq_factor,
                low_freq_factor,
                original_max_position_embeddings,
                ..
            } => {
                assert_eq!(factor, 8.0);
                assert_eq!(high_freq_factor, 4.0);
                assert_eq!(low_freq_factor, 1.0);
                assert_eq!(original_max_position_embeddings, 8192);
            }
            other => panic!("expected Llama 3 RoPE scaling, got {other:?}"),
        }
    }

    #[test]
    fn rope_scaling_rejects_unsupported_type() {
        let err = serde_json::from_str::<RopeScaling>(r#"{"type":"linear","factor":8.0}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("unknown variant"));
        assert!(err.contains("linear"));
    }
}
