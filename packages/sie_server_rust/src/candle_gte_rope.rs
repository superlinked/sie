//! GTE/NewModel encoder with packed RoPE FlashAttention for Candle.

#[cfg(feature = "cuda")]
use std::sync::OnceLock;
use std::time::Instant;

use crate::candle_layers::{index_select, FastLayerNorm, FastLinear, HiddenAct};
use crate::candle_rope::{
    apply_rotary_packed_inplace, cos_sin, inv_freqs as build_inv_freqs, RopeScaling,
};
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub layer_norm_type: Option<String>,
    #[serde(default)]
    pub pack_qkv: Option<bool>,
    #[serde(default)]
    pub position_embedding_type: Option<String>,
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    #[serde(default)]
    pub logn_attention_scale: bool,
    #[serde(default)]
    pub logn_attention_clip1: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GteRopeForwardProfile {
    pub total_tokens: usize,
    pub max_seqlen: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub attention_heads: usize,
    pub attention_head_size: usize,
    pub linear_gflops: f64,
    pub embedding_ms: f64,
    pub rope_select_ms: f64,
    pub attention_ms: f64,
    pub attention_qkv_ms: f64,
    pub attention_rotary_ms: f64,
    pub attention_flash_ms: f64,
    pub attention_output_dense_ms: f64,
    pub attention_output_layernorm_ms: f64,
    pub ffn_ms: f64,
    pub ffn_up_gate_ms: f64,
    pub ffn_activation_ms: f64,
    pub ffn_down_ms: f64,
    pub ffn_output_layernorm_ms: f64,
    pub layers: usize,
}

struct GteRopeEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Option<Embedding>,
    layer_norm: FastLayerNorm,
}

impl GteRopeEmbeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let token_type_embeddings = if config.type_vocab_size > 0 {
            Some(embedding(
                config.type_vocab_size,
                config.hidden_size,
                vb.pp("token_type_embeddings"),
            )?)
        } else {
            None
        };
        let layer_norm = FastLayerNorm::load(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps,
        )?;
        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }

    fn forward_packed(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let word_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self
            .token_type_embeddings
            .as_ref()
            .map(|embeddings| embeddings.forward(token_type_ids))
            .transpose()?;
        self.layer_norm
            .forward(&word_embeddings, token_type_embeddings.as_ref())
    }
}

struct GteRopeAttention {
    qkv: FastLinear,
    output: FastLinear,
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
}

impl GteRopeAttention {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;
        let qkv_weight = vb
            .pp("qkv_proj")
            .get((config.hidden_size * 3, config.hidden_size), "weight")?;
        let qkv_bias = vb.pp("qkv_proj").get(config.hidden_size * 3, "bias")?;
        let output_weight = vb
            .pp("o_proj")
            .get((config.hidden_size, config.hidden_size), "weight")?;
        let output_bias = vb.pp("o_proj").get(config.hidden_size, "bias")?;

        Ok(Self {
            qkv: FastLinear::new(qkv_weight, Some(qkv_bias), None),
            output: FastLinear::new(output_weight, Some(output_bias), None),
            num_attention_heads,
            attention_head_size,
            all_head_size,
        })
    }

    fn forward_packed(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let total_tokens = hidden_states.dim(0)?;
        let qkv = self.qkv.forward(hidden_states)?;
        let qkv = qkv.reshape((
            total_tokens,
            self.num_attention_heads * 3,
            self.attention_head_size,
        ))?;

        let query = qkv.narrow(1, 0, self.num_attention_heads)?;
        let key = qkv.narrow(1, self.num_attention_heads, self.num_attention_heads)?;
        let value = qkv.narrow(1, self.num_attention_heads * 2, self.num_attention_heads)?;

        apply_rotary_packed_inplace(&query, &key, cos, sin)?;
        let attention_output = flash_attn_varlen(
            &query,
            &key,
            &value,
            seqlens,
            max_seqlen,
            self.attention_head_size,
        )?;
        let attention_output = attention_output.reshape((total_tokens, self.all_head_size))?;
        self.output.forward(&attention_output)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_packed_profiled(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_seqlen: usize,
        profile: &mut GteRopeForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        let total_tokens = hidden_states.dim(0)?;

        synchronize_if(hidden_states.device(), sync_timings)?;
        let qkv_start = Instant::now();
        let qkv = self.qkv.forward(hidden_states)?;
        let qkv = qkv.reshape((
            total_tokens,
            self.num_attention_heads * 3,
            self.attention_head_size,
        ))?;
        let query = qkv.narrow(1, 0, self.num_attention_heads)?;
        let key = qkv.narrow(1, self.num_attention_heads, self.num_attention_heads)?;
        let value = qkv.narrow(1, self.num_attention_heads * 2, self.num_attention_heads)?;
        synchronize_if(query.device(), sync_timings)?;
        profile.attention_qkv_ms += elapsed_ms(qkv_start);

        let rotary_start = Instant::now();
        apply_rotary_packed_inplace(&query, &key, cos, sin)?;
        synchronize_if(query.device(), sync_timings)?;
        profile.attention_rotary_ms += elapsed_ms(rotary_start);

        let flash_start = Instant::now();
        let attention_output = flash_attn_varlen(
            &query,
            &key,
            &value,
            seqlens,
            max_seqlen,
            self.attention_head_size,
        )?;
        synchronize_if(attention_output.device(), sync_timings)?;
        profile.attention_flash_ms += elapsed_ms(flash_start);

        let dense_start = Instant::now();
        let attention_output = attention_output.reshape((total_tokens, self.all_head_size))?;
        let attention_output = self.output.forward(&attention_output)?;
        synchronize_if(attention_output.device(), sync_timings)?;
        profile.attention_output_dense_ms += elapsed_ms(dense_start);
        Ok(attention_output)
    }
}

struct GteRopeMlp {
    up_gate: FastLinear,
    down: FastLinear,
    act: HiddenAct,
    intermediate_size: usize,
}

impl GteRopeMlp {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let up_gate_weight = vb
            .pp("up_gate_proj")
            .get((config.intermediate_size * 2, config.hidden_size), "weight")?;
        let down_weight = vb
            .pp("down_proj")
            .get((config.hidden_size, config.intermediate_size), "weight")?;
        let down_bias = vb.pp("down_proj").get(config.hidden_size, "bias")?;
        Ok(Self {
            up_gate: FastLinear::new(up_gate_weight, None, None),
            down: FastLinear::new(down_weight, Some(down_bias), None),
            act: config.hidden_act,
            intermediate_size: config.intermediate_size,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let up_gate = self.up_gate.forward(hidden_states)?;
        let gated = self.gated_activation(&up_gate)?;
        self.down.forward(&gated)
    }

    fn forward_profiled(
        &self,
        hidden_states: &Tensor,
        profile: &mut GteRopeForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        synchronize_if(hidden_states.device(), sync_timings)?;
        let up_gate_start = Instant::now();
        let up_gate = self.up_gate.forward(hidden_states)?;
        synchronize_if(up_gate.device(), sync_timings)?;
        profile.ffn_up_gate_ms += elapsed_ms(up_gate_start);

        let activation_start = Instant::now();
        let gated = self.gated_activation(&up_gate)?;
        synchronize_if(gated.device(), sync_timings)?;
        profile.ffn_activation_ms += elapsed_ms(activation_start);

        let down_start = Instant::now();
        let output = self.down.forward(&gated)?;
        synchronize_if(output.device(), sync_timings)?;
        profile.ffn_down_ms += elapsed_ms(down_start);
        Ok(output)
    }

    fn gated_activation(&self, up_gate: &Tensor) -> Result<Tensor> {
        if self.can_use_fused_gated_activation(up_gate) {
            #[cfg(feature = "cuda")]
            {
                return candle_gated_activation::gelu_gate(up_gate, self.intermediate_size);
            }
        }

        let up = up_gate.narrow(D::Minus1, 0, self.intermediate_size)?;
        let gate = up_gate.narrow(D::Minus1, self.intermediate_size, self.intermediate_size)?;
        let gate = self.act.forward(&gate)?;
        gate * up
    }

    fn can_use_fused_gated_activation(&self, up_gate: &Tensor) -> bool {
        if !self.gated_activation_fused() {
            return false;
        }
        if !matches!(up_gate.device(), Device::Cuda(_)) || !up_gate.is_contiguous() {
            return false;
        }
        if !matches!(up_gate.dtype(), DType::F16 | DType::BF16 | DType::F32) {
            return false;
        }
        up_gate
            .dims()
            .last()
            .is_some_and(|last_dim| *last_dim == self.intermediate_size * 2)
    }

    fn gated_activation_fused(&self) -> bool {
        matches!(self.act, HiddenAct::Gelu) && fused_gated_gelu_enabled()
    }
}

struct GteRopeLayer {
    attention: GteRopeAttention,
    mlp: GteRopeMlp,
    attention_layer_norm: FastLayerNorm,
    mlp_layer_norm: FastLayerNorm,
}

impl GteRopeLayer {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let attention = GteRopeAttention::load(vb.pp("attention"), config)?;
        let mlp = GteRopeMlp::load(vb.pp("mlp"), config)?;
        let attention_layer_norm =
            FastLayerNorm::load(vb.pp("attn_ln"), config.hidden_size, config.layer_norm_eps)?;
        let mlp_layer_norm =
            FastLayerNorm::load(vb.pp("mlp_ln"), config.hidden_size, config.layer_norm_eps)?;
        Ok(Self {
            attention,
            mlp,
            attention_layer_norm,
            mlp_layer_norm,
        })
    }

    fn forward_packed(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let attention_output =
            self.attention
                .forward_packed(hidden_states, seqlens, cos, sin, max_seqlen)?;
        let attention_output = self
            .attention_layer_norm
            .forward(&attention_output, Some(hidden_states))?;
        let mlp_output = self.mlp.forward(&attention_output)?;
        self.mlp_layer_norm
            .forward(&mlp_output, Some(&attention_output))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_packed_profiled(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_seqlen: usize,
        profile: &mut GteRopeForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        synchronize_if(hidden_states.device(), sync_timings)?;
        let attention_start = Instant::now();
        let attention_output = self.attention.forward_packed_profiled(
            hidden_states,
            seqlens,
            cos,
            sin,
            max_seqlen,
            profile,
            sync_timings,
        )?;

        let layernorm_start = Instant::now();
        let attention_output = self
            .attention_layer_norm
            .forward(&attention_output, Some(hidden_states))?;
        synchronize_if(attention_output.device(), sync_timings)?;
        profile.attention_output_layernorm_ms += elapsed_ms(layernorm_start);
        profile.attention_ms += elapsed_ms(attention_start);

        let ffn_start = Instant::now();
        let mlp_output = self
            .mlp
            .forward_profiled(&attention_output, profile, sync_timings)?;

        let layernorm_start = Instant::now();
        let output = self
            .mlp_layer_norm
            .forward(&mlp_output, Some(&attention_output))?;
        synchronize_if(output.device(), sync_timings)?;
        profile.ffn_output_layernorm_ms += elapsed_ms(layernorm_start);
        profile.ffn_ms += elapsed_ms(ffn_start);
        profile.layers += 1;
        Ok(output)
    }
}

pub struct GteRopeModel {
    embeddings: GteRopeEmbeddings,
    layers: Vec<GteRopeLayer>,
    cos_cache: Tensor,
    sin_cache: Tensor,
    config: Config,
}

impl GteRopeModel {
    pub fn load(config: &Config, vb: VarBuilder) -> Result<Self> {
        if !matches!(vb.device(), Device::Cuda(_)) {
            candle::bail!("GTE-RoPE packed Candle path requires CUDA");
        }
        if vb.dtype() != DType::F16 {
            candle::bail!("GTE-RoPE packed Candle path requires float16 compute_precision");
        }
        if config.logn_attention_clip1 {
            candle::bail!("GTE-RoPE logn_attention_clip1 is not supported");
        }
        if config.logn_attention_scale {
            candle::bail!("GTE-RoPE logn_attention_scale is not supported");
        }
        if !matches!(
            config.position_embedding_type.as_deref(),
            Some(value) if value.eq_ignore_ascii_case("rope")
        ) {
            candle::bail!("GTE-RoPE requires position_embedding_type=rope");
        }
        if matches!(config.pack_qkv, Some(false)) {
            candle::bail!("GTE-RoPE requires packed qkv_proj weights");
        }
        if let Some(layer_norm_type) = config.layer_norm_type.as_deref() {
            if !layer_norm_type.eq_ignore_ascii_case("layer_norm") {
                candle::bail!("GTE-RoPE requires layer_norm_type=layer_norm");
            }
        }
        if config.num_attention_heads == 0 {
            candle::bail!("GTE-RoPE num_attention_heads must be greater than zero");
        }
        if !config
            .hidden_size
            .is_multiple_of(config.num_attention_heads)
        {
            candle::bail!(
                "GTE-RoPE hidden_size={} must be divisible by num_attention_heads={}",
                config.hidden_size,
                config.num_attention_heads
            );
        }

        let (embeddings, layers) = Self::inner_load(vb.pp("new"), config)
            .or_else(|_| Self::inner_load(vb.clone(), config))?;
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        if !attention_head_size.is_multiple_of(2) {
            candle::bail!("GTE-RoPE attention_head_size must be even");
        }
        let rope_theta = config
            .rope_theta
            .or_else(|| {
                config
                    .rope_parameters
                    .as_ref()
                    .map(|params| params.rope_theta)
            })
            .ok_or_else(|| candle::Error::msg("GTE-RoPE requires rope_theta"))?;
        let inv_freqs = build_inv_freqs(
            attention_head_size,
            rope_theta,
            vb.device(),
            config.rope_scaling.as_ref(),
        )?;
        let (cos_cache, sin_cache) = cos_sin(
            config.max_position_embeddings,
            &inv_freqs,
            vb.dtype(),
            false,
        )?;
        Ok(Self {
            embeddings,
            layers,
            cos_cache,
            sin_cache,
            config: config.clone(),
        })
    }

    fn inner_load(
        vb: VarBuilder,
        config: &Config,
    ) -> Result<(GteRopeEmbeddings, Vec<GteRopeLayer>)> {
        let embeddings = GteRopeEmbeddings::load(vb.pp("embeddings"), config)?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| GteRopeLayer::load(vb.pp(format!("encoder.layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        Ok((embeddings, layers))
    }

    pub fn kernel_backend(&self) -> (&'static str, &'static str, bool, bool, bool) {
        let linear_backend = self
            .layers
            .first()
            .map(|layer| layer.mlp.up_gate.backend())
            .unwrap_or("candle_matmul");
        let layernorm_backend = self.embeddings.layer_norm.backend();
        let ffn_activation_fused = self
            .layers
            .first()
            .map(|layer| layer.mlp.gated_activation_fused())
            .unwrap_or(false);
        (
            linear_backend,
            layernorm_backend,
            true,
            cfg!(feature = "cuda"),
            ffn_activation_fused,
        )
    }

    pub fn ignores_token_type_ids(&self) -> bool {
        self.config.type_vocab_size < 2
    }

    pub fn forward_packed(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward_packed(input_ids, token_type_ids)?;
        let cos = index_select(&self.cos_cache, position_ids, 0)?;
        let sin = index_select(&self.sin_cache, position_ids, 0)?;
        for layer in self.layers.iter() {
            hidden_states =
                layer.forward_packed(&hidden_states, seqlens, &cos, &sin, max_seqlen)?;
        }
        Ok(hidden_states)
    }

    pub fn forward_packed_profiled(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        sync_timings: bool,
    ) -> Result<(Tensor, GteRopeForwardProfile)> {
        synchronize_if(input_ids.device(), sync_timings)?;
        let embedding_start = Instant::now();
        let mut hidden_states = self.embeddings.forward_packed(input_ids, token_type_ids)?;
        synchronize_if(hidden_states.device(), sync_timings)?;

        let rope_select_start = Instant::now();
        let cos = index_select(&self.cos_cache, position_ids, 0)?;
        let sin = index_select(&self.sin_cache, position_ids, 0)?;
        synchronize_if(cos.device(), sync_timings)?;

        let total_tokens = input_ids.dim(0)?;
        let mut profile = GteRopeForwardProfile {
            total_tokens,
            max_seqlen,
            hidden_size: self.config.hidden_size,
            intermediate_size: self.config.intermediate_size,
            attention_heads: self.config.num_attention_heads,
            attention_head_size: self.config.hidden_size / self.config.num_attention_heads,
            linear_gflops: linear_gflops(
                total_tokens,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_hidden_layers,
            ),
            embedding_ms: elapsed_ms(embedding_start),
            rope_select_ms: elapsed_ms(rope_select_start),
            ..Default::default()
        };
        for layer in self.layers.iter() {
            hidden_states = layer.forward_packed_profiled(
                &hidden_states,
                seqlens,
                &cos,
                &sin,
                max_seqlen,
                &mut profile,
                sync_timings,
            )?;
        }
        Ok((hidden_states, profile))
    }
}

#[cfg(feature = "cuda")]
fn fused_gated_gelu_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        let disabled = env_bool_any(&["SIE_CANDLE_DISABLE_FUSED_GATED_GELU"], false);
        let enabled = env_bool_any(&["SIE_CANDLE_ENABLE_FUSED_GATED_GELU"], true);
        enabled && !disabled
    })
}

#[cfg(not(feature = "cuda"))]
fn fused_gated_gelu_enabled() -> bool {
    false
}

#[cfg(feature = "cuda")]
fn env_bool_any(names: &[&str], default: bool) -> bool {
    names
        .iter()
        .find_map(|name| std::env::var(name).ok())
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

#[cfg(feature = "cuda")]
fn flash_attn_varlen(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    seqlens: &Tensor,
    max_seqlen: usize,
    head_size: usize,
) -> Result<Tensor> {
    let softmax_scale = 1f32 / f32::sqrt(head_size as f32);
    candle_flash_attn::flash_attn_varlen(
        query,
        key,
        value,
        seqlens,
        seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale,
        false,
    )
}

#[cfg(not(feature = "cuda"))]
fn flash_attn_varlen(
    _query: &Tensor,
    _key: &Tensor,
    _value: &Tensor,
    _seqlens: &Tensor,
    _max_seqlen: usize,
    _head_size: usize,
) -> Result<Tensor> {
    candle::bail!("packed GTE-RoPE FlashAttention requires the cuda feature")
}

fn synchronize_if(device: &Device, enabled: bool) -> Result<()> {
    if enabled {
        device.synchronize()?;
    }
    Ok(())
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn linear_gflops(
    tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
    layers: usize,
) -> f64 {
    let qkv = hidden_size * 3 * hidden_size;
    let attention_output = hidden_size * hidden_size;
    let ffn_up_gate = hidden_size * intermediate_size * 2;
    let ffn_down = intermediate_size * hidden_size;
    2.0 * tokens as f64 * layers as f64 * (qkv + attention_output + ffn_up_gate + ffn_down) as f64
        / 1_000_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_new_model_rope_ntk_config() {
        let config: Config = serde_json::from_str(
            r#"{
                "vocab_size": 250048,
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "hidden_act": "gelu",
                "max_position_embeddings": 8192,
                "type_vocab_size": 1,
                "layer_norm_eps": 1e-12,
                "layer_norm_type": "layer_norm",
                "model_type": "new",
                "pack_qkv": true,
                "position_embedding_type": "rope",
                "rope_theta": 20000,
                "rope_scaling": {"factor": 8.0, "type": "ntk"}
            }"#,
        )
        .unwrap();

        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.position_embedding_type.as_deref(), Some("rope"));
        assert_eq!(config.rope_theta, Some(20000.0));
        match config.rope_scaling {
            Some(RopeScaling::Ntk { factor, .. }) => assert_eq!(factor, 8.0),
            other => panic!("expected NTK RoPE scaling, got {other:?}"),
        }
    }
}
