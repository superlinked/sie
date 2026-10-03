//! Local ModernBERT backbone wrapper for Candle embedding models.
//!
//! This mirrors `candle_transformers::models::modernbert::ModernBert`
//! closely, but keeps the attention mask dtype aligned with the attention
//! logits before adding it. The upstream implementation builds the mask in
//! F32, which fails on BF16 ModernBERT checkpoints on CUDA.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::candle_layers::{index_select, FastLayerNorm, FastLinear};
use crate::candle_rope::apply_rotary_packed_inplace;
use candle::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, ops::softmax, Embedding, Module, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub norm_bias: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: String,
    pub pad_token_id: u32,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    #[serde(default)]
    #[serde(flatten)]
    pub classifier_config: Option<ClassifierConfig>,
}

fn default_norm_eps() -> f64 {
    1e-5
}

fn default_hidden_activation() -> String {
    "gelu".to_string()
}

#[derive(Debug, Clone, Deserialize, PartialEq, Copy, Default)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierPooling {
    #[default]
    Cls,
    Mean,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ClassifierConfig {
    pub id2label: HashMap<String, String>,
    pub label2id: HashMap<String, String>,
    pub classifier_pooling: ClassifierPooling,
}

impl Config {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            candle::bail!("ModernBERT vocab_size must be greater than zero");
        }
        if self.pad_token_id as usize >= self.vocab_size {
            candle::bail!(
                "ModernBERT pad_token_id {} must be below vocab_size {}",
                self.pad_token_id,
                self.vocab_size
            );
        }
        if self.hidden_size == 0 {
            candle::bail!("ModernBERT hidden_size must be greater than zero");
        }
        if self.num_hidden_layers == 0 {
            candle::bail!("ModernBERT num_hidden_layers must be greater than zero");
        }
        if self.num_attention_heads == 0 {
            candle::bail!("ModernBERT num_attention_heads must be greater than zero");
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            candle::bail!(
                "ModernBERT hidden_size {} must be divisible by num_attention_heads {}",
                self.hidden_size,
                self.num_attention_heads
            );
        }
        if self.hidden_size.checked_mul(3).is_none() {
            candle::bail!("ModernBERT hidden_size is too large");
        }
        let head_size = self.hidden_size / self.num_attention_heads;
        if !head_size.is_multiple_of(2) {
            candle::bail!("ModernBERT attention head size {head_size} must be even for RoPE");
        }
        if self.global_attn_every_n_layers == 0 {
            candle::bail!("ModernBERT global_attn_every_n_layers must be greater than zero");
        }
        if self.intermediate_size == 0 {
            candle::bail!("ModernBERT intermediate_size must be greater than zero");
        }
        if self.intermediate_size > i32::MAX as usize
            || self.intermediate_size.checked_mul(2).is_none()
        {
            candle::bail!("ModernBERT intermediate_size must fit the CUDA gated-activation ABI");
        }
        if self.max_position_embeddings == 0 || self.max_position_embeddings > i32::MAX as usize {
            candle::bail!("ModernBERT max_position_embeddings must be in 1..=i32::MAX");
        }
        if !self.norm_eps.is_finite() || self.norm_eps <= 0.0 {
            candle::bail!("ModernBERT norm_eps must be finite and greater than zero");
        }
        let uses_local_attention =
            (0..self.num_hidden_layers).any(|layer| layer % self.global_attn_every_n_layers != 0);
        if uses_local_attention
            && (self.local_attention == 0 || self.local_attention / 2 > i32::MAX as usize)
        {
            candle::bail!(
                "ModernBERT local_attention must be positive and its half-window must fit i32 when local-attention layers are present"
            );
        }
        if self.norm_bias || self.attention_bias || self.mlp_bias {
            candle::bail!(
                "ModernBERT Candle supports only norm_bias=false, attention_bias=false, and mlp_bias=false"
            );
        }
        if self.hidden_activation != "gelu" {
            candle::bail!(
                "ModernBERT Candle supports only hidden_activation=gelu, got {}",
                self.hidden_activation
            );
        }
        for (name, theta) in [
            ("global_rope_theta", self.global_rope_theta),
            ("local_rope_theta", self.local_rope_theta),
        ] {
            if !theta.is_finite() || theta <= 0.0 {
                candle::bail!("ModernBERT {name} must be finite and greater than zero");
            }
        }
        self.validate_rope_configuration()?;
        Ok(())
    }

    fn validate_rope_configuration(&self) -> Result<()> {
        if let Some(rope_scaling) = self.rope_scaling.as_ref() {
            let params = rope_scaling.as_object().ok_or_else(|| {
                candle::Error::msg("ModernBERT rope_scaling must be an object when present")
            })?;
            let rope_type = validated_rope_type(params, "rope_scaling")?;
            if rope_type != "default"
                || params.keys().any(|key| key != "rope_type" && key != "type")
            {
                candle::bail!(
                    "ModernBERT Candle supports only default, unscaled rope_scaling metadata"
                );
            }
        }

        let Some(rope_parameters) = self.rope_parameters.as_ref() else {
            return Ok(());
        };
        let attention_parameters = rope_parameters.as_object().ok_or_else(|| {
            candle::Error::msg("ModernBERT rope_parameters must be an object when present")
        })?;
        if attention_parameters.len() != 2 {
            candle::bail!(
                "ModernBERT rope_parameters must contain exactly full_attention and sliding_attention"
            );
        }
        for (attention_kind, expected_theta) in [
            ("full_attention", self.global_rope_theta),
            ("sliding_attention", self.local_rope_theta),
        ] {
            let params = attention_parameters
                .get(attention_kind)
                .and_then(serde_json::Value::as_object)
                .ok_or_else(|| {
                    candle::Error::msg(format!(
                        "ModernBERT rope_parameters is missing object {attention_kind}"
                    ))
                })?;
            let rope_type =
                validated_rope_type(params, &format!("rope_parameters.{attention_kind}"))?;
            let theta = params
                .get("rope_theta")
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| {
                    candle::Error::msg(format!(
                        "ModernBERT rope_parameters.{attention_kind}.rope_theta must be numeric"
                    ))
                })?;
            if rope_type != "default"
                || params
                    .keys()
                    .any(|key| key != "rope_theta" && key != "rope_type" && key != "type")
                || theta != expected_theta
            {
                candle::bail!(
                    "ModernBERT Candle supports only matching default {attention_kind} RoPE parameters"
                );
            }
        }
        Ok(())
    }
}

fn validated_rope_type<'a>(
    params: &'a serde_json::Map<String, serde_json::Value>,
    context: &str,
) -> Result<&'a str> {
    let rope_type = params
        .get("rope_type")
        .map(|value| {
            value.as_str().ok_or_else(|| {
                candle::Error::msg(format!("ModernBERT {context}.rope_type must be a string"))
            })
        })
        .transpose()?;
    let legacy_type = params
        .get("type")
        .map(|value| {
            value.as_str().ok_or_else(|| {
                candle::Error::msg(format!("ModernBERT {context}.type must be a string"))
            })
        })
        .transpose()?;
    if let (Some(rope_type), Some(legacy_type)) = (rope_type, legacy_type) {
        if rope_type != legacy_type {
            candle::bail!(
                "ModernBERT {context}.rope_type={rope_type} conflicts with type={legacy_type}"
            );
        }
    }
    Ok(rope_type.or(legacy_type).unwrap_or("default"))
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ModernBertForwardProfile {
    pub total_tokens: usize,
    pub max_seqlen: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub attention_heads: usize,
    pub attention_head_size: usize,
    pub linear_gflops: f64,
    pub embedding_ms: f64,
    pub embedding_norm_ms: f64,
    pub rope_select_ms: f64,
    pub attention_ms: f64,
    pub attention_norm_ms: f64,
    pub attention_qkv_ms: f64,
    pub attention_rotary_ms: f64,
    pub attention_flash_ms: f64,
    pub attention_output_dense_ms: f64,
    pub mlp_ms: f64,
    pub mlp_norm_ms: f64,
    pub mlp_wi_ms: f64,
    pub mlp_activation_ms: f64,
    pub mlp_wo_ms: f64,
    pub final_norm_ms: f64,
    pub layers: usize,
}

#[derive(Clone, Copy)]
struct ModernBertPackedForward<'a> {
    seqlens: &'a Tensor,
    cos: &'a Tensor,
    sin: &'a Tensor,
    max_seqlen: usize,
    window_size: Option<usize>,
}

struct PackedRotaryEmbeddings {
    global: (Tensor, Tensor),
    local: Option<(Tensor, Tensor)>,
}

impl PackedRotaryEmbeddings {
    fn new(
        global: &RotaryEmbedding,
        local: &RotaryEmbedding,
        position_ids: &Tensor,
        has_local_layers: bool,
    ) -> Result<Self> {
        Ok(Self {
            global: global.packed_cos_sin(position_ids)?,
            local: has_local_layers
                .then(|| local.packed_cos_sin(position_ids))
                .transpose()?,
        })
    }

    fn for_layer(&self, uses_local_attention: bool) -> (&Tensor, &Tensor) {
        if uses_local_attention {
            let (cos, sin) = self
                .local
                .as_ref()
                .expect("local RoPE is precomputed when a local-attention layer exists");
            (cos, sin)
        } else {
            (&self.global.0, &self.global.1)
        }
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, config: &Config, rope_theta: f64, dev: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let max_seq_len = config.max_position_embeddings;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?;
        Ok((q_embed, k_embed))
    }

    fn packed_cos_sin(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            index_select(&self.cos, position_ids, 0)?,
            index_select(&self.sin, position_ids, 0)?,
        ))
    }
}

#[derive(Clone)]
struct ModernBertAttention {
    qkv: FastLinear,
    proj: FastLinear,
    num_attention_heads: usize,
    attention_head_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl ModernBertAttention {
    fn load(vb: VarBuilder, config: &Config, rotary_emb: Arc<RotaryEmbedding>) -> Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let qkv = FastLinear::load_no_bias(
            config.hidden_size,
            config.hidden_size * 3,
            vb.pp("Wqkv"),
            None,
        )?;
        let proj =
            FastLinear::load_no_bias(config.hidden_size, config.hidden_size, vb.pp("Wo"), None)?;

        Ok(Self {
            qkv,
            proj,
            num_attention_heads,
            attention_head_size,
            rotary_emb,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let xs = hidden_states.clone();
        let (b, seq_len, d) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(&xs)?
            .reshape((
                b,
                seq_len,
                3,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.get(0)?;
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k)?;

        let scale = (self.attention_head_size as f64).powf(-0.5);
        let q = (q * scale)?;

        let att = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attention_mask = if attention_mask.dtype() == att.dtype() {
            attention_mask.clone()
        } else {
            attention_mask.to_dtype(att.dtype())?
        };
        let att = att.broadcast_add(&attention_mask)?;
        let att = softmax(&att, D::Minus1)?;

        let xs = att.matmul(&v)?;

        let xs = xs.transpose(1, 2)?.reshape((b, seq_len, d))?;
        let xs = self.proj.forward(&xs)?;
        let xs = xs.reshape((b, seq_len, d))?;

        Ok(xs)
    }

    fn forward_packed(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_seqlen: usize,
        window_size: Option<usize>,
    ) -> Result<Tensor> {
        let total_tokens = hidden_states.dim(0)?;
        let qkv = self.qkv.forward(hidden_states)?.reshape((
            total_tokens,
            self.num_attention_heads * 3,
            self.attention_head_size,
        ))?;

        let query = qkv.narrow(1, 0, self.num_attention_heads)?;
        let key = qkv.narrow(1, self.num_attention_heads, self.num_attention_heads)?;
        let value = qkv.narrow(1, self.num_attention_heads * 2, self.num_attention_heads)?;

        apply_rotary_packed_inplace(&query, &key, cos, sin)?;

        let attention_output = flash_attn_varlen_windowed(
            &query,
            &key,
            &value,
            seqlens,
            max_seqlen,
            self.attention_head_size,
            window_size,
        )?;
        let attention_output = attention_output.reshape((
            total_tokens,
            self.num_attention_heads * self.attention_head_size,
        ))?;
        self.proj.forward(&attention_output)
    }

    fn forward_packed_profiled(
        &self,
        hidden_states: &Tensor,
        ctx: ModernBertPackedForward<'_>,
        profile: &mut ModernBertForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        let attention_start = Instant::now();
        let total_tokens = hidden_states.dim(0)?;
        let qkv_start = Instant::now();
        let qkv = self.qkv.forward(hidden_states)?.reshape((
            total_tokens,
            self.num_attention_heads * 3,
            self.attention_head_size,
        ))?;
        let query = qkv.narrow(1, 0, self.num_attention_heads)?;
        let key = qkv.narrow(1, self.num_attention_heads, self.num_attention_heads)?;
        let value = qkv.narrow(1, self.num_attention_heads * 2, self.num_attention_heads)?;
        synchronize_if(hidden_states.device(), sync_timings)?;
        profile.attention_qkv_ms += elapsed_ms(qkv_start);

        let rotary_start = Instant::now();
        apply_rotary_packed_inplace(&query, &key, ctx.cos, ctx.sin)?;
        synchronize_if(hidden_states.device(), sync_timings)?;
        profile.attention_rotary_ms += elapsed_ms(rotary_start);

        let flash_start = Instant::now();
        let attention_output = flash_attn_varlen_windowed(
            &query,
            &key,
            &value,
            ctx.seqlens,
            ctx.max_seqlen,
            self.attention_head_size,
            ctx.window_size,
        )?;
        synchronize_if(attention_output.device(), sync_timings)?;
        profile.attention_flash_ms += elapsed_ms(flash_start);

        let output_start = Instant::now();
        let attention_output = attention_output.reshape((
            total_tokens,
            self.num_attention_heads * self.attention_head_size,
        ))?;
        let output = self.proj.forward(&attention_output)?;
        synchronize_if(output.device(), sync_timings)?;
        profile.attention_output_dense_ms += elapsed_ms(output_start);
        profile.attention_ms += elapsed_ms(attention_start);
        Ok(output)
    }
}

#[derive(Clone)]
struct ModernBertGatedActivation {
    intermediate_size: usize,
    #[cfg(feature = "cuda")]
    cuda: candle_gated_activation::GeluErfGate,
}

impl ModernBertGatedActivation {
    fn new(intermediate_size: usize) -> Result<Self> {
        if intermediate_size == 0 {
            candle::bail!("ModernBERT intermediate_size must be greater than zero");
        }
        Ok(Self {
            intermediate_size,
            #[cfg(feature = "cuda")]
            cuda: candle_gated_activation::GeluErfGate::new(intermediate_size)?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let width = xs.dim(D::Minus1)?;
        let expected_width = self
            .intermediate_size
            .checked_mul(2)
            .ok_or_else(|| candle::Error::msg("ModernBERT MLP activation width overflow"))?;
        if width != expected_width {
            candle::bail!("ModernBERT MLP activation expected width {expected_width}, got {width}");
        }

        #[cfg(feature = "cuda")]
        if matches!(xs.device(), Device::Cuda(_)) && xs.is_contiguous() {
            return self.cuda.forward(xs);
        }

        let xs = xs.chunk(2, D::Minus1)?;
        &xs[0].gelu_erf()? * &xs[1]
    }
}

#[derive(Clone)]
struct ModernBertMLP {
    wi: FastLinear,
    wo: FastLinear,
    activation: ModernBertGatedActivation,
}

impl ModernBertMLP {
    fn load(
        vb: VarBuilder,
        config: &Config,
        activation: ModernBertGatedActivation,
    ) -> Result<Self> {
        let wi = FastLinear::load_no_bias(
            config.hidden_size,
            config.intermediate_size * 2,
            vb.pp("Wi"),
            None,
        )?;
        let wo = FastLinear::load_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("Wo"),
            None,
        )?;
        Ok(Self { wi, wo, activation })
    }

    fn activate(&self, xs: &Tensor) -> Result<Tensor> {
        self.activation.forward(xs)
    }
}

impl Module for ModernBertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.wi.forward(xs)?;
        self.wo.forward(&self.activate(&xs)?)
    }
}

impl ModernBertMLP {
    fn forward_profiled(
        &self,
        xs: &Tensor,
        profile: &mut ModernBertForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        let mlp_start = Instant::now();
        let wi_start = Instant::now();
        let xs = self.wi.forward(xs)?;
        synchronize_if(xs.device(), sync_timings)?;
        profile.mlp_wi_ms += elapsed_ms(wi_start);

        let activation_start = Instant::now();
        let xs = self.activate(&xs)?;
        synchronize_if(xs.device(), sync_timings)?;
        profile.mlp_activation_ms += elapsed_ms(activation_start);

        let wo_start = Instant::now();
        let xs = self.wo.forward(&xs)?;
        synchronize_if(xs.device(), sync_timings)?;
        profile.mlp_wo_ms += elapsed_ms(wo_start);
        profile.mlp_ms += elapsed_ms(mlp_start);
        Ok(xs)
    }
}

#[derive(Clone)]
struct ModernBertLayer {
    attn: ModernBertAttention,
    mlp: ModernBertMLP,
    attn_norm: Option<FastLayerNorm>,
    mlp_norm: FastLayerNorm,
    uses_local_attention: bool,
}

impl ModernBertLayer {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
        layer_id: usize,
        uses_local_attention: bool,
        activation: ModernBertGatedActivation,
    ) -> Result<Self> {
        let attn = ModernBertAttention::load(vb.pp("attn"), config, rotary_emb)?;
        let mlp = ModernBertMLP::load(vb.pp("mlp"), config, activation)?;
        let attn_norm = if layer_id == 0 {
            None
        } else {
            Some(FastLayerNorm::load_no_bias(
                vb.pp("attn_norm"),
                config.hidden_size,
                config.norm_eps,
            )?)
        };
        let mlp_norm =
            FastLayerNorm::load_no_bias(vb.pp("mlp_norm"), config.hidden_size, config.norm_eps)?;
        Ok(Self {
            attn,
            mlp,
            attn_norm,
            mlp_norm,
            uses_local_attention,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        global_attention_mask: &Tensor,
        local_attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = norm.forward(&xs, None)?;
        }

        let attention_mask = if self.uses_local_attention {
            global_attention_mask.broadcast_add(local_attention_mask)?
        } else {
            global_attention_mask.clone()
        };
        let xs = self.attn.forward(&xs, &attention_mask)?;
        let xs = (xs + residual)?;
        let mlp_out = self.mlp.forward(&self.mlp_norm.forward(&xs, None)?)?;
        let xs = (xs + mlp_out)?;
        Ok(xs)
    }

    fn forward_packed(
        &self,
        xs: &Tensor,
        seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        max_seqlen: usize,
        local_window_size: Option<usize>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = norm.forward(&xs, None)?;
        }

        let window_size = if self.uses_local_attention {
            local_window_size
        } else {
            None
        };
        let xs = self
            .attn
            .forward_packed(&xs, seqlens, cos, sin, max_seqlen, window_size)?;
        let xs = (xs + residual)?;
        let mlp_out = self.mlp.forward(&self.mlp_norm.forward(&xs, None)?)?;
        let xs = (xs + mlp_out)?;
        Ok(xs)
    }

    fn forward_packed_profiled(
        &self,
        xs: &Tensor,
        ctx: ModernBertPackedForward<'_>,
        profile: &mut ModernBertForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            let norm_start = Instant::now();
            xs = norm.forward(&xs, None)?;
            synchronize_if(xs.device(), sync_timings)?;
            profile.attention_norm_ms += elapsed_ms(norm_start);
        }

        let window_size = if self.uses_local_attention {
            ctx.window_size
        } else {
            None
        };
        let attention_ctx = ModernBertPackedForward { window_size, ..ctx };
        let xs = self
            .attn
            .forward_packed_profiled(&xs, attention_ctx, profile, sync_timings)?;
        let xs = (xs + residual)?;

        let mlp_norm_start = Instant::now();
        let mlp_input = self.mlp_norm.forward(&xs, None)?;
        synchronize_if(mlp_input.device(), sync_timings)?;
        profile.mlp_norm_ms += elapsed_ms(mlp_norm_start);

        let mlp_out = self
            .mlp
            .forward_profiled(&mlp_input, profile, sync_timings)?;
        let xs = (xs + mlp_out)?;
        profile.layers += 1;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBert {
    word_embeddings: Embedding,
    norm: FastLayerNorm,
    global_rotary_emb: Arc<RotaryEmbedding>,
    local_rotary_emb: Arc<RotaryEmbedding>,
    layers: Vec<ModernBertLayer>,
    final_norm: FastLayerNorm,
    local_attention_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
}

impl ModernBert {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        config.validate()?;
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embeddings.tok_embeddings"),
        )?;
        let norm = FastLayerNorm::load_no_bias(
            vb.pp("model.embeddings.norm"),
            config.hidden_size,
            config.norm_eps,
        )?;
        let global_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.global_rope_theta,
            vb.device(),
        )?);
        let local_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.local_rope_theta,
            vb.device(),
        )?);

        let activation = ModernBertGatedActivation::new(config.intermediate_size)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            let layer_uses_local_attention = layer_id % config.global_attn_every_n_layers != 0;
            layers.push(ModernBertLayer::load(
                vb.pp(format!("model.layers.{layer_id}")),
                config,
                if layer_uses_local_attention {
                    local_rotary_emb.clone()
                } else {
                    global_rotary_emb.clone()
                },
                layer_id,
                layer_uses_local_attention,
                activation.clone(),
            )?);
        }

        let final_norm = FastLayerNorm::load_no_bias(
            vb.pp("model.final_norm"),
            config.hidden_size,
            config.norm_eps,
        )?;

        Ok(Self {
            word_embeddings,
            norm,
            global_rotary_emb,
            local_rotary_emb,
            layers,
            final_norm,
            local_attention_size: config.local_attention,
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            num_hidden_layers: config.num_hidden_layers,
            num_attention_heads: config.num_attention_heads,
        })
    }

    pub fn kernel_backend(&self) -> (&'static str, &'static str) {
        let linear_backend = self
            .layers
            .first()
            .map(|layer| layer.attn.qkv.backend())
            .unwrap_or("candle_matmul");
        (linear_backend, self.norm.backend())
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let seq_len = xs.shape().dims()[1];
        let global_attention_mask =
            prepare_4d_attention_mask(mask, DType::F32, None)?.to_device(xs.device())?;
        let local_attention_mask =
            get_local_attention_mask(seq_len, self.local_attention_size / 2, xs.device())?;
        let embeddings = xs.apply(&self.word_embeddings)?;
        let mut xs = self.norm.forward(&embeddings, None)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, &global_attention_mask, &local_attention_mask)?;
        }
        let xs = self.final_norm.forward(&xs, None)?;
        Ok(xs)
    }

    pub fn forward_packed(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let embeddings = input_ids.apply(&self.word_embeddings)?;
        let mut xs = self.norm.forward(&embeddings, None)?;
        let rope = PackedRotaryEmbeddings::new(
            &self.global_rotary_emb,
            &self.local_rotary_emb,
            position_ids,
            self.layers.iter().any(|layer| layer.uses_local_attention),
        )?;
        let local_window_size =
            (self.local_attention_size > 0).then_some(self.local_attention_size / 2);
        for layer in self.layers.iter() {
            let (cos, sin) = rope.for_layer(layer.uses_local_attention);
            xs = layer.forward_packed(&xs, seqlens, cos, sin, max_seqlen, local_window_size)?;
        }
        self.final_norm.forward(&xs, None)
    }

    pub fn forward_packed_profiled(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        sync_timings: bool,
    ) -> Result<(Tensor, ModernBertForwardProfile)> {
        synchronize_if(input_ids.device(), sync_timings)?;
        let embedding_start = Instant::now();
        let mut xs = input_ids.apply(&self.word_embeddings)?;
        synchronize_if(xs.device(), sync_timings)?;
        let embedding_ms = elapsed_ms(embedding_start);

        let norm_start = Instant::now();
        xs = self.norm.forward(&xs, None)?;
        synchronize_if(xs.device(), sync_timings)?;
        let embedding_norm_ms = elapsed_ms(norm_start);

        let rope_select_start = Instant::now();
        let rope = PackedRotaryEmbeddings::new(
            &self.global_rotary_emb,
            &self.local_rotary_emb,
            position_ids,
            self.layers.iter().any(|layer| layer.uses_local_attention),
        )?;
        synchronize_if(input_ids.device(), sync_timings)?;
        let rope_select_ms = elapsed_ms(rope_select_start);

        let total_tokens = input_ids.dim(0)?;
        let mut profile = ModernBertForwardProfile {
            total_tokens,
            max_seqlen,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            attention_heads: self.num_attention_heads,
            attention_head_size: self.hidden_size / self.num_attention_heads,
            linear_gflops: linear_gflops(
                total_tokens,
                self.hidden_size,
                self.intermediate_size,
                self.num_hidden_layers,
            ),
            embedding_ms,
            embedding_norm_ms,
            rope_select_ms,
            ..Default::default()
        };

        let local_window_size =
            (self.local_attention_size > 0).then_some(self.local_attention_size / 2);
        for layer in self.layers.iter() {
            let (cos, sin) = rope.for_layer(layer.uses_local_attention);
            let ctx = ModernBertPackedForward {
                seqlens,
                cos,
                sin,
                max_seqlen,
                window_size: local_window_size,
            };
            xs = layer.forward_packed_profiled(&xs, ctx, &mut profile, sync_timings)?;
        }

        let final_norm_start = Instant::now();
        let xs = self.final_norm.forward(&xs, None)?;
        synchronize_if(xs.device(), sync_timings)?;
        profile.final_norm_ms = elapsed_ms(final_norm_start);
        Ok((xs, profile))
    }
}

fn prepare_4d_attention_mask(
    mask: &Tensor,
    dtype: DType,
    tgt_len: Option<usize>,
) -> Result<Tensor> {
    let bsz = mask.dim(0)?;
    let src_len = mask.dim(1)?;
    let tgt_len = tgt_len.unwrap_or(src_len);

    let expanded_mask = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((bsz, 1, tgt_len, src_len))?
        .to_dtype(dtype)?;

    let inverted_mask = (1.0 - expanded_mask)?;

    (inverted_mask * f32::MIN as f64)?.to_dtype(dtype)
}

fn get_local_attention_mask(
    seq_len: usize,
    max_distance: usize,
    device: &Device,
) -> Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if (j as i32 - i as i32).abs() > max_distance as i32 {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
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
    total_tokens: usize,
    hidden_size: usize,
    intermediate_size: usize,
    layers: usize,
) -> f64 {
    let tokens = total_tokens as f64;
    let hidden = hidden_size as f64;
    let intermediate = intermediate_size as f64;
    let layers = layers as f64;
    let attention_qkv = 2.0 * tokens * hidden * (3.0 * hidden);
    let attention_output = 2.0 * tokens * hidden * hidden;
    let mlp_wi = 2.0 * tokens * hidden * (2.0 * intermediate);
    let mlp_wo = 2.0 * tokens * intermediate * hidden;
    (attention_qkv + attention_output + mlp_wi + mlp_wo) * layers / 1e9
}

#[cfg(feature = "cuda")]
fn flash_attn_varlen_windowed(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    seqlens: &Tensor,
    max_seqlen: usize,
    head_size: usize,
    window_size: Option<usize>,
) -> Result<Tensor> {
    let softmax_scale = 1f32 / f32::sqrt(head_size as f32);
    candle_flash_attn::flash_attn_varlen_windowed(
        query,
        key,
        value,
        seqlens,
        seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale,
        window_size,
        window_size,
    )
}

#[cfg(not(feature = "cuda"))]
fn flash_attn_varlen_windowed(
    _query: &Tensor,
    _key: &Tensor,
    _value: &Tensor,
    _seqlens: &Tensor,
    _max_seqlen: usize,
    _head_size: usize,
    _window_size: Option<usize>,
) -> Result<Tensor> {
    candle::bail!("packed ModernBERT FlashAttention requires the cuda feature")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepares_attention_mask_with_requested_dtype() {
        let mask = Tensor::from_vec(vec![1u32, 0], (1, 2), &Device::Cpu).unwrap();
        let prepared = prepare_4d_attention_mask(&mask, DType::F32, None).unwrap();
        assert_eq!(prepared.dtype(), DType::F32);
        assert_eq!(prepared.dims(), &[1, 1, 2, 2]);
    }

    #[test]
    fn local_attention_window_matches_pytorch_adapter() {
        let local_attention_size = 128usize;
        assert_eq!(local_attention_size / 2, 64);
    }

    #[test]
    fn modernbert_mlp_activation_uses_exact_gelu_on_first_half() -> Result<()> {
        let device = Device::Cpu;
        let unused_weight = Tensor::zeros((1, 1), DType::F32, &device)?;
        let mlp = ModernBertMLP {
            wi: FastLinear::new(unused_weight.clone(), None, None),
            wo: FastLinear::new(unused_weight, None, None),
            activation: ModernBertGatedActivation::new(2)?,
        };
        let xs = Tensor::new(&[[-1f32, 2., 3., 4.]], &device)?;

        let actual = mlp.activate(&xs)?;
        let halves = xs.chunk(2, D::Minus1)?;
        let expected = (&halves[0].gelu_erf()? * &halves[1])?;

        assert_eq!(actual.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn modernbert_config_rejects_invalid_attention_geometry() {
        let mut config = test_config();
        config.num_attention_heads = 0;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("num_attention_heads"));

        let mut config = test_config();
        config.hidden_size = 10;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must be divisible"));

        let mut config = test_config();
        config.global_attn_every_n_layers = 0;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("global_attn_every_n_layers"));

        let mut config = test_config();
        config.local_attention = 0;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("half-window"));

        let mut config = test_config();
        config.norm_eps = f64::NAN;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("norm_eps"));

        let mut config = test_config();
        config.attention_bias = true;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("attention_bias=false"));

        let mut config = test_config();
        config.hidden_activation = "silu".to_string();
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("hidden_activation=gelu"));

        let mut config = test_config();
        config.rope_scaling = Some(serde_json::json!({
            "rope_type": "linear",
            "factor": 2.0
        }));
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("unscaled rope_scaling"));

        let mut config = test_config();
        config.rope_scaling = Some(serde_json::json!({"rope_type": 7}));
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must be a string"));

        let mut config = test_config();
        config.rope_scaling = Some(serde_json::json!({
            "rope_type": "default",
            "type": "linear"
        }));
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("conflicts"));

        let mut config = test_config();
        config.rope_parameters = Some(serde_json::json!({
            "full_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 20_000.0, "rope_type": "default"}
        }));
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("sliding_attention"));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fused_modernbert_mlp_activation_matches_unfused_bf16_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let xs = Tensor::new(
            &[
                [-3f32, -1., 0.5, 2., 7., -4., 3., 0.25],
                [1.5, -0.25, 4., -2., -3., 6., 0.5, 8.],
            ],
            &device,
        )?
        .to_dtype(DType::BF16)?;

        let actual = candle_gated_activation::GeluErfGate::new(4)?.forward(&xs)?;
        let halves = xs.chunk(2, D::Minus1)?;
        let expected = (&halves[0].gelu_erf()? * &halves[1])?;

        assert_eq!(
            actual.to_vec2::<half::bf16>()?,
            expected.to_vec2::<half::bf16>()?
        );
        Ok(())
    }

    #[test]
    fn packed_rotary_embeddings_gather_global_and_local_positions() -> Result<()> {
        let device = Device::Cpu;
        let global = RotaryEmbedding {
            cos: Tensor::new(&[[1f32, 2.], [3., 4.], [5., 6.]], &device)?,
            sin: Tensor::new(&[[11f32, 12.], [13., 14.], [15., 16.]], &device)?,
        };
        let local = RotaryEmbedding {
            cos: Tensor::new(&[[21f32, 22.], [23., 24.], [25., 26.]], &device)?,
            sin: Tensor::new(&[[31f32, 32.], [33., 34.], [35., 36.]], &device)?,
        };
        let position_ids = Tensor::new(&[2u32, 0], &device)?;

        let packed = PackedRotaryEmbeddings::new(&global, &local, &position_ids, true)?;
        let (global_cos, global_sin) = packed.for_layer(false);
        let (local_cos, local_sin) = packed.for_layer(true);

        assert_eq!(
            global_cos.to_vec2::<f32>()?,
            vec![vec![5., 6.], vec![1., 2.]]
        );
        assert_eq!(
            global_sin.to_vec2::<f32>()?,
            vec![vec![15., 16.], vec![11., 12.]]
        );
        assert_eq!(
            local_cos.to_vec2::<f32>()?,
            vec![vec![25., 26.], vec![21., 22.]]
        );
        assert_eq!(
            local_sin.to_vec2::<f32>()?,
            vec![vec![35., 36.], vec![31., 32.]]
        );
        Ok(())
    }

    #[test]
    fn packed_rotary_embeddings_skip_unused_local_cache() -> Result<()> {
        let device = Device::Cpu;
        let global = RotaryEmbedding {
            cos: Tensor::new(&[[1f32], [2.]], &device)?,
            sin: Tensor::new(&[[3f32], [4.]], &device)?,
        };
        // Position 1 is deliberately out of range for the local cache. This
        // succeeds only if a global-only model does not gather local RoPE.
        let local = RotaryEmbedding {
            cos: Tensor::new(&[[5f32]], &device)?,
            sin: Tensor::new(&[[6f32]], &device)?,
        };
        let position_ids = Tensor::new(&[1u32], &device)?;

        let packed = PackedRotaryEmbeddings::new(&global, &local, &position_ids, false)?;

        assert!(packed.local.is_none());
        assert_eq!(packed.for_layer(false).0.to_vec2::<f32>()?, vec![vec![2.]]);
        Ok(())
    }

    fn test_config() -> Config {
        Config {
            vocab_size: 128,
            hidden_size: 12,
            num_hidden_layers: 2,
            num_attention_heads: 3,
            intermediate_size: 16,
            max_position_embeddings: 64,
            norm_eps: 1e-5,
            norm_bias: false,
            attention_bias: false,
            mlp_bias: false,
            hidden_activation: "gelu".to_string(),
            pad_token_id: 0,
            global_attn_every_n_layers: 2,
            global_rope_theta: 10_000.0,
            local_attention: 8,
            local_rope_theta: 10_000.0,
            rope_scaling: None,
            rope_parameters: None,
            classifier_config: None,
        }
    }
}
