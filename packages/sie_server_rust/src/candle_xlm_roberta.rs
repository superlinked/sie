//! XLM-RoBERTa encoder with a packed FlashAttention path for Candle.

use std::time::Instant;

use crate::candle_layers::{index_select, FastLayerNorm, FastLinear, HiddenAct};
use candle::{DType, Device, Result, Tensor};
use candle_nn::{embedding, ops::softmax_last_dim, Embedding, Module, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub layer_norm_eps: f64,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub pad_token_id: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct XlmRobertaForwardProfile {
    pub total_tokens: usize,
    pub max_seqlen: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub attention_heads: usize,
    pub attention_head_size: usize,
    pub linear_gflops: f64,
    pub embedding_ms: f64,
    pub attention_ms: f64,
    pub attention_qkv_ms: f64,
    pub attention_flash_ms: f64,
    pub attention_output_dense_ms: f64,
    pub attention_output_layernorm_ms: f64,
    pub ffn_ms: f64,
    pub ffn_intermediate_dense_ms: f64,
    pub ffn_activation_ms: f64,
    pub ffn_output_dense_ms: f64,
    pub ffn_output_layernorm_ms: f64,
    pub layers: usize,
}

struct XlmRobertaEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: FastLayerNorm,
    padding_idx: u32,
}

impl XlmRobertaEmbeddings {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.type_vocab_size,
            config.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = FastLayerNorm::load(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps,
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            padding_idx: config.pad_token_id,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let embeddings = (&input_embeddings + token_type_embeddings)?;
        let mask = input_ids
            .ne(self.padding_idx)?
            .to_dtype(input_embeddings.dtype())?;
        let cumsum = mask.cumsum(1)?;
        let position_ids = (cumsum * mask)?
            .broadcast_add(
                &Tensor::try_from(self.padding_idx)?
                    .to_dtype(input_embeddings.dtype())?
                    .to_device(input_embeddings.device())?,
            )?
            .to_dtype(DType::U32)?;
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        self.layer_norm
            .forward(&embeddings, Some(&position_embeddings))
    }

    fn forward_packed(&self, input_ids: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let input_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(position_ids)?;
        // Packed XLM-R always uses token type zero. Broadcast that single row instead of
        // allocating zero ids and gathering the same embedding for every token.
        let token_type_embedding = packed_token_type_embedding(&self.token_type_embeddings)?;
        let embeddings = input_embeddings.broadcast_add(&token_type_embedding)?;
        self.layer_norm
            .forward(&embeddings, Some(&position_embeddings))
    }
}

fn packed_token_type_embedding(token_type_embeddings: &Embedding) -> Result<Tensor> {
    token_type_embeddings.embeddings().narrow(0, 0, 1)
}

fn packed_cls_indices(seqlens: &Tensor) -> Result<Tensor> {
    let offset_count = seqlens.dim(0)?;
    if offset_count == 0 {
        candle::bail!("packed XLM-R sequence offsets are empty")
    }
    seqlens.narrow(0, 0, offset_count - 1)
}

fn select_packed_cls_rows(
    self_outputs: &Tensor,
    hidden_states: &Tensor,
    seqlens: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let indices = packed_cls_indices(seqlens)?;
    Ok((
        index_select(self_outputs, &indices, 0)?,
        index_select(hidden_states, &indices, 0)?,
    ))
}

struct XlmRobertaSelfAttention {
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
    qkv: FastLinear,
}

impl XlmRobertaSelfAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention_head_size = cfg.hidden_size / cfg.num_attention_heads;
        let all_head_size = cfg.num_attention_heads * attention_head_size;
        Ok(Self {
            num_attention_heads: cfg.num_attention_heads,
            attention_head_size,
            all_head_size,
            qkv: FastLinear::load_qkv(cfg.hidden_size, all_head_size, vb)?,
        })
    }

    fn qkv(&self, hidden_states: &Tensor) -> Result<Vec<Tensor>> {
        let qkv = self.qkv.forward(hidden_states)?;
        let mut qkv_shape = qkv.dims().to_vec();
        qkv_shape.pop();
        qkv_shape.push(self.num_attention_heads * 3);
        qkv_shape.push(self.attention_head_size);
        let qkv = qkv.reshape(qkv_shape)?;
        qkv.chunk(3, qkv.rank() - 2)
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let qkv = self.qkv(hidden_states)?;
        let query_layer = qkv[0].permute((0, 2, 1, 3))?.contiguous()?;
        let key_layer = qkv[1].permute((0, 2, 1, 3))?.contiguous()?;
        let value_layer = qkv[2].permute((0, 2, 1, 3))?.contiguous()?;
        let mut attention_scores = query_layer.matmul(&key_layer.transpose(2, 3)?)?;
        let scale = 1f64 / f64::sqrt(self.attention_head_size as f64);

        attention_scores = (attention_scores * scale)?;
        attention_scores =
            attention_scores.broadcast_add(&attention_mask.to_dtype(attention_scores.dtype())?)?;
        let attention_probs = softmax_last_dim(&attention_scores)?;

        let context_layer = attention_probs
            .matmul(&value_layer)?
            .permute((0, 2, 1, 3))?
            .contiguous()?;
        let mut new_context_layer_shape =
            context_layer.dims()[..context_layer.dims().len() - 2].to_vec();
        new_context_layer_shape.push(self.all_head_size);
        context_layer.reshape(new_context_layer_shape)
    }

    fn forward_packed(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let total_tokens = hidden_states.dim(0)?;
        let qkv = self.qkv(hidden_states)?;
        let query = qkv[0].reshape((
            total_tokens,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let key = qkv[1].reshape((
            total_tokens,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let value = qkv[2].reshape((
            total_tokens,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let attention_output = flash_attn_varlen(
            &query,
            &key,
            &value,
            seqlens,
            max_seqlen,
            self.attention_head_size,
        )?;
        attention_output.reshape((total_tokens, self.all_head_size))
    }

    fn forward_packed_profiled(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        profile: &mut XlmRobertaForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        let total_tokens = hidden_states.dim(0)?;

        synchronize_if(hidden_states.device(), sync_timings)?;
        let qkv_start = Instant::now();
        let qkv = self.qkv(hidden_states)?;
        let query = qkv[0].reshape((
            total_tokens,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let key = qkv[1].reshape((
            total_tokens,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        let value = qkv[2].reshape((
            total_tokens,
            self.num_attention_heads,
            self.attention_head_size,
        ))?;
        synchronize_if(query.device(), sync_timings)?;
        profile.attention_qkv_ms += elapsed_ms(qkv_start);

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

        attention_output.reshape((total_tokens, self.all_head_size))
    }
}

struct XlmRobertaSelfOutput {
    dense: FastLinear,
    layernorm: FastLayerNorm,
}

impl XlmRobertaSelfOutput {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = FastLinear::load(cfg.hidden_size, cfg.hidden_size, vb.pp("dense"), None)?;
        let layernorm =
            FastLayerNorm::load(vb.pp("LayerNorm"), cfg.hidden_size, cfg.layer_norm_eps)?;
        Ok(Self { dense, layernorm })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layernorm.forward(&hidden_states, Some(input_tensor))
    }

    fn forward_profiled(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
        profile: &mut XlmRobertaForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        synchronize_if(hidden_states.device(), sync_timings)?;
        let dense_start = Instant::now();
        let hidden_states = self.dense.forward(hidden_states)?;
        synchronize_if(hidden_states.device(), sync_timings)?;
        profile.attention_output_dense_ms += elapsed_ms(dense_start);

        let layernorm_start = Instant::now();
        let output = self.layernorm.forward(&hidden_states, Some(input_tensor))?;
        synchronize_if(output.device(), sync_timings)?;
        profile.attention_output_layernorm_ms += elapsed_ms(layernorm_start);
        Ok(output)
    }
}

struct XlmRobertaAttention {
    output: XlmRobertaSelfOutput,
    self_attention: XlmRobertaSelfAttention,
}

impl XlmRobertaAttention {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let output = XlmRobertaSelfOutput::new(cfg, vb.pp("output"))?;
        let self_attention = XlmRobertaSelfAttention::new(cfg, vb.pp("self"))?;
        Ok(Self {
            output,
            self_attention,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask)?;
        self.output.forward(&self_outputs, hidden_states)
    }

    fn forward_packed_cls(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let self_outputs =
            self.self_attention
                .forward_packed(hidden_states, seqlens, max_seqlen)?;
        let (self_outputs, hidden_states) =
            select_packed_cls_rows(&self_outputs, hidden_states, seqlens)?;
        self.output.forward(&self_outputs, &hidden_states)
    }

    fn forward_packed(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let self_outputs =
            self.self_attention
                .forward_packed(hidden_states, seqlens, max_seqlen)?;
        self.output.forward(&self_outputs, hidden_states)
    }

    fn forward_packed_profiled(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        profile: &mut XlmRobertaForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward_packed_profiled(
            hidden_states,
            seqlens,
            max_seqlen,
            profile,
            sync_timings,
        )?;
        self.output
            .forward_profiled(&self_outputs, hidden_states, profile, sync_timings)
    }

    fn forward_packed_cls_profiled(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        profile: &mut XlmRobertaForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        let self_outputs = self.self_attention.forward_packed_profiled(
            hidden_states,
            seqlens,
            max_seqlen,
            profile,
            sync_timings,
        )?;
        let (self_outputs, hidden_states) =
            select_packed_cls_rows(&self_outputs, hidden_states, seqlens)?;
        self.output
            .forward_profiled(&self_outputs, &hidden_states, profile, sync_timings)
    }
}

struct XlmRobertaOutput {
    dense: FastLinear,
    layernorm: FastLayerNorm,
}

impl XlmRobertaOutput {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = FastLinear::load(cfg.intermediate_size, cfg.hidden_size, vb.pp("dense"), None)?;
        let layernorm =
            FastLayerNorm::load(vb.pp("LayerNorm"), cfg.hidden_size, cfg.layer_norm_eps)?;
        Ok(Self { dense, layernorm })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layernorm.forward(&hidden_states, Some(input_tensor))
    }

    fn forward_profiled(
        &self,
        hidden_states: &Tensor,
        input_tensor: &Tensor,
        profile: &mut XlmRobertaForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        synchronize_if(hidden_states.device(), sync_timings)?;
        let dense_start = Instant::now();
        let hidden_states = self.dense.forward(hidden_states)?;
        synchronize_if(hidden_states.device(), sync_timings)?;
        profile.ffn_output_dense_ms += elapsed_ms(dense_start);

        let layernorm_start = Instant::now();
        let output = self.layernorm.forward(&hidden_states, Some(input_tensor))?;
        synchronize_if(output.device(), sync_timings)?;
        profile.ffn_output_layernorm_ms += elapsed_ms(layernorm_start);
        Ok(output)
    }
}

struct XlmRobertaIntermediate {
    dense: FastLinear,
}

impl XlmRobertaIntermediate {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let dense = FastLinear::load(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("dense"),
            Some(cfg.hidden_act),
        )?;
        Ok(Self { dense })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.dense.forward(hidden_states)
    }

    fn forward_profiled(
        &self,
        hidden_states: &Tensor,
        profile: &mut XlmRobertaForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        synchronize_if(hidden_states.device(), sync_timings)?;
        let dense_start = Instant::now();
        let hidden_states = self.dense.forward(hidden_states)?;
        synchronize_if(hidden_states.device(), sync_timings)?;
        profile.ffn_intermediate_dense_ms += elapsed_ms(dense_start);

        Ok(hidden_states)
    }
}

struct XlmRobertaLayer {
    attention: XlmRobertaAttention,
    intermediate: XlmRobertaIntermediate,
    output: XlmRobertaOutput,
}

impl XlmRobertaLayer {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let attention = XlmRobertaAttention::new(cfg, vb.pp("attention"))?;
        let intermediate = XlmRobertaIntermediate::new(cfg, vb.pp("intermediate"))?;
        let output = XlmRobertaOutput::new(cfg, vb.pp("output"))?;
        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }

    fn forward_packed(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let attention_output = self
            .attention
            .forward_packed(hidden_states, seqlens, max_seqlen)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }

    fn forward_packed_cls(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let attention_output =
            self.attention
                .forward_packed_cls(hidden_states, seqlens, max_seqlen)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }

    fn forward_packed_profiled(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        profile: &mut XlmRobertaForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        synchronize_if(hidden_states.device(), sync_timings)?;
        let attention_start = Instant::now();
        let attention_output = self.attention.forward_packed_profiled(
            hidden_states,
            seqlens,
            max_seqlen,
            profile,
            sync_timings,
        )?;
        synchronize_if(attention_output.device(), sync_timings)?;
        profile.attention_ms += elapsed_ms(attention_start);

        let ffn_start = Instant::now();
        let intermediate_output =
            self.intermediate
                .forward_profiled(&attention_output, profile, sync_timings)?;
        let output = self.output.forward_profiled(
            &intermediate_output,
            &attention_output,
            profile,
            sync_timings,
        )?;
        synchronize_if(output.device(), sync_timings)?;
        profile.ffn_ms += elapsed_ms(ffn_start);
        profile.layers += 1;
        Ok(output)
    }

    fn forward_packed_cls_profiled(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        profile: &mut XlmRobertaForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        synchronize_if(hidden_states.device(), sync_timings)?;
        let attention_start = Instant::now();
        let attention_output = self.attention.forward_packed_cls_profiled(
            hidden_states,
            seqlens,
            max_seqlen,
            profile,
            sync_timings,
        )?;
        synchronize_if(attention_output.device(), sync_timings)?;
        profile.attention_ms += elapsed_ms(attention_start);

        let ffn_start = Instant::now();
        let intermediate_output =
            self.intermediate
                .forward_profiled(&attention_output, profile, sync_timings)?;
        let output = self.output.forward_profiled(
            &intermediate_output,
            &attention_output,
            profile,
            sync_timings,
        )?;
        synchronize_if(output.device(), sync_timings)?;
        profile.ffn_ms += elapsed_ms(ffn_start);
        profile.layers += 1;
        Ok(output)
    }
}

struct XlmRobertaEncoder {
    layers: Vec<XlmRobertaLayer>,
}

impl XlmRobertaEncoder {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| XlmRobertaLayer::new(cfg, vb.pp(format!("layer.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer_module in self.layers.iter() {
            hidden_states = layer_module.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }

    fn forward_packed(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer_module in self.layers.iter() {
            hidden_states = layer_module.forward_packed(&hidden_states, seqlens, max_seqlen)?;
        }
        Ok(hidden_states)
    }

    fn forward_packed_cls(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let Some((last_layer, prefix_layers)) = self.layers.split_last() else {
            candle::bail!("packed XLM-R CLS forward requires at least one encoder layer")
        };
        let mut hidden_states = hidden_states.clone();
        for layer_module in prefix_layers {
            hidden_states = layer_module.forward_packed(&hidden_states, seqlens, max_seqlen)?;
        }
        last_layer.forward_packed_cls(&hidden_states, seqlens, max_seqlen)
    }

    fn forward_packed_profiled(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        profile: &mut XlmRobertaForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer_module in self.layers.iter() {
            hidden_states = layer_module.forward_packed_profiled(
                &hidden_states,
                seqlens,
                max_seqlen,
                profile,
                sync_timings,
            )?;
        }
        Ok(hidden_states)
    }

    fn forward_packed_cls_profiled(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        profile: &mut XlmRobertaForwardProfile,
        sync_timings: bool,
    ) -> Result<Tensor> {
        let Some((last_layer, prefix_layers)) = self.layers.split_last() else {
            candle::bail!("packed XLM-R CLS forward requires at least one encoder layer")
        };
        let mut hidden_states = hidden_states.clone();
        for layer_module in prefix_layers {
            hidden_states = layer_module.forward_packed_profiled(
                &hidden_states,
                seqlens,
                max_seqlen,
                profile,
                sync_timings,
            )?;
        }
        last_layer.forward_packed_cls_profiled(
            &hidden_states,
            seqlens,
            max_seqlen,
            profile,
            sync_timings,
        )
    }
}

pub struct XlmRobertaModel {
    encoder: XlmRobertaEncoder,
    embeddings: XlmRobertaEmbeddings,
    config: Config,
}

impl XlmRobertaModel {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let encoder = XlmRobertaEncoder::new(cfg, vb.pp("encoder"))?;
        let embeddings = XlmRobertaEmbeddings::load(vb.pp("embeddings"), cfg)?;
        Ok(Self {
            encoder,
            embeddings,
            config: cfg.clone(),
        })
    }

    pub fn kernel_backend(&self) -> (&'static str, &'static str, bool, bool) {
        let linear_backend = self
            .encoder
            .layers
            .first()
            .map(|layer| layer.intermediate.dense.backend())
            .unwrap_or("candle_matmul");
        let ffn_activation_fused = self
            .encoder
            .layers
            .first()
            .map(|layer| layer.intermediate.dense.activation_kernel_fused())
            .unwrap_or(false);
        let layernorm_backend = self.embeddings.layer_norm.backend();
        (
            linear_backend,
            layernorm_backend,
            true,
            ffn_activation_fused,
        )
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        token_type_ids: &Tensor,
    ) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward(input_ids, token_type_ids)?;
        let attention_mask = prepare_4d_attention_mask(attention_mask, DType::F32, None)?
            .to_device(hidden_states.device())?;
        self.encoder.forward(&hidden_states, &attention_mask)
    }

    pub fn forward_packed(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward_packed(input_ids, position_ids)?;
        self.encoder
            .forward_packed(&hidden_states, seqlens, max_seqlen)
    }

    /// Returns only the final CLS row for each packed sequence as `[batch, hidden]`.
    pub fn forward_packed_cls(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let hidden_states = self.embeddings.forward_packed(input_ids, position_ids)?;
        self.encoder
            .forward_packed_cls(&hidden_states, seqlens, max_seqlen)
    }

    pub fn forward_packed_profiled(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        sync_timings: bool,
    ) -> Result<(Tensor, XlmRobertaForwardProfile)> {
        synchronize_if(input_ids.device(), sync_timings)?;
        let embedding_start = Instant::now();
        let hidden_states = self.embeddings.forward_packed(input_ids, position_ids)?;
        synchronize_if(hidden_states.device(), sync_timings)?;

        let total_tokens = input_ids.dim(0)?;
        let mut profile = XlmRobertaForwardProfile {
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
            ..Default::default()
        };
        let hidden_states = self.encoder.forward_packed_profiled(
            &hidden_states,
            seqlens,
            max_seqlen,
            &mut profile,
            sync_timings,
        )?;
        Ok((hidden_states, profile))
    }

    /// Profiled variant of [`Self::forward_packed_cls`].
    pub fn forward_packed_cls_profiled(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
        sync_timings: bool,
    ) -> Result<(Tensor, XlmRobertaForwardProfile)> {
        synchronize_if(input_ids.device(), sync_timings)?;
        let embedding_start = Instant::now();
        let hidden_states = self.embeddings.forward_packed(input_ids, position_ids)?;
        synchronize_if(hidden_states.device(), sync_timings)?;

        let total_tokens = input_ids.dim(0)?;
        let batch_size = seqlens.dim(0)?.saturating_sub(1);
        let mut profile = XlmRobertaForwardProfile {
            total_tokens,
            max_seqlen,
            hidden_size: self.config.hidden_size,
            intermediate_size: self.config.intermediate_size,
            attention_heads: self.config.num_attention_heads,
            attention_head_size: self.config.hidden_size / self.config.num_attention_heads,
            linear_gflops: cls_linear_gflops(
                total_tokens,
                batch_size,
                self.config.hidden_size,
                self.config.intermediate_size,
                self.config.num_hidden_layers,
            ),
            embedding_ms: elapsed_ms(embedding_start),
            ..Default::default()
        };
        let hidden_states = self.encoder.forward_packed_cls_profiled(
            &hidden_states,
            seqlens,
            max_seqlen,
            &mut profile,
            sync_timings,
        )?;
        Ok((hidden_states, profile))
    }
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
    candle::bail!("packed XLM-R FlashAttention requires the cuda feature")
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
    (inverted_mask * get_dtype_min_val(dtype))?.to_dtype(dtype)
}

fn get_dtype_min_val(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => f32::MIN as f64,
        DType::F64 => f64::MIN,
        _ => panic!("unsupported attention mask dtype"),
    }
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
    let ffn_intermediate = hidden_size * intermediate_size;
    let ffn_output = intermediate_size * hidden_size;
    2.0 * tokens as f64
        * layers as f64
        * (qkv + attention_output + ffn_intermediate + ffn_output) as f64
        / 1_000_000_000.0
}

fn cls_linear_gflops(
    tokens: usize,
    batch_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    layers: usize,
) -> f64 {
    let qkv = hidden_size * 3 * hidden_size;
    let token_local = hidden_size * hidden_size
        + hidden_size * intermediate_size
        + intermediate_size * hidden_size;
    let prefix_rows = tokens * layers.saturating_sub(1);
    2.0 * (tokens * layers * qkv + (prefix_rows + batch_size) * token_local) as f64
        / 1_000_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_cls_selection_uses_sequence_starts() -> Result<()> {
        let device = Device::Cpu;
        let self_outputs = Tensor::new(
            &[[1f32, 2.], [3., 4.], [5., 6.], [7., 8.], [9., 10.]],
            &device,
        )?;
        let hidden_states = (&self_outputs + 100.0)?;
        let seqlens = Tensor::new(&[0u32, 2, 5], &device)?;

        let (selected_outputs, selected_residuals) =
            select_packed_cls_rows(&self_outputs, &hidden_states, &seqlens)?;

        assert_eq!(
            selected_outputs.to_vec2::<f32>()?,
            vec![vec![1., 2.], vec![5., 6.]]
        );
        assert_eq!(
            selected_residuals.to_vec2::<f32>()?,
            vec![vec![101., 102.], vec![105., 106.]]
        );
        Ok(())
    }

    #[test]
    fn last_layer_cls_tail_matches_full_token_local_tail() -> Result<()> {
        let device = Device::Cpu;
        let config = Config {
            hidden_size: 4,
            layer_norm_eps: 1e-5,
            num_attention_heads: 2,
            intermediate_size: 6,
            hidden_act: HiddenAct::Gelu,
            num_hidden_layers: 1,
            vocab_size: 8,
            max_position_embeddings: 8,
            type_vocab_size: 1,
            pad_token_id: 1,
        };
        let matrix = |rows, columns, offset: f32| {
            Tensor::from_vec(
                (0..rows * columns)
                    .map(|index| offset + (index as f32 - 7.0) / 19.0)
                    .collect::<Vec<_>>(),
                (rows, columns),
                &device,
            )
        };
        let tensors = std::collections::HashMap::from([
            (
                "attention.output.dense.weight".to_string(),
                matrix(4, 4, -0.2)?,
            ),
            (
                "attention.output.dense.bias".to_string(),
                Tensor::new(&[0.1f32, -0.2, 0.3, -0.4], &device)?,
            ),
            (
                "attention.output.LayerNorm.weight".to_string(),
                Tensor::new(&[0.8f32, 1.1, 0.9, 1.2], &device)?,
            ),
            (
                "attention.output.LayerNorm.bias".to_string(),
                Tensor::new(&[-0.1f32, 0.2, -0.3, 0.4], &device)?,
            ),
            ("intermediate.dense.weight".to_string(), matrix(6, 4, 0.15)?),
            (
                "intermediate.dense.bias".to_string(),
                Tensor::new(&[0.05f32, -0.1, 0.15, -0.2, 0.25, -0.3], &device)?,
            ),
            ("output.dense.weight".to_string(), matrix(4, 6, -0.1)?),
            (
                "output.dense.bias".to_string(),
                Tensor::new(&[-0.05f32, 0.1, -0.15, 0.2], &device)?,
            ),
            (
                "output.LayerNorm.weight".to_string(),
                Tensor::new(&[1.2f32, 0.7, 1.1, 0.9], &device)?,
            ),
            (
                "output.LayerNorm.bias".to_string(),
                Tensor::new(&[0.2f32, -0.1, 0.05, -0.15], &device)?,
            ),
        ]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let attention_output = XlmRobertaSelfOutput::new(&config, vb.pp("attention.output"))?;
        let intermediate = XlmRobertaIntermediate::new(&config, vb.pp("intermediate"))?;
        let output = XlmRobertaOutput::new(&config, vb.pp("output"))?;
        let self_outputs = matrix(5, 4, -0.35)?;
        let hidden_states = matrix(5, 4, 0.45)?;
        let seqlens = Tensor::new(&[0u32, 2, 5], &device)?;

        let full_attention = attention_output.forward(&self_outputs, &hidden_states)?;
        let full_intermediate = intermediate.forward(&full_attention)?;
        let full_tail = output.forward(&full_intermediate, &full_attention)?;
        let expected = index_select(&full_tail, &packed_cls_indices(&seqlens)?, 0)?;

        let (selected_outputs, selected_residuals) =
            select_packed_cls_rows(&self_outputs, &hidden_states, &seqlens)?;
        let selected_attention =
            attention_output.forward(&selected_outputs, &selected_residuals)?;
        let selected_intermediate = intermediate.forward(&selected_attention)?;
        let actual = output.forward(&selected_intermediate, &selected_attention)?;

        let expected = expected.to_vec2::<f32>()?;
        let actual = actual.to_vec2::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(&expected) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (&actual, &expected) in actual_row.iter().zip(expected_row) {
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "CLS tail diverged: actual={actual}, expected={expected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn packed_token_type_row_matches_zero_id_gather() -> Result<()> {
        let device = Device::Cpu;
        let table = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?;
        let embeddings = Embedding::new(table, 3);
        let zero_ids = Tensor::new(&[0u32, 0, 0], &device)?;
        let gathered = embeddings.forward(&zero_ids)?;
        let input = Tensor::zeros((3, 3), DType::F32, &device)?;
        let broadcast = input.broadcast_add(&packed_token_type_embedding(&embeddings)?)?;

        assert_eq!(broadcast.to_vec2::<f32>()?, gathered.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn cls_linear_gflops_reduces_only_final_token_local_work() {
        let full = linear_gflops(10, 4, 8, 3);
        let cls = cls_linear_gflops(10, 2, 4, 8, 3);
        let one_layer_saved_rows = 10 - 2;
        let token_local = 4 * 4 + 4 * 8 + 8 * 4;
        let expected_saved = 2.0 * (one_layer_saved_rows * token_local) as f64 / 1_000_000_000.0;

        assert!((full - cls - expected_saved).abs() < f64::EPSILON);
    }
}
