//! Packed BERT encoder with variable-length FlashAttention for Candle.
//!
//! This path is intentionally narrower than the generic Candle BERT model: it
//! serves standard post-layer-norm BERT checkpoints with absolute learned
//! positions. The caller keeps the generic padded model as the correctness
//! fallback for unsupported devices, dtypes, and attention-mask shapes.

use crate::candle_layers::{FastLayerNorm, FastLinear, HiddenAct};
use candle::{Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};
use candle_transformers::models::bert::{
    Config as BertConfig, HiddenAct as BertHiddenAct, PositionEmbeddingType,
};

#[derive(Debug)]
struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: FastLayerNorm,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder<'_>, config: &BertConfig) -> Result<Self> {
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
        })
    }

    fn forward_packed(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor> {
        let word_embeddings = self.word_embeddings.forward(input_ids)?;
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
        let position_embeddings = self.position_embeddings.forward(position_ids)?;
        let word_and_type = (&word_embeddings + token_type_embeddings)?;
        self.layer_norm
            .forward(&word_and_type, Some(&position_embeddings))
    }
}

#[derive(Debug)]
struct BertAttention {
    qkv: FastLinear,
    output: FastLinear,
    output_layer_norm: FastLayerNorm,
    num_attention_heads: usize,
    attention_head_size: usize,
    all_head_size: usize,
}

impl BertAttention {
    fn load(vb: VarBuilder<'_>, config: &BertConfig) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let all_head_size = config.num_attention_heads * attention_head_size;
        let qkv = FastLinear::load_qkv(config.hidden_size, all_head_size, vb.pp("self"))?;
        let output = FastLinear::load(
            all_head_size,
            config.hidden_size,
            vb.pp("output.dense"),
            None,
        )?;
        let output_layer_norm = FastLayerNorm::load(
            vb.pp("output.LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps,
        )?;
        Ok(Self {
            qkv,
            output,
            output_layer_norm,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            all_head_size,
        })
    }

    fn forward_packed(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let total_tokens = hidden_states.dim(0)?;
        let qkv = self.qkv.forward(hidden_states)?.reshape((
            total_tokens,
            self.num_attention_heads * 3,
            self.attention_head_size,
        ))?;
        let qkv = qkv.chunk(3, 1)?;
        let attention_output = flash_attn_varlen(
            &qkv[0],
            &qkv[1],
            &qkv[2],
            seqlens,
            max_seqlen,
            self.attention_head_size,
        )?
        .reshape((total_tokens, self.all_head_size))?;
        let attention_output = self.output.forward(&attention_output)?;
        self.output_layer_norm
            .forward(&attention_output, Some(hidden_states))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BertIntermediateActivation {
    ExactGelu,
    Fused(HiddenAct),
}

impl BertIntermediateActivation {
    fn from_config(value: BertHiddenAct) -> Self {
        match value {
            // Candle's generic BERT model and Hugging Face both interpret
            // `gelu` as the erf formulation. TEI intentionally approximates
            // it for fusion, but this path preserves checkpoint semantics.
            BertHiddenAct::Gelu => Self::ExactGelu,
            BertHiddenAct::GeluApproximate => Self::Fused(HiddenAct::Gelu),
            BertHiddenAct::Relu => Self::Fused(HiddenAct::Relu),
        }
    }

    fn fused_activation(self) -> Option<HiddenAct> {
        match self {
            Self::ExactGelu => None,
            Self::Fused(activation) => Some(activation),
        }
    }

    fn finish(self, hidden_states: Tensor) -> Result<Tensor> {
        match self {
            Self::ExactGelu => hidden_states.gelu_erf(),
            Self::Fused(_) => Ok(hidden_states),
        }
    }
}

#[derive(Debug)]
struct BertIntermediate {
    dense: FastLinear,
    activation: BertIntermediateActivation,
}

impl BertIntermediate {
    fn load(vb: VarBuilder<'_>, config: &BertConfig) -> Result<Self> {
        let activation = BertIntermediateActivation::from_config(config.hidden_act);
        let dense = FastLinear::load(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("dense"),
            activation.fused_activation(),
        )?;
        Ok(Self { dense, activation })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.activation.finish(self.dense.forward(hidden_states)?)
    }
}

#[derive(Debug)]
struct BertOutput {
    dense: FastLinear,
    layer_norm: FastLayerNorm,
}

impl BertOutput {
    fn load(vb: VarBuilder<'_>, config: &BertConfig) -> Result<Self> {
        let dense = FastLinear::load(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("dense"),
            None,
        )?;
        let layer_norm = FastLayerNorm::load(
            vb.pp("LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps,
        )?;
        Ok(Self { dense, layer_norm })
    }

    fn forward(&self, hidden_states: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layer_norm.forward(&hidden_states, Some(residual))
    }
}

#[derive(Debug)]
struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn load(vb: VarBuilder<'_>, config: &BertConfig) -> Result<Self> {
        Ok(Self {
            attention: BertAttention::load(vb.pp("attention"), config)?,
            intermediate: BertIntermediate::load(vb.pp("intermediate"), config)?,
            output: BertOutput::load(vb.pp("output"), config)?,
        })
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
}

#[derive(Debug)]
struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    fn load(vb: VarBuilder<'_>, config: &BertConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(vb.pp(format!("layer.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    fn forward_packed(
        &self,
        hidden_states: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states.clone();
        for layer in &self.layers {
            hidden_states = layer.forward_packed(&hidden_states, seqlens, max_seqlen)?;
        }
        Ok(hidden_states)
    }
}

/// Standard post-LN BERT backbone evaluated over unpadded token rows.
#[derive(Debug)]
pub(crate) struct PackedBertModel {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
}

impl PackedBertModel {
    pub(crate) fn load(vb: VarBuilder<'_>, config: &BertConfig) -> Result<Self> {
        validate_config(config)?;
        let vb = if vb.contains_tensor("bert.embeddings.word_embeddings.weight") {
            vb.pp("bert")
        } else {
            vb
        };
        Ok(Self {
            embeddings: BertEmbeddings::load(vb.pp("embeddings"), config)?,
            encoder: BertEncoder::load(vb.pp("encoder"), config)?,
        })
    }

    pub(crate) fn forward_packed(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: &Tensor,
        seqlens: &Tensor,
        max_seqlen: usize,
    ) -> Result<Tensor> {
        let hidden_states =
            self.embeddings
                .forward_packed(input_ids, token_type_ids, position_ids)?;
        self.encoder
            .forward_packed(&hidden_states, seqlens, max_seqlen)
    }
}

fn validate_config(config: &BertConfig) -> Result<()> {
    if config.position_embedding_type != PositionEmbeddingType::Absolute {
        candle::bail!("packed BERT supports only absolute position embeddings")
    }
    if config.num_attention_heads == 0
        || !config
            .hidden_size
            .is_multiple_of(config.num_attention_heads)
    {
        candle::bail!(
            "packed BERT hidden_size {} must be divisible by non-zero num_attention_heads {}",
            config.hidden_size,
            config.num_attention_heads
        )
    }
    if config.num_hidden_layers == 0 {
        candle::bail!("packed BERT requires at least one encoder layer")
    }
    if config.type_vocab_size == 0 {
        candle::bail!("packed BERT type_vocab_size must be greater than zero")
    }
    Ok(())
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
    candle_flash_attn::flash_attn_varlen(
        query,
        key,
        value,
        seqlens,
        seqlens,
        max_seqlen,
        max_seqlen,
        1f32 / f32::sqrt(head_size as f32),
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
    candle::bail!("packed BERT FlashAttention requires the cuda feature")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device};

    #[test]
    fn bert_activation_preserves_exact_gelu_semantics() -> Result<()> {
        assert_eq!(
            BertIntermediateActivation::from_config(BertHiddenAct::Gelu),
            BertIntermediateActivation::ExactGelu
        );
        assert_eq!(
            BertIntermediateActivation::from_config(BertHiddenAct::GeluApproximate),
            BertIntermediateActivation::Fused(HiddenAct::Gelu)
        );
        assert_eq!(
            BertIntermediateActivation::from_config(BertHiddenAct::Relu),
            BertIntermediateActivation::Fused(HiddenAct::Relu)
        );

        let input = Tensor::new(&[-2.0f32, -0.5, 0.5, 2.0], &Device::Cpu)?;
        let exact = BertIntermediateActivation::ExactGelu
            .finish(input.clone())?
            .to_vec1::<f32>()?;
        let approximate = input.gelu()?.to_vec1::<f32>()?;
        assert!(exact
            .iter()
            .zip(approximate)
            .any(|(exact, approximate)| (exact - approximate).abs() > 1e-6));
        Ok(())
    }

    #[test]
    fn embedding_forward_uses_supplied_positions_and_token_types() -> Result<()> {
        let hidden_size = 4;
        let tensors = std::collections::HashMap::from([
            (
                "word_embeddings.weight".to_string(),
                Tensor::new(
                    &[
                        [0.1f32, 0.2, 0.3, 0.4],
                        [0.5, 0.6, 0.7, 0.8],
                        [0.9, 1.0, 1.1, 1.2],
                    ],
                    &Device::Cpu,
                )?,
            ),
            (
                "position_embeddings.weight".to_string(),
                Tensor::new(
                    &[
                        [0.0f32, 0.1, 0.2, 0.3],
                        [0.4, 0.3, 0.2, 0.1],
                        [0.8, 0.7, 0.6, 0.5],
                    ],
                    &Device::Cpu,
                )?,
            ),
            (
                "token_type_embeddings.weight".to_string(),
                Tensor::new(
                    &[[0.0f32, 0.0, 0.0, 0.0], [0.3, -0.2, 0.1, -0.4]],
                    &Device::Cpu,
                )?,
            ),
            (
                "LayerNorm.weight".to_string(),
                Tensor::ones(hidden_size, DType::F32, &Device::Cpu)?,
            ),
            (
                "LayerNorm.bias".to_string(),
                Tensor::zeros(hidden_size, DType::F32, &Device::Cpu)?,
            ),
        ]);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu);
        let config = BertConfig {
            vocab_size: 3,
            hidden_size,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 8,
            hidden_act: BertHiddenAct::Gelu,
            hidden_dropout_prob: 0.0,
            max_position_embeddings: 3,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            position_embedding_type: PositionEmbeddingType::Absolute,
            use_cache: false,
            classifier_dropout: None,
            model_type: Some("bert".to_string()),
        };
        let embeddings = BertEmbeddings::load(vb, &config)?;
        let input_ids = Tensor::new(&[1u32, 2], &Device::Cpu)?;
        let zero_types = Tensor::new(&[0u32, 0], &Device::Cpu)?;
        let mixed_types = Tensor::new(&[0u32, 1], &Device::Cpu)?;
        let forward_positions = Tensor::new(&[0u32, 1], &Device::Cpu)?;
        let reverse_positions = Tensor::new(&[1u32, 0], &Device::Cpu)?;

        let baseline = embeddings
            .forward_packed(&input_ids, &zero_types, &forward_positions)?
            .to_vec2::<f32>()?;
        let typed = embeddings
            .forward_packed(&input_ids, &mixed_types, &forward_positions)?
            .to_vec2::<f32>()?;
        let repositioned = embeddings
            .forward_packed(&input_ids, &zero_types, &reverse_positions)?
            .to_vec2::<f32>()?;

        assert_ne!(baseline[1], typed[1]);
        assert_ne!(baseline, repositioned);
        Ok(())
    }
}
