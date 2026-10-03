//! SPLADE masked-language-model head and sparse output conversion.

use anyhow::{Context, Result};
use candle::{CpuStorage, DType, Device, Storage, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{Config as BertConfig, HiddenAct as BertHiddenAct};
use half::{bf16, f16};

use crate::candle_layers::{FastLayerNorm, FastLinear, HiddenAct};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CandleSparseEmbedding {
    pub indices: Vec<i32>,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub(crate) struct CandleBertSpladeHead {
    transform: FastLinear,
    transform_activation: BertHiddenAct,
    transform_layer_norm: FastLayerNorm,
    decoder: FastLinear,
    vocab_size: usize,
}

impl CandleBertSpladeHead {
    pub(crate) fn load(vb: VarBuilder<'_>, config: &BertConfig) -> Result<Self> {
        let transform = FastLinear::load(
            config.hidden_size,
            config.hidden_size,
            vb.pp("cls.predictions.transform.dense"),
            None,
        )
        .context("load BERT SPLADE prediction transform")?;
        let transform_layer_norm = FastLayerNorm::load(
            vb.pp("cls.predictions.transform.LayerNorm"),
            config.hidden_size,
            config.layer_norm_eps,
        )
        .context("load BERT SPLADE prediction layer norm")?;

        // PyTorch-to-safetensors conversion may omit either alias of a tied
        // tensor. Accept both canonical BERT MLM layouts without synthesizing
        // weights or changing checkpoint semantics.
        let decoder_weight = if vb.contains_tensor("cls.predictions.decoder.weight") {
            vb.pp("cls.predictions.decoder")
                .get((config.vocab_size, config.hidden_size), "weight")?
        } else {
            vb.pp("bert.embeddings.word_embeddings")
                .get((config.vocab_size, config.hidden_size), "weight")?
        };
        let decoder_bias = if vb.contains_tensor("cls.predictions.bias") {
            vb.pp("cls.predictions").get(config.vocab_size, "bias")?
        } else {
            vb.pp("cls.predictions.decoder")
                .get(config.vocab_size, "bias")?
        };
        let decoder = FastLinear::new(decoder_weight, Some(decoder_bias), Some(HiddenAct::Relu));

        Ok(Self {
            transform,
            transform_activation: config.hidden_act,
            transform_layer_norm,
            decoder,
            vocab_size: config.vocab_size,
        })
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Apply the BERT MLM head through its ReLU to
    /// `[batch, tokens, hidden]` states. The caller applies `log1p` after
    /// sequence max-pooling: `log1p(relu(x))` is monotone, so moving the log
    /// after max is exact and avoids a vocabulary-wide token-level kernel.
    pub(crate) fn forward_activated(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let transformed = self.transform.forward(hidden_states)?;
        let transformed = match self.transform_activation {
            BertHiddenAct::Gelu => transformed.gelu_erf()?,
            BertHiddenAct::GeluApproximate => transformed.gelu()?,
            BertHiddenAct::Relu => transformed.relu()?,
        };
        let transformed = self.transform_layer_norm.forward(&transformed, None)?;
        Ok(self.decoder.forward(&transformed)?)
    }
}

pub(crate) fn sparse_embeddings_from_dense(
    dense_embeddings: &Tensor,
) -> Result<Vec<CandleSparseEmbedding>> {
    let (batch_size, vocab_size) = dense_embeddings
        .dims2()
        .context("SPLADE pooled vocabulary weights must have shape [batch, vocab]")?;
    // Preserve the pooled compute dtype across the device-to-host transfer.
    // F16 and BF16 widen exactly to F32, so scanning their CPU storage avoids
    // a vocabulary-wide GPU cast and halves the readback bytes.
    let host_embeddings = dense_embeddings.to_device(&Device::Cpu)?;
    if let Some((start, end)) = host_embeddings
        .is_contiguous()
        .then(|| host_embeddings.layout().contiguous_offsets())
        .flatten()
    {
        let (storage, _) = host_embeddings.storage_and_layout();
        match &*storage {
            Storage::Cpu(CpuStorage::F16(values)) => {
                return sparse_embeddings_from_contiguous_values(
                    values
                        .get(start..end)
                        .context("read contiguous F16 SPLADE vocabulary weights")?,
                    batch_size,
                    vocab_size,
                    f16::to_f32,
                );
            }
            Storage::Cpu(CpuStorage::BF16(values)) => {
                return sparse_embeddings_from_contiguous_values(
                    values
                        .get(start..end)
                        .context("read contiguous BF16 SPLADE vocabulary weights")?,
                    batch_size,
                    vocab_size,
                    bf16::to_f32,
                );
            }
            Storage::Cpu(CpuStorage::F32(values)) => {
                return sparse_embeddings_from_contiguous_values(
                    values
                        .get(start..end)
                        .context("read contiguous F32 SPLADE vocabulary weights")?,
                    batch_size,
                    vocab_size,
                    |value| value,
                );
            }
            _ => {}
        }
    }

    let rows = host_embeddings
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()
        .context("read SPLADE pooled vocabulary weights")?;
    rows.into_iter()
        .map(sparse_embedding_from_f32_values)
        .collect()
}

fn sparse_embeddings_from_contiguous_values<T: Copy>(
    values: &[T],
    batch_size: usize,
    vocab_size: usize,
    to_f32: impl Fn(T) -> f32,
) -> Result<Vec<CandleSparseEmbedding>> {
    let expected_values = batch_size
        .checked_mul(vocab_size)
        .context("SPLADE pooled vocabulary size overflow")?;
    if values.len() != expected_values {
        anyhow::bail!(
            "SPLADE pooled vocabulary storage mismatch: values={} expected={expected_values}",
            values.len()
        )
    }

    let mut sparse_embeddings = Vec::with_capacity(batch_size);
    for row_index in 0..batch_size {
        let start = row_index * vocab_size;
        let end = start + vocab_size;
        sparse_embeddings.push(sparse_embedding_from_f32_values(
            values[start..end].iter().copied().map(&to_f32),
        )?);
    }
    Ok(sparse_embeddings)
}

fn sparse_embedding_from_f32_values(
    values: impl IntoIterator<Item = f32>,
) -> Result<CandleSparseEmbedding> {
    let mut indices = Vec::new();
    let mut sparse_values = Vec::new();
    for (index, value) in values.into_iter().enumerate() {
        if !value.is_finite() {
            anyhow::bail!("SPLADE produced a non-finite vocabulary weight at {index}");
        }
        if value > 0.0 {
            indices.push(i32::try_from(index).context("SPLADE vocabulary exceeds i32")?);
            sparse_values.push(value);
        }
    }
    Ok(CandleSparseEmbedding {
        indices,
        values: sparse_values,
    })
}

/// Apply true `log1p` semantics to non-negative pooled SPLADE weights.
///
/// Candle does not expose a tensor `log1p` operation. Computing `(x + 1).log()`
/// in F16/BF16 first rounds small positive weights away. Widen the epilogue and
/// use the compensated identity `log(u) * x / (u - 1)`, with the `u == 1`
/// limit selected explicitly. The infinity branch preserves the existing
/// non-finite validation behavior without turning positive infinity into NaN.
fn splade_log1p_nonnegative(pooled: &Tensor) -> Result<Tensor> {
    let output_dtype = pooled.dtype();
    let compute_dtype = match output_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    let x = pooled.to_dtype(compute_dtype)?;
    let one_plus_x = (&x + 1.0)?;
    let denominator = (&one_plus_x - 1.0)?;
    let compensated = ((one_plus_x.log()? * &x)? / &denominator)?;
    let stable = one_plus_x.eq(1.0)?.where_cond(&x, &compensated)?;
    let stable = x.eq(f64::INFINITY)?.where_cond(&x, &stable)?;
    Ok(stable.to_dtype(output_dtype)?)
}

/// Pool non-negative MLM activations across the sequence dimension.
///
/// Padding is zeroed before max-pooling. Applying `log1p` after the max is
/// exactly equivalent to applying it token-wise first because the activation
/// is monotone, and avoids a token-by-vocabulary elementwise kernel.
pub(crate) fn pool_splade_activations(
    activated: &Tensor,
    attention_mask: &Tensor,
) -> Result<Tensor> {
    let mask = attention_mask.unsqueeze(2)?.to_dtype(activated.dtype())?;
    let masked = activated.broadcast_mul(&mask)?;
    let max_weights = masked.max(1)?;
    splade_log1p_nonnegative(&max_weights)
}

/// Max-pool non-negative packed MLM activations using host-side sequence
/// lengths, then apply `log1p` once per pooled vocabulary row.
pub(crate) fn pool_packed_splade_activations(
    activated: &Tensor,
    seq_lengths: &[usize],
) -> Result<Tensor> {
    let (_, vocab_size) = validate_packed_splade_activations(activated, seq_lengths)?;

    let max_weights = if seq_lengths
        .first()
        .is_some_and(|first| seq_lengths.iter().all(|length| length == first))
    {
        activated
            .reshape((seq_lengths.len(), seq_lengths[0], vocab_size))?
            .max(1)?
    } else {
        let mut cursor = 0usize;
        let mut rows = Vec::with_capacity(seq_lengths.len());
        for length in seq_lengths {
            rows.push(activated.narrow(0, cursor, *length)?.max_keepdim(0)?);
            cursor += *length;
        }
        Tensor::cat(&rows, 0)?
    };
    splade_log1p_nonnegative(&max_weights)
}

fn validate_packed_splade_activations(
    activated: &Tensor,
    seq_lengths: &[usize],
) -> Result<(usize, usize)> {
    let (total_tokens, vocab_size) = activated
        .dims2()
        .context("packed SPLADE activations must have shape [tokens, vocab]")?;
    if seq_lengths.is_empty() {
        anyhow::bail!("packed SPLADE sequence lengths are empty")
    }
    if seq_lengths.contains(&0) {
        anyhow::bail!("packed SPLADE sequence lengths must be positive")
    }
    let expected_tokens = seq_lengths.iter().try_fold(0usize, |total, length| {
        total
            .checked_add(*length)
            .context("packed SPLADE token count overflow")
    })?;
    if expected_tokens != total_tokens {
        anyhow::bail!(
            "packed SPLADE token count mismatch: activations={total_tokens} lengths={expected_tokens}"
        )
    }
    if vocab_size == 0 {
        anyhow::bail!("packed SPLADE vocabulary is empty")
    }

    Ok((total_tokens, vocab_size))
}

fn validate_packed_splade_offsets(
    activated: &Tensor,
    seqlens: &Tensor,
    seq_lengths: &[usize],
) -> Result<()> {
    let offset_count = seqlens
        .dims1()
        .context("packed SPLADE cumulative offsets must have shape [batch + 1]")?;
    let expected_offsets = seq_lengths
        .len()
        .checked_add(1)
        .context("packed SPLADE cumulative offset count overflow")?;
    if offset_count != expected_offsets {
        anyhow::bail!(
            "packed SPLADE cumulative offset count mismatch: offsets={offset_count} expected={expected_offsets}"
        )
    }
    if seqlens.dtype() != DType::U32 {
        anyhow::bail!(
            "packed SPLADE cumulative offsets must use U32, got {:?}",
            seqlens.dtype()
        )
    }
    if !activated.device().same_device(seqlens.device()) {
        anyhow::bail!("packed SPLADE activations and cumulative offsets must use the same device")
    }

    Ok(())
}

#[cfg(feature = "cuda")]
fn can_use_fused_packed_splade_pool(
    activated: &Tensor,
    seqlens: &Tensor,
    seq_lengths: &[usize],
) -> bool {
    let Ok((total_tokens, vocab_size)) = activated.dims2() else {
        return false;
    };
    activated.device().is_cuda()
        && activated.dtype() == DType::F16
        && activated.is_contiguous()
        && seqlens.is_contiguous()
        && total_tokens <= u32::MAX as usize
        && vocab_size <= i32::MAX as usize
        // CUDA grid.y is capped at 65,535 for the supported deployment GPUs.
        && seq_lengths.len() <= 65_535
}

/// Dispatch packed SPLADE pooling to the coalesced CUDA kernel when its strict
/// contract is met. CPU, non-F16, and non-contiguous inputs retain the generic
/// Candle implementation as a correctness fallback.
pub(crate) fn pool_packed_splade_activations_dispatch(
    activated: &Tensor,
    seqlens: &Tensor,
    seq_lengths: &[usize],
) -> Result<Tensor> {
    validate_packed_splade_activations(activated, seq_lengths)?;
    validate_packed_splade_offsets(activated, seqlens, seq_lengths)?;

    #[cfg(feature = "cuda")]
    if can_use_fused_packed_splade_pool(activated, seqlens, seq_lengths) {
        return candle_splade_pool::segmented_max_log1p(activated, seqlens)
            .context("run fused packed SPLADE pooling");
    }

    pool_packed_splade_activations(activated, seq_lengths)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_native_sparse_conversion<T: candle::WithDType + Copy>(
        values: Vec<T>,
        expected_values: [Vec<f32>; 2],
    ) -> Result<()> {
        let dense = Tensor::from_vec(values, (2, 7), &Device::Cpu)?;
        let sparse = sparse_embeddings_from_dense(&dense)?;

        assert_eq!(sparse.len(), 2);
        assert_eq!(sparse[0].indices, vec![3, 4, 6]);
        assert_eq!(sparse[1].indices, vec![0, 5, 6]);
        for (actual, expected) in sparse.iter().zip(expected_values) {
            assert_eq!(
                actual
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    fn assert_non_finite_rejected<T: candle::WithDType + Copy>(
        zero: T,
        non_finite: T,
    ) -> Result<()> {
        let dense = Tensor::from_vec(vec![zero, non_finite], (1, 2), &Device::Cpu)?;
        let error = sparse_embeddings_from_dense(&dense).unwrap_err();
        assert!(error
            .to_string()
            .contains("non-finite vocabulary weight at 1"));
        Ok(())
    }

    #[test]
    fn sparse_conversion_keeps_positive_finite_weights() -> Result<()> {
        let dense = Tensor::new(
            &[[0.0_f32, 1.5, -0.0, 2.25], [3.0, 0.0, 0.5, 0.0]],
            &Device::Cpu,
        )?;

        let sparse = sparse_embeddings_from_dense(&dense)?;

        assert_eq!(
            sparse,
            vec![
                CandleSparseEmbedding {
                    indices: vec![1, 3],
                    values: vec![1.5, 2.25],
                },
                CandleSparseEmbedding {
                    indices: vec![0, 2],
                    values: vec![3.0, 0.5],
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn sparse_conversion_rejects_non_finite_weights() -> Result<()> {
        let dense = Tensor::new(&[[0.0_f32, f32::NAN]], &Device::Cpu)?;

        let error = sparse_embeddings_from_dense(&dense).unwrap_err();

        assert!(error.to_string().contains("non-finite"));
        Ok(())
    }

    #[test]
    fn sparse_conversion_preserves_native_float_bits_and_order() -> Result<()> {
        let f32_subnormal = f32::from_bits(1);
        assert_native_sparse_conversion(
            vec![
                0.0,
                -0.0,
                -1.0,
                f32_subnormal,
                1.5,
                -f32_subnormal,
                2.0,
                3.0,
                0.0,
                -2.0,
                -0.0,
                -f32_subnormal,
                f32_subnormal,
                4.0,
            ],
            [vec![f32_subnormal, 1.5, 2.0], vec![3.0, f32_subnormal, 4.0]],
        )?;

        let f16_subnormal = f16::from_bits(1);
        assert_native_sparse_conversion(
            vec![
                f16::ZERO,
                f16::NEG_ZERO,
                f16::from_f32(-1.0),
                f16_subnormal,
                f16::from_f32(1.5),
                -f16_subnormal,
                f16::from_f32(2.0),
                f16::from_f32(3.0),
                f16::ZERO,
                f16::from_f32(-2.0),
                f16::NEG_ZERO,
                -f16_subnormal,
                f16_subnormal,
                f16::from_f32(4.0),
            ],
            [
                vec![f16_subnormal.to_f32(), 1.5, 2.0],
                vec![3.0, f16_subnormal.to_f32(), 4.0],
            ],
        )?;

        let bf16_subnormal = bf16::from_bits(1);
        assert_native_sparse_conversion(
            vec![
                bf16::ZERO,
                bf16::NEG_ZERO,
                bf16::from_f32(-1.0),
                bf16_subnormal,
                bf16::from_f32(1.5),
                -bf16_subnormal,
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::ZERO,
                bf16::from_f32(-2.0),
                bf16::NEG_ZERO,
                -bf16_subnormal,
                bf16_subnormal,
                bf16::from_f32(4.0),
            ],
            [
                vec![bf16_subnormal.to_f32(), 1.5, 2.0],
                vec![3.0, bf16_subnormal.to_f32(), 4.0],
            ],
        )?;

        Ok(())
    }

    #[test]
    fn sparse_conversion_rejects_all_native_non_finite_values() -> Result<()> {
        for non_finite in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_non_finite_rejected(0.0f32, non_finite)?;
        }
        for non_finite in [f16::NAN, f16::INFINITY, f16::NEG_INFINITY] {
            assert_non_finite_rejected(f16::ZERO, non_finite)?;
        }
        for non_finite in [bf16::NAN, bf16::INFINITY, bf16::NEG_INFINITY] {
            assert_non_finite_rejected(bf16::ZERO, non_finite)?;
        }
        Ok(())
    }

    #[test]
    fn sparse_conversion_preserves_non_contiguous_row_order() -> Result<()> {
        let base = Tensor::from_vec(
            vec![
                f16::ZERO,
                f16::from_f32(3.0),
                f16::from_f32(1.0),
                f16::ZERO,
                f16::from_f32(-1.0),
                f16::from_f32(4.0),
                f16::from_f32(2.0),
                f16::NEG_ZERO,
            ],
            (4, 2),
            &Device::Cpu,
        )?;
        let dense = base.t()?;
        assert!(!dense.is_contiguous());

        assert_eq!(
            sparse_embeddings_from_dense(&dense)?,
            vec![
                CandleSparseEmbedding {
                    indices: vec![1, 3],
                    values: vec![1.0, 2.0],
                },
                CandleSparseEmbedding {
                    indices: vec![0, 2],
                    values: vec![3.0, 4.0],
                },
            ]
        );
        Ok(())
    }

    #[test]
    fn pooling_masks_padding_and_matches_tokenwise_log1p() -> Result<()> {
        let activated = Tensor::new(
            &[
                [[0.0_f32, 2.0, 1.0], [4.0, 1.0, 3.0], [99.0, 99.0, 99.0]],
                [[1.0, 0.0, 5.0], [9.0, 9.0, 9.0], [8.0, 8.0, 8.0]],
            ],
            &Device::Cpu,
        )?;
        let mask = Tensor::new(&[[1_u32, 1, 0], [1, 0, 0]], &Device::Cpu)?;

        let optimized = pool_splade_activations(&activated, &mask)?;
        let direct = (activated + 1.0)?
            .log()?
            .broadcast_mul(&mask.unsqueeze(2)?.to_dtype(DType::F32)?)?
            .max(1)?;

        assert_eq!(optimized.to_vec2::<f32>()?, direct.to_vec2::<f32>()?);
        let pooled = optimized.to_vec2::<f32>()?;
        assert!((pooled[0][0] - 4.0_f32.ln_1p()).abs() < 1e-6);
        assert!((pooled[1][2] - 5.0_f32.ln_1p()).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn packed_pooling_matches_per_sequence_tokenwise_log1p() -> Result<()> {
        let activated = Tensor::new(
            &[
                [0.0f32, 2.0, 1.0],
                [4.0, 1.0, 3.0],
                [1.0, 0.0, 5.0],
                [7.0, 3.0, 2.0],
                [2.0, 8.0, 4.0],
            ],
            &Device::Cpu,
        )?;

        let optimized = pool_packed_splade_activations(&activated, &[2, 3])?;
        let tokenwise = (activated + 1.0)?.log()?;
        let expected = Tensor::cat(
            &[
                tokenwise.narrow(0, 0, 2)?.max_keepdim(0)?,
                tokenwise.narrow(0, 2, 3)?.max_keepdim(0)?,
            ],
            0,
        )?;

        assert_eq!(optimized.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn packed_pooling_preserves_true_log1p_for_tiny_f16_weights() -> Result<()> {
        let values = vec![
            f16::from_bits(1),
            f16::from_f32(2.0f32.powi(-14)),
            f16::from_f32(2.0f32.powi(-12)),
            f16::from_f32(2.0f32.powi(-11)),
            f16::from_f32(2.0f32.powi(-10)),
        ];
        let activated = Tensor::from_vec(values.clone(), (1, values.len()), &Device::Cpu)?;

        let actual = pool_packed_splade_activations(&activated, &[1])?
            .to_vec2::<f16>()?
            .remove(0);

        for (index, (actual, input)) in actual.iter().zip(values).enumerate() {
            let expected = f16::from_f32(input.to_f32().ln_1p());
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "tiny log1p mismatch at index {index}: input={}",
                input.to_f32()
            );
            assert!(*actual > f16::ZERO);
        }
        Ok(())
    }

    #[test]
    fn packed_pooling_uses_uniform_length_fast_path() -> Result<()> {
        let activated = Tensor::new(
            &[[0.0f32, 2.0], [4.0, 1.0], [1.0, 5.0], [7.0, 3.0]],
            &Device::Cpu,
        )?;

        let optimized = pool_packed_splade_activations(&activated, &[2, 2])?;
        let expected = activated.reshape((2, 2, 2))?.max(1)?;
        let expected = (expected + 1.0)?.log()?;

        assert_eq!(optimized.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn packed_pooling_rejects_mismatched_lengths() -> Result<()> {
        let activated = Tensor::zeros((3, 2), DType::F32, &Device::Cpu)?;

        let error = pool_packed_splade_activations(&activated, &[1, 1])
            .unwrap_err()
            .to_string();

        assert!(error.contains("token count mismatch"));
        Ok(())
    }

    #[test]
    fn packed_pooling_dispatch_falls_back_on_cpu() -> Result<()> {
        let activated = Tensor::new(
            &[
                [0.0f32, 2.0, 1.0],
                [4.0, 1.0, 3.0],
                [1.0, 0.0, 5.0],
                [7.0, 3.0, 2.0],
                [2.0, 8.0, 4.0],
            ],
            &Device::Cpu,
        )?;
        let seqlens = Tensor::new(&[0u32, 2, 5], &Device::Cpu)?;

        let dispatched = pool_packed_splade_activations_dispatch(&activated, &seqlens, &[2, 3])?;
        let reference = pool_packed_splade_activations(&activated, &[2, 3])?;

        assert_eq!(dispatched.dims(), &[2, 3]);
        assert_eq!(dispatched.to_vec2::<f32>()?, reference.to_vec2::<f32>()?);
        Ok(())
    }

    #[test]
    fn packed_pooling_dispatch_validates_offsets() -> Result<()> {
        let activated = Tensor::zeros((3, 2), DType::F32, &Device::Cpu)?;
        let short_offsets = Tensor::new(&[0u32, 3], &Device::Cpu)?;
        let float_offsets = Tensor::new(&[0.0f32, 1.0, 3.0], &Device::Cpu)?;
        let matrix_offsets = Tensor::new(&[[0u32, 1, 3]], &Device::Cpu)?;

        let error = pool_packed_splade_activations_dispatch(&activated, &short_offsets, &[1, 2])
            .unwrap_err()
            .to_string();
        assert!(error.contains("offset count mismatch"));

        let error = pool_packed_splade_activations_dispatch(&activated, &float_offsets, &[1, 2])
            .unwrap_err()
            .to_string();
        assert!(error.contains("must use U32"));

        let error = pool_packed_splade_activations_dispatch(&activated, &matrix_offsets, &[1, 2])
            .unwrap_err()
            .to_string();
        assert!(error.contains("shape [batch + 1]"));
        Ok(())
    }

    #[test]
    fn packed_pooling_preserves_full_vocabulary_tail_without_normalizing() -> Result<()> {
        const VOCAB_SIZE: usize = 30_522;
        let mut values = vec![0.0f32; 2 * VOCAB_SIZE];
        values[VOCAB_SIZE - 1] = 2.0;
        values[2 * VOCAB_SIZE - 1] = 5.0;
        values[VOCAB_SIZE] = 3.0;
        let activated = Tensor::from_vec(values, (2, VOCAB_SIZE), &Device::Cpu)?;

        let pooled = pool_packed_splade_activations(&activated, &[2])?;
        let row = pooled.to_vec2::<f32>()?.remove(0);

        assert_eq!(pooled.dims(), &[1, VOCAB_SIZE]);
        assert!((row[0] - 3.0f32.ln_1p()).abs() < 1e-6);
        assert!((row[VOCAB_SIZE - 1] - 5.0f32.ln_1p()).abs() < 1e-6);
        assert!(row.iter().map(|value| value * value).sum::<f32>() > 1.0);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn fused_packed_pooling_matches_host_reference_and_handles_vocab_tail() -> Result<()> {
        use half::f16;

        const TOKENS: usize = 5;
        const VOCAB_SIZE: usize = 30_522;
        let device = Device::new_cuda(0)?;
        let values = (0..TOKENS)
            .flat_map(|token| {
                (0..VOCAB_SIZE)
                    .map(move |vocab| f16::from_f32(((token * 11 + vocab % 23) % 31) as f32 / 8.0))
            })
            .collect::<Vec<_>>();
        let activated = Tensor::from_vec(values, (TOKENS, VOCAB_SIZE), &device)?;
        let seqlens = Tensor::new(&[0u32, 2, 5], &device)?;

        let reference = pool_packed_splade_activations(&activated, &[2, 3])?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let fused = pool_packed_splade_activations_dispatch(&activated, &seqlens, &[2, 3])?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].len(), VOCAB_SIZE);
        for batch in 0..fused.len() {
            for vocab in 0..VOCAB_SIZE {
                assert_eq!(
                    fused[batch][vocab], reference[batch][vocab],
                    "batch={batch} vocab={vocab}"
                );
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn fused_packed_pooling_preserves_true_log1p_for_tiny_f16_weights() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let values = vec![
            f16::from_bits(1),
            f16::from_f32(2.0f32.powi(-14)),
            f16::from_f32(2.0f32.powi(-12)),
            f16::from_f32(2.0f32.powi(-11)),
            f16::from_f32(2.0f32.powi(-10)),
        ];
        let activated = Tensor::from_vec(values.clone(), (1, values.len()), &device)?;
        let seqlens = Tensor::new(&[0u32, 1], &device)?;

        let actual = pool_packed_splade_activations_dispatch(&activated, &seqlens, &[1])?
            .to_device(&Device::Cpu)?
            .to_vec2::<f16>()?
            .remove(0);

        for (index, (actual, input)) in actual.iter().zip(values).enumerate() {
            let expected = f16::from_f32(input.to_f32().ln_1p());
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "fused tiny log1p mismatch at index {index}: input={}",
                input.to_f32()
            );
            assert!(*actual > f16::ZERO);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn fused_packed_pooling_preserves_batch_one_extrema_nan_and_infinity() -> Result<()> {
        use half::f16;

        let device = Device::new_cuda(0)?;
        let values = vec![
            f16::from_f32(5.0),
            f16::from_f32(1.0),
            f16::NAN,
            f16::INFINITY,
            f16::from_f32(0.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(2.0),
            f16::from_f32(4.0),
            f16::from_f32(1.0),
            f16::from_f32(1.0),
            f16::from_f32(7.0),
            f16::from_f32(3.0),
            f16::from_f32(2.0),
            f16::from_f32(6.0),
        ];
        let activated = Tensor::from_vec(values, (3, 5), &device)?;
        let seqlens = Tensor::new(&[0u32, 3], &device)?;

        let reference = pool_packed_splade_activations(&activated, &[3])?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let fused = pool_packed_splade_activations_dispatch(&activated, &seqlens, &[3])?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;

        assert_eq!(fused[0][0], reference[0][0]);
        assert_eq!(fused[0][1], reference[0][1]);
        assert!(fused[0][2].is_nan() && reference[0][2].is_nan());
        assert!(fused[0][3].is_infinite() && reference[0][3].is_infinite());
        assert_eq!(fused[0][4], reference[0][4]);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn fused_packed_pooling_rejects_invalid_device_offsets_without_oob_reads() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let activated = Tensor::zeros((4, 3), DType::F16, &device)?;

        for offsets in [[1u32, 4], [0, 3], [0, 0], [0, 5]] {
            let seqlens = Tensor::new(&offsets, &device)?;
            let pooled = pool_packed_splade_activations_dispatch(&activated, &seqlens, &[4])?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()?;
            assert!(pooled[0].iter().all(|value| value.is_nan()));
        }

        let decreasing = Tensor::new(&[0u32, 3, 2], &device)?;
        let pooled = pool_packed_splade_activations_dispatch(&activated, &decreasing, &[2, 2])?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        assert!(pooled[1].iter().all(|value| value.is_nan()));
        Ok(())
    }
}
