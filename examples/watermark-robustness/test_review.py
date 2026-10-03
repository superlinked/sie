import math
import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel, PretrainedConfig, WatermarkingConfig
from transformers.generation.logits_process import (
    LogitsProcessorList, RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
    TopKLogitsWarper, TopPLogitsWarper, WatermarkLogitsProcessor,
)

from wm_detector import WatermarkDetector
from bgen_analysis import green_tilt_kl
from arms import restore


class DetectorRegressionTests(unittest.TestCase):
    def detector(self, unique=True):
        return WatermarkDetector(
            PretrainedConfig(vocab_size=100, bos_token_id=None), "cpu",
            WatermarkingConfig(greenlist_ratio=0.25, context_width=1),
            ignore_repeated_ngrams=unique,
        )

    def test_repeated_bigrams_are_counted_once(self):
        ids = torch.tensor([[1, 2, 1, 2, 1, 2]])
        unique = self.detector()(ids, return_dict=True)
        all_positions = self.detector(False)(ids, return_dict=True)
        self.assertEqual(unique.num_tokens_scored[0], 2)
        self.assertEqual(all_positions.num_tokens_scored[0], 5)

    def test_green_count_and_z_match_direct_enumeration(self):
        detector = self.detector()
        pairs = {(1, 2), (2, 1)}
        green = sum(b in detector.processor._get_greenlist_ids(torch.tensor([a])) for a, b in pairs)
        out = detector(torch.tensor([[1, 2, 1, 2]]), return_dict=True)
        self.assertEqual(out.num_green_tokens[0], green)
        self.assertAlmostEqual(out.z_score[0], (green - 0.5) / math.sqrt(0.375))

    def test_normal_tail_is_accurate_and_does_not_cancel(self):
        detector = self.detector()
        self.assertAlmostEqual(float(detector._compute_pval(3.0)), 0.0013498980316300957)
        self.assertGreater(float(detector._compute_pval(11.0)), 0)
        self.assertAlmostEqual(float(detector._compute_pval(0.0)), 0.5)

    def test_kl_agrees_with_direct_distribution_calculation(self):
        base = torch.tensor([0.1, 0.2, 0.7], dtype=torch.float64)
        log_weights = base.log()
        log_weights[:2] += 2.0
        marked = log_weights.softmax(0)
        direct = float((marked * (marked.log() - base.log())).sum())
        self.assertAlmostEqual(green_tilt_kl(0.3, 2.0), direct)
        self.assertEqual(green_tilt_kl(0.0, 2.0), 0)
        self.assertEqual(green_tilt_kl(1.0, 2.0), 0)
        self.assertEqual(green_tilt_kl(0.3, 0.0), 0)

    def test_placeholder_count_is_unique_and_unknown_ids_are_preserved(self):
        text, survived, total = restore("⟦0⟧ and ⟦0⟧, then ⟦99⟧", {0: "Ada", 1: "London"})
        self.assertEqual((survived, total), (1, 2))
        self.assertEqual(text, "Ada and Ada, then ⟦99⟧")

    def test_reconstructed_processor_order_matches_generate(self):
        torch.manual_seed(42)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=100, n_layer=1, n_head=2, n_embd=8, eos_token_id=99, pad_token_id=99,
        )).eval()
        prefix = torch.tensor([[1, 2, 1, 3]])
        config = WatermarkingConfig(greenlist_ratio=0.25, bias=2.0, context_width=1)
        processors = LogitsProcessorList([
            RepetitionPenaltyLogitsProcessor(1.1), TemperatureLogitsWarper(0.8),
            TopKLogitsWarper(20), TopPLogitsWarper(0.95),
            WatermarkLogitsProcessor(vocab_size=100, device="cpu", **config.to_dict()),
        ])
        with torch.inference_mode():
            expected = processors(prefix, model(prefix).logits[:, -1, :])
            generated = model.generate(
                prefix, max_new_tokens=1, do_sample=True, temperature=0.8,
                repetition_penalty=1.1, top_k=20, top_p=0.95,
                watermarking_config=config, output_scores=True, return_dict_in_generate=True,
                attention_mask=torch.ones_like(prefix),
            )
        torch.testing.assert_close(generated.scores[0], expected)


if __name__ == "__main__":
    unittest.main()
