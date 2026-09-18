"""CPU detector with value-based n-gram counting and a stable normal tail.

Transformers 4.57.6 counts Tensor objects by identity, so its
ignore_repeated_ngrams=True option does not actually deduplicate equal n-grams.
This small adapter fixes counting without changing the watermark green lists.
Normal-tail p-values remain nominal, not an empirical calibration guarantee.
"""

from collections import Counter
from math import erfc, sqrt

import numpy as np
import torch
from transformers import WatermarkDetector as TransformersWatermarkDetector


class WatermarkDetector(TransformersWatermarkDetector):
    def _get_ngram_score(self, prefix, target):
        ids = torch.tensor(prefix, dtype=torch.long, device=self.processor.rng.device)
        return bool(target in self.processor._get_greenlist_ids(ids))

    def _score_ngrams_in_passage(self, input_ids):
        selfhash = self.processor.seeding_scheme == "selfhash"
        n = self.processor.context_width + 1 - int(selfhash)
        totals, greens = [], []
        for ids in input_ids.tolist():
            counts = Counter(tuple(ids[i:i + n]) for i in range(len(ids) - n + 1))
            total = green = 0
            for ngram, frequency in counts.items():
                weight = 1 if self.ignore_repeated_ngrams else frequency
                prefix = ngram if selfhash else ngram[:-1]
                total += weight
                green += weight * self._get_ngram_score_cached(prefix, ngram[-1])
            totals.append(total)
            greens.append(green)
        return np.asarray(totals, dtype=float), np.asarray(greens, dtype=float)

    def _compute_pval(self, x, loc=0, scale=1):
        z = (np.asarray(x) - loc) / scale
        return np.vectorize(lambda v: 0.5 * erfc(v / sqrt(2)), otypes=[float])(z)
