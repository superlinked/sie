# Can translation erase an AI text watermark?

*A sentence can keep its meaning while losing the statistical clues to where it came from. Here's how that happens, and what a small experiment can actually tell us.*

I was curious how watermarking of AI-generated text actually works. When I was a student, every thesis went through a plagiarism checker, so software judging where a text came from is familiar to me. But a plagiarism checker compares your text against a library of existing texts. A watermark is planted inside the text itself, at the moment of generation, and only someone with the right key can test for it. I wanted to see how a lab would implement that, so I built a small version myself.

Everything below rests on one separation: what a sentence says, and the particular words chosen to say it. This kind of watermark lives entirely in the second. So any transformation that keeps what a text says while re-choosing its words is a potential eraser. That is the idea we are going to test.

## A watermark is a small bias, repeated

Imagine a model finishing this sentence:

> The garden was unusually ___ for October.

"Warm," "quiet," and "green" might all be plausible. A watermark gently favors some of the options. One nudged choice tells you nothing; a few hundred add up to something measurable.

In the green-list scheme (from Kirchenbauer and colleagues' [*A Watermark for Large Language Models*](https://proceedings.mlr.press/v202/kirchenbauer23a.html)), a key and the preceding token decide which vocabulary entries get a small probability boost. The boosted entries are the "green list," the rest are "red," and the colors reshuffle at every position. Anyone holding the same key and tokenizer can replay the splits and count how often the text landed green.

Two clarifications before we measure anything. First, this is not what an ordinary AI-writing classifier does: a classifier hunts learned patterns of machine *style*, while a watermark detector checks for a deliberately planted signal using a key. (I ran this article and its Arabic round trip through a commercial checker; both scored "100% AI." Style survives translation completely. The keyed signal is what may not.) Second, designs differ: Google's [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) uses tournament sampling and [runs in Gemini](https://deepmind.google/models/synthid/). My experiment uses the simpler green-list scheme, under my own key, and measures nothing about any commercial assistant.

## Counting the signal

Say the green list covers a quarter of the vocabulary. Unwatermarked text lands green about a quarter of the time by chance, so detection is just counting the excess. With `T` scored choices, `G` green hits, and green fraction `γ`:

```text
z = (G − γT) / √[Tγ(1 − γ)]
```

With `T = 200`, `γ = 0.25`, and `G = 70`: we expected 50 green hits, saw 20 extra, and get `z ≈ 3.27`. In plain Python:

```python
from math import erfc, sqrt

T, G, gamma = 200, 70, 0.25
z = (G - gamma * T) / sqrt(T * gamma * (1 - gamma))
p_normal = 0.5 * erfc(z / sqrt(2))
print(f"z={z:.2f}, nominal p={p_normal:.5f}")
# z=3.27, nominal p=0.00055
```

I flag scores strictly above 3, a nominal one-in-741 false-alarm rate under the normal approximation. Nominal is the operative word: short passages, repetition, the chosen key, and testing many texts all push on it, and a p-value is never the probability that a particular author used AI.

One counting rule matters throughout: with a one-token context, an identical token pair always gets the same color, so the detector counts each distinct pair once. `T` always means deduplicated pairs here, not word count.

## Why keeping the words is only part of the story

> The small boat crossed the quiet lake.
>
> A little boat crossed the quiet lake.

Most words survive this rewrite, but changing the token *before* another token changes the green list used to score it. What matters is the survival of *token-and-context pairs*, not the fraction of words copied.

That intuition can be made exact. Let `T₀` and `T₁` be the scored pairs before and after rewriting, `R` the pairs that survive verbatim, and `q_ret`, `q_new` the green-hit rates among retained and new pairs. Then:

```text
z_after = [R(q_ret − γ) + (T₁ − R)(q_new − γ)]
          / √[T₁γ(1 − γ)]
```

This just splits the excess green hits into two piles. If retained pairs keep their original green rate and new pairs land at baseline, then with `a = R/T₁`:

```text
z_after ≈ a × √(T₁/T₀) × z_before
```

and simply `z_after ≈ a × z_before` at similar lengths. Two consequences are worth internalizing. Dilution scales the score, so the same rewrite leaves a strong watermark flagged (10 becomes 5) and pushes a weak one under the bar (5 becomes 2.5). And shortening alone weakens evidence: keep a quarter of the pairs unchanged and the expected score halves, even though every surviving word is identical. There is no universal "translation multiplier"; rewriting can also preserve favorite phrases, mint new green pairs, or restructure the text.

## Why translation, of all things

The robustness literature keeps circling two stress tests: rewriting and translation. Google's SynthID documentation says detection confidence can drop sharply when a text is thoroughly rewritten or translated, and a 2024 ACL paper asked directly, *Can Watermarks Survive Translation?*

The intuition: translate an AI paragraph into Arabic and back. The wording changes, the argument survives. Does the watermark? The answer turns out to be *sometimes*, and the rest of the article is about where that "sometimes" comes from.

## What the saved experiment shows

Main set: eight watermarked outputs from `Qwen/Qwen2.5-1.5B-Instruct` and eight unwatermarked controls from the same prompts (200-token cap; green fraction 0.25, bias 4, one-token context). The Arabic round trips and paraphrases ran through the SIE API with `Qwen/Qwen3.6-27B` (translation at temperature 0, paraphrase at temperature 1). Scores were recomputed from the saved texts after fixing the detector's repeated-pair counting.

| Text | Passages | Mean z | Above z = 3 |
|---|---:|---:|---:|
| Original watermarked output | 8 | 10.86 | 8/8 |
| English → Arabic → English | 8 | 5.91 | 8/8 |
| One paraphrase pass | 8 | 0.69 | 0/8 |
| Two paraphrase passes | 8 | 1.10 | 0/8 |
| Unwatermarked model output | 8 | −0.09 | 0/8 |

The mechanism behind the table: round trips kept about 64% of their distinct token pairs from the original; one-pass paraphrases kept about 8%. (Pair overlap only, not a percentage of meaning or of watermark information.) The round trip left a substantial signal because it returned much of the original wording; the paraphraser replaced the wording that carried it. Note the second paraphrase pass did not lower the average further: rewriting is not a dial where every turn reduces the score. And note the scope: eight short passages under one setting, with model-generated controls, cannot establish rare false-positive rates or prove that paraphrasing always works.

To isolate the starting-strength effect, a second experiment used 21 new prompts, three keys, and four settings (no watermark, then biases 2, 3, 4). All 84 sources went through an English → Arabic → English round trip with the local MADLAD-400-3B translator, giving 168 scored records with every original and intermediate text saved. The prompts are deliberately ordinary synthetic writing tasks (libraries, trees, backups, fictional cafes), and one key covers each group of seven prompts, so key and prompt effects are not independently separated.

| Watermark bias | Mean z before | Mean z after | Detected before | Detected after |
|---|---:|---:|---:|---:|
| 0: unwatermarked control | −0.13 | −0.11 | 0/21 | 0/21 |
| 2 | 4.48 | 2.48 | 19/21 | 8/21 |
| 3 | 7.11 | 3.56 | 21/21 | 15/21 |
| 4 | 9.23 | 4.07 | 21/21 | 15/21 |

This is the accounting identity playing out. At bias 2 the round trip pushed most passages under the threshold; at higher settings most stayed detectable. Going from bias 3 to 4 did not raise the post-translation count in this sample, and none of these settings is established as typical of commercial deployments. The table includes every pair; excluding the three pairs flagged by completeness and length checks leaves 8/21, 14/20, and 14/19, so the pattern survives, without proving that every detail of meaning did.

Read the evidence itself, not just the averages: `records.csv` in the companion dataset pairs each source with its returned English by `id`, and `originals.json`/`translations.json` keep every intermediate step. One library paragraph turns "accessible and affordable" into "easier and more sustainable," a shift in the claim even though the topic stays close.

## A translation can preserve meaning and still lose the mark

A translator carries a message into another language; it has no reason to preserve a keyed preference among English token pairs. This is the *semantic bottleneck*: many wordings carry the same message, so a transformation that keeps the message is free to discard the original wording, and the watermark's pattern is part of that wording. It needs no randomness to do so: even a deterministic rewrite maps many sentences onto one standard phrasing. The converse also holds, and the first table shows it: a rewrite can drag enough original wording along to leave evidence behind.

The research record matches this mixed picture. [He and colleagues](https://arxiv.org/abs/2402.14007) found serious cross-lingual weaknesses in the schemes they tested and proposed X-SIR as a more consistent design, so different tokens do not make detection impossible by definition. [DIPPER](https://arxiv.org/abs/2303.13408) showed large detector drops under paraphrase, while [Kirchenbauer's reliability study](https://arxiv.org/abs/2306.04634) found residual watermarks detectable after human paraphrase given roughly 800 observed tokens at a 10⁻⁵ false-positive rate. Both can be true: a weak residual signal is inconclusive in one paragraph and decisive across a larger sample. The detector side moves too: a [2026 revision by Mohamed and Gubri](https://arxiv.org/abs/2510.18019) searches over back-translations *inside* the detector to recover a weakened signal.

## "Meaning preserved" needs its own check

Embedding cosine similarity averaged about 0.95 for the Arabic round trips and 0.84 for one paraphrase pass. Do not read those as "95% and 84% of the meaning preserved": cosine measures how close two model representations sit, not factual accuracy, and not a percentage of anything.

The saved texts show why the distinction matters. One passage's dubious claim that lower noise improves air quality is faithfully preserved by translation (repeating an error does not make it true); elsewhere a translation turns bus services into "these centres." And length can masquerade as success: in an early seven-language sweep, one German round trip collapsed a 200-token passage to a 14-token sentence, so its low score reflects lost material, not erased watermark. A stronger experiment would check omissions, names, numbers, and causal claims alongside the similarity metric, and compare transformations at matched lengths.

## The deeper connection: information and room to choose

A watermark can only operate where several continuations are plausible; where the model is forced into one token, a small bias changes nothing. The [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4) discusses both length and next-token uncertainty as detectability factors. The precise vocabulary for this is *KL divergence*, the gap between the watermarked and unwatermarked distributions; Leon Chlon's [watermark toolkit](https://github.com/leochlon/watermark-edfl-toolkit) prompted me to look at the experiment through it.

For a green-list watermark the one-step calculation is compact. Let `q` be the probability mass on the green list before the boost (not the same as `γ`: a quarter of the vocabulary need not hold a quarter of the probability). Adding bias `δ` to the green logits gives:

```text
q_watermarked = e^δ q / (1 − q + e^δ q)
```

so the odds of green are multiplied by `e^δ` (with `q = 0.25`, `δ = ln 2` lifts the green probability to 0.40). The distributional change at that step, in nats:

```text
KL = δ × q_watermarked − ln(1 − q + e^δ q)
```

It approaches zero when all the probability is already green, or none is: a color preference only matters where there is mass to move. Summing these along a passage is tempting, but the sum is not a detector score, and the exact sequence-level quantity averages over the prefixes the watermarked model could generate:

```text
KL(Pw(sequence) || P(sequence))
  = E under Pw [sum over steps of KL(Pw(next | prefix) || P(next | prefix))]
```

Here `P` is the same generator without its watermark, not "all human writing." This distinction cost this article a rewrite: an earlier draft claimed that 87% of the watermark information sat at uncertain positions and that retained information predicted detection at correlation 0.90. Its reconstruction omitted the prompt and sampling settings, and its retention measure was a proxy, so I withdrew both numbers. What survives is the clean part: by the data-processing inequality, applying the same transformation to watermarked and unwatermarked text cannot increase their distributional KL. That *permits* a rewrite to discard evidence; it does not say how much a particular translation discards, nor force any individual z-score down.

## What a low score actually tells you

A score below the threshold means this detector did not find enough evidence to flag this text under this key. It does not prove the text is human, that every trace is gone, or that the wording is unlike a provider's earlier output; unwatermarked text lands at mildly positive scores by chance all the time. Residual evidence can still accumulate across independent passages if repeated material and multiple testing are handled properly, and retrieval against a provider's stored outputs ([studied in the DIPPER paper](https://arxiv.org/abs/2303.13408)) is a separate defense with its own calibration needs.

The overall picture: watermarking hands a provider a specific statistical signal, ordinary copying preserves it perfectly, and translation and rewriting mark its boundary. Provenance stored in word choices lasts exactly as long as the word choices do, while readers keep recognizing the idea long after the evidence about its wording is gone.

## Running it yourself

The saved data needs no key, weights, or Python: open `fresh-translation-2026-09-13/records.csv` in a spreadsheet. All code and data live in the companion `watermark-robustness` folder, and its README covers every path below in detail.

For a live transformation, get an API key from the [Superlinked console](https://console.superlinked.com) ([inference-grant application](https://superlinked.com/) if hosted access is not enabled), then:

```bash
uv sync --locked
uv run --frozen python run_paraphrase.py --ask-key --limit 1 \
  --arms rt_ar paraphrase_1x --output runs/my-sie-demo.json
```

Three hosted calls: one passage translated to Arabic and back, and paraphrased once, then scored locally with the original watermark key. The SIE key authorizes requests; the numeric watermark key defines the pattern. Never paste a real key into a published example. The scripts use `SIEClient.chat_completions` against `https://api.superlinked.com` ([docs](https://superlinked.com/docs/generate)); the saved comparison used `Qwen/Qwen3.6-27B`, the demo now defaults to `Qwen/Qwen3.8-27B-FP8` (`SIE_GENERATOR_MODEL` to change), and a new model is a new comparison, not a rerun of the first table. Results go to a new file, never into the published tables.

Two hard-won practical notes. The green-list random generator is device-dependent (my first detector run scored everything near zero because generation ran on GPU and detection on CPU), so both run on CPU here. And the strength sweep is a separate local MADLAD experiment needing `ctranslate2==4.8.1` and no SIE key; the README has the commands. The companion folder also holds this very article after its own Arabic round trip: the argument survives, the phrasing drifts on nearly every line.

---

*Experiment note, September 2026: These are synthetic test passages, including fictional topics and some incomplete or factually unreliable output. All watermark tests concern locally generated text under known experimental keys. The first table reports offline re-scoring of archived transformations. The second reports the fresh 21-prompt local translation experiment. The companion files preserve the new texts, settings, completeness flags, all 168 text-and-score records, and the sensitivity check. Neither experiment measures deployed commercial assistants.*

### Further reading

- Kirchenbauer et al., [*A Watermark for Large Language Models*](https://proceedings.mlr.press/v202/kirchenbauer23a.html), ICML 2023. The green-list scheme.
- Kirchenbauer et al., [*On the Reliability of Watermarks for Large Language Models*](https://arxiv.org/abs/2306.04634), ICLR 2024. Residual detection after rewriting.
- Dathathri et al., [*Scalable watermarking for identifying large language model outputs*](https://www.nature.com/articles/s41586-024-08025-4), Nature 634, 818–823, 2024. SynthID-Text.
- He et al., [*Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models*](https://arxiv.org/abs/2402.14007), ACL 2024. Translation attacks and X-SIR.
- Krishna et al., [*Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense*](https://arxiv.org/abs/2303.13408), NeurIPS 2023. DIPPER and retrieval.
- Mohamed and Gubri, [*Is Multilingual LLM Watermarking Truly Multilingual? Scaling Robustness to 100+ Languages via Back-Translation*](https://arxiv.org/abs/2510.18019), preprint, revised March 2026. Translation as part of detection.
