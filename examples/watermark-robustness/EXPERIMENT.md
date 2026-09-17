# Can translation erase an AI text watermark?

*A sentence can keep its meaning while losing the statistical clues to where it came from. Here's how that happens, and what a small experiment can actually tell us.*

I was curious how watermarking of AI-generated text actually works. There is plenty of talk about it, but I could not have explained what happens between a model picking a word and a detector later deciding that word looks marked, so I went and traced it.

Part of the curiosity is old. When I was a student, every thesis went through a plagiarism checker before submission, so software passing judgment on where a text came from is familiar territory for me. But a plagiarism checker compares your text against a library of existing texts. A watermark promises something different: the text itself carries a signature from the model that generated it, planted at the moment of generation, and you need the right key to see it. I wanted to see how a lab would implement that in newly generated text, so I built a small version myself.

The whole trick rests on a separation we never make while reading: what a sentence says, and the particular words chosen to say it. This kind of watermark lives entirely in the second. That has a consequence worth testing: any transformation that keeps what a text says while re-choosing its words is a potential eraser.

So I tested it, with a small model, a watermark key I controlled, and a handful of generated passages. The experiment shows the mechanism nicely. It also caught me making a familiar mistake, the one where a pattern in a few examples quietly hardens into a rule about all watermarks.

## A watermark is a small bias, repeated

Imagine a model finishing this sentence:

> The garden was unusually ___ for October.

"Warm," "quiet," and "green" might all be plausible, depending on what came before. A watermark gently favors some of the available options. One nudged choice tells you nothing. A few hundred of them add up to something you can measure.

In the green-list scheme I used, a key and the preceding token decide which vocabulary entries get a small probability boost. (A *token* is the model's unit of text: sometimes a word, sometimes a fragment or punctuation.) The boosted entries form the green list, everything else is red, and the colors mean nothing. They're just labels. The scheme comes from Kirchenbauer and colleagues' [*A Watermark for Large Language Models*](https://proceedings.mlr.press/v202/kirchenbauer23a.html).

The list reshuffles with context. "Quiet" might be green after one token and red after another. Anyone holding the same key and tokenizer can replay those splits and count how often the text landed green.

Worth pausing on what this is not. An ordinary AI-writing classifier hunts for learned patterns of machine style. A watermark detector checks for a deliberately planted signal, and it needs the key to do it. A familiar style doesn't establish who wrote a passage, and neither does a failed watermark test. I got a live demonstration of the difference while drafting this: I ran the article and its Arabic round trip through a commercial AI-writing checker, and both came back "100% AI." The style signal such tools read survives translation completely. The keyed signal this article studies is precisely the one that may not.

Designs differ, too. Google's [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4) runs a tournament between candidate tokens instead of boosting a list, and Google [documents its use in Gemini text](https://deepmind.google/models/synthid/). Everything below uses the simpler green-list watermark, on my machine, under my key. None of it measures Gemini or any other commercial assistant.

## Counting the signal

Say the green list covers a quarter of the vocabulary. Unwatermarked text should land green about a quarter of the time, purely by chance.

Call the number of scored choices `T`, the number of green hits `G`, and the green-list fraction `γ` (gamma). The usual score is:

```text
z = (G − γT) / √[Tγ(1 − γ)]
```

The numerator asks how many extra green hits showed up. The denominator puts that excess on the scale of ordinary random variation.

With `T = 200`, `γ = 0.25`, and `G = 70`: we expected 50 green hits and saw 20 extra, which gives `z ≈ 3.27`.

Here's the same calculation in plain Python. It starts from counts; it doesn't tokenize text or reconstruct a green list.

```python
from math import erfc, sqrt

T, G, gamma = 200, 70, 0.25
z = (G - gamma * T) / sqrt(T * gamma * (1 - gamma))
p_normal = 0.5 * erfc(z / sqrt(2))
print(f"z={z:.2f}, nominal p={p_normal:.5f}")
# z=3.27, nominal p=0.00055
```

I flag scores strictly above 3. Under a standard-normal approximation the one-sided tail at 3 is about 0.00135, roughly one in 741. I want to be careful with that number. It's a *nominal* false-alarm probability under the model's assumptions, not a measured guarantee for arbitrary writing; short passages, repetition, the chosen key, and testing many texts all push on it. And a p-value is not the probability that a particular author used AI.

One counting detail bit me early. With a one-token context, an identical token pair always gets the same color, so repeating it isn't a fresh random trial. The detector counts each distinct pair once. Throughout this article, `T` means the choices actually scored after that deduplication, not the word count.

## Why keeping the words is only part of the story

Consider a made-up pair of sentences:

> The small boat crossed the quiet lake.
>
> A little boat crossed the quiet lake.

Most of the wording survives. But changing the token before another token can also change the green list used to score it. What matters for this watermark is the survival of *token-and-context pairs*, not simply the fraction of words copied.

There's a useful accounting identity here. Let:

- `T₀` be the original number of scored pairs and `T₁` the number after rewriting.
- `R` be the number of scored pairs in the output that also occur in the original.
- `q_ret` and `q_new` be the green-hit fractions among retained and new pairs.

Then the output's score is exactly:

```text
z_after = [R(q_ret − γ) + (T₁ − R)(q_new − γ)]
          / √[T₁γ(1 − γ)]
```

This just splits the excess green hits into two piles.

Now assume the retained pairs keep roughly their original green-hit rate and the new pairs land at roughly the baseline `γ`. With `a = R/T₁`:

```text
z_after ≈ a × √(T₁/T₀) × z_before
```

And when the lengths are similar, simply `z_after ≈ a × z_before`. A handy intuition, with conditions attached. Rewriting can preferentially keep particular phrases, mint new green pairs by accident, shorten the passage, or change its structure. There is no universal "translation multiplier."

Two consequences fall out. With equal lengths and `a = 0.5`, a score of 10 becomes about 5 while a score of 5 becomes about 2.5, so the same dilution leaves one passage flagged and pushes another under the bar. And length cuts on its own: keep only a quarter of the original's distinct pairs without adding any, and `a = 1` while `T₁/T₀ = 1/4`, so the expected score halves even though every surviving word is untouched.

## Why translation, of all things

The robustness research keeps circling two stress tests: rewriting and translation. Google's SynthID documentation says detection confidence can drop sharply when a text is thoroughly rewritten or translated into another language. A 2024 ACL paper put the question in its title, *Can Watermarks Survive Translation?*, and we'll come back to what it found.

The intuition is easy to state. Take an AI-generated paragraph, translate it into Arabic, then translate it back into English. The wording changes. The argument mostly survives. Does the watermark survive too?

Sometimes. Less satisfying than a clean yes or no, and much more interesting. The rest of this article is about where that "sometimes" comes from.

## What the saved experiment shows

The main set contains eight watermarked outputs from `Qwen/Qwen2.5-1.5B-Instruct`, plus eight unwatermarked outputs from the same prompts. Generation was capped at 200 new tokens. The watermark used a green-list fraction of 0.25, bias 4, and a one-token context.

The comparison below uses the saved Arabic round trips and paraphrases produced through the SIE API with `Qwen/Qwen3.6-27B`. Translation ran at temperature 0, paraphrasing at temperature 1. The scores were recomputed from the saved text after correcting the detector's repeated-pair counting.

| Text | Passages | Mean z | Above z = 3 |
|---|---:|---:|---:|
| Original watermarked output | 8 | 10.86 | 8/8 |
| English → Arabic → English | 8 | 5.91 | 8/8 |
| One paraphrase pass | 8 | 0.69 | 0/8 |
| Two paraphrase passes | 8 | 1.10 | 0/8 |
| Unwatermarked model output | 8 | −0.09 | 0/8 |

The round trips kept about 64% of their output's distinct token pairs from the original. One-pass paraphrases kept about 8%. Those percentages describe exact pair overlap, nothing more; they are not percentages of meaning, words, or watermark information preserved.

Within these examples the pattern is stark. The Arabic round trip left a substantial signal. The paraphraser replaced the wording that carried it. And a second paraphrase pass did not push the average lower; rewriting isn't a dial where every turn reduces the score.

Eight passages is eight passages, though. Eight non-detections don't demonstrate that paraphrasing always works, and eight clean controls can't establish a rare false-positive rate, especially since the controls are model-generated rather than human-written.

To see how the starting strength changes the outcome, I ran a separate experiment: 21 new prompts, three watermark keys, four settings (no watermark, then biases of 2, 3, and 4). All 84 source passages went through an English → Arabic → English round trip using the local MADLAD-400-3B translation model. That gives 168 scored records, with every original and intermediate text saved. Generation was capped at 224 tokens.

What's actually in that dataset? Short synthetic paragraphs. Explanations about libraries, trees, backups and bread; fictional scenes in cafes and seaside towns; arguments about buses or cooking lessons. Each prompt asks for 70 to 90 words, which the model doesn't always respect. Deliberately ordinary writing tasks, not a sample of published human articles or a standard multilingual benchmark.

The arithmetic is `21 prompts × 4 settings × 2 text versions = 168 records`. Each source has a matching returned-English version. The Arabic intermediate is saved too, but isn't another row in the score table. One watermark key covers each group of seven prompts, and that assignment stays fixed across settings, which helps compare settings within a prompt but doesn't separate key effects from prompt effects.

| Watermark bias | Mean z before | Mean z after | Detected before | Detected after |
|---|---:|---:|---:|---:|
| 0: unwatermarked control | −0.13 | −0.11 | 0/21 | 0/21 |
| 2 | 4.48 | 2.48 | 19/21 | 8/21 |
| 3 | 7.11 | 3.56 | 21/21 | 15/21 |
| 4 | 9.23 | 4.07 | 21/21 | 15/21 |

Here is the starting-strength effect head-on. At bias 2, translation pushed most passages under the threshold. At the higher settings, most stayed detectable. Notice that going from bias 3 to 4 didn't increase the number flagged after translation in this sample; a higher average is not a guarantee for every passage. And none of these settings is established as typical of commercial deployments.

The table includes every pair. Two source passages hit the generation cap, two pairs hit a translation cap, and those flags overlap once. One translation broke down into repetitive text, which is a failure of translation, not evidence of a faithful rewrite removing a mark. Excluding the three flagged pairs leaves post-translation detections of 8/21, 14/20, and 14/19 for biases 2, 3, and 4. The pattern survives the check. Passing it still doesn't prove every detail of the meaning survived.

Also worth keeping in mind: these are 21 prompts reused across settings, not 168 independent prompts. And this fresh run uses a different translator from the earlier cloud comparison, so the two tables answer separate questions: how those particular paraphrases compared with round trips, and how initial watermark strength played out in a local translation pipeline.

You can inspect the evidence rather than trust the averages. In the companion dataset, `records.csv` holds each text with its prompt ID, watermark setting, green-hit count, scored-pair count and z-score; match the same `id` across the two `arm` values to read a before-and-after pair. `originals.json` and `translations.json` preserve the source, the Arabic step and the returned English, and `results.json` adds length and completeness checks. One library paragraph turns "accessible and affordable" into "easier and more sustainable," which is a shift in the claim even though the topic stays close. Reading the text is part of the experiment, not an optional extra.

## A translation can preserve meaning and still lose the mark

A translator is trying to carry a message into another language. It has no reason to preserve a keyed preference among English token pairs. On the way back, some familiar phrases return. Others don't.

This is the *semantic bottleneck*: many different wordings can carry much the same message. A transformation that keeps the message is free to discard details of the original wording, and the watermark's statistical pattern is exactly such a detail.

It doesn't even need randomness to do this. A deterministic rewrite can map several different sentences onto one standard phrasing, destroying the information about which wording was chosen originally. And the converse holds too: a sampled rewrite can drag enough of the original wording along to leave evidence behind.

One-directional translation and there-and-back translation are different tests. In [*Can Watermarks Survive Translation?*](https://arxiv.org/abs/2402.14007), He and colleagues found serious cross-lingual weaknesses in the methods they tested, and proposed X-SIR to improve cross-lingual consistency. So different tokens don't make detection impossible by definition. The watermark's design matters.

Paraphrasing has a similarly mixed record. [Krishna and colleagues' DIPPER study](https://arxiv.org/abs/2303.13408) demonstrated substantial drops in detector performance. But [Kirchenbauer and colleagues' reliability study](https://arxiv.org/abs/2306.04634) found residual watermarks that stayed detectable after human and machine paraphrasing; in their human-paraphrase experiment, detection took roughly 800 observed tokens on average at a reported false-positive rate of `10⁻⁵`. The findings coexist because the transformations, lengths, and evaluation conditions differ. A weak residual signal can be inconclusive in one paragraph and damning across a much larger sample.

And the story keeps moving. A [2026 revision of work by Mohamed and Gubri](https://arxiv.org/abs/2510.18019) puts back-translation inside the *detector*, searching for translations that recover a weakened signal. A useful reminder that the detector side can change too.

## "Meaning preserved" needs its own check

The eight-passage comparison used an embedding model to compare original and transformed text. Average cosine similarity came out around 0.95 for the Arabic round trip and 0.84 for one paraphrase pass.

It would be tempting to call those "95% of the meaning preserved" and "84% preserved." That would be wrong. Cosine similarity measures how close two model representations sit. It is not a factual accuracy score, and it is not a percentage of anything.

The saved examples show why the distinction earns its keep. One passage about night buses claims that lower noise improves air quality; the translation faithfully preserves that dubious causal claim, and faithfully repeating an error doesn't make it true. Elsewhere a local translation turns a reference to bus services into "these centres." The topic stays recognizable while a detail quietly walks away.

The seven-language sweep I ran early on had a worse version of this problem. Some outputs came back much shorter; one German round trip reduced a 200-token passage to a 14-token sentence. A low detection score there reflects lost material as much as changed wording, which makes that sweep a poor basis for ranking languages by how well they erase a watermark.

If I were doing the stronger version of this experiment, I'd check omissions, names, numbers, causal claims, and incomplete sentences alongside the similarity metric, compare transformations at similar output lengths, and use more prompts, keys, and decoding seeds. Otherwise a method that deletes half the argument looks deceptively successful.

## The deeper connection: information and room to choose

The watermark has room to operate only where several continuations are plausible. If the model is practically forced into one token, a small bias changes nothing. A stronger bias can force a different choice, but then keeping the answer's quality gets harder. The [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4) discusses both text length and next-token uncertainty as factors in detectability.

The precise way to talk about this is *KL divergence*: how far the watermarked probability distribution sits from the unwatermarked one. It's a standard information-theory quantity, and Leon Chlon's [watermark toolkit](https://github.com/leochlon/watermark-edfl-toolkit) is what prompted me to look at the experiment through it.

For a green-list watermark the calculation is compact. Let `q` be the probability mass on the green list before the boost. Note that's different from `γ`, the fraction of vocabulary entries on the list; a quarter of the vocabulary needn't hold a quarter of the probability. If the watermark adds bias `δ` to the green logits (the scores used to produce probabilities), then:

```text
q_watermarked = e^δ q / (1 − q + e^δ q)
```

The odds of choosing green get multiplied by `e^δ`. As a toy example, `q = 0.25` with `δ = ln(2)` doubles those odds and lifts the green probability to 0.40.

The corresponding conditional KL is:

```text
KL = δ × q_watermarked − ln(1 − q + e^δ q)
```

That's the distributional change at one step, in nats. It approaches zero when all the probability already sits on green tokens, or none does. A color preference only changes outcomes where there's probability mass to move between the groups.

You can sum these conditional KL values along a generated passage. But the sum is not automatically a detector score, and it certifies nothing about survival after translation. The exact sequence-level KL averages over the prefixes the watermarked model could generate:

```text
KL(Pw(sequence) || P(sequence))
  = E under Pw [sum over steps of KL(Pw(next | prefix) || P(next | prefix))]
```

Here `P` means the same generator without its watermark, not all human writing. Matching a few surviving bigrams is not a measurement of how much of this distributional information survives a transformation.

That distinction cost this article a rewrite. An earlier draft claimed 87% of the watermark information sat at uncertain positions and that "retained information" predicted detection with a correlation of 0.90. The probability reconstruction behind those numbers omitted the prompt and some sampling settings, and the retention measure was a weighted overlap proxy. I withdrew both claims. The revised code makes its assumptions explicit, which is what the first version should have done.

What survives is the sound part: applying the same transformation to watermarked and unwatermarked text cannot increase their full distributional KL. That's the data-processing inequality. It permits a rewrite to discard evidence. It does not say how much evidence a particular translation discards, and it doesn't force any individual z-score to go down.

## What a low score actually tells you

A score below the threshold means this detector did not find enough evidence to flag this text under this key. That is the conclusion it supports, and no more.

It does not prove the text was human-written. It doesn't prove every trace of the watermark is gone, or that the wording is unlike a provider's earlier output. And a moderately positive score below the threshold is not some separate category of "clearly nonhuman" writing; unwatermarked text lands there by chance all the time.

Residual evidence can also accumulate across independent passages, provided the detector handles repeated material and multiple testing properly. And there's a second source of evidence entirely: searching a provider's stored outputs. [The DIPPER paper](https://arxiv.org/abs/2303.13408) studies retrieval as a defense against paraphrasing. Its usefulness depends on having the records, choosing a sensible matching method, and calibrating false matches, so it's a tool rather than a guarantee.

Where does that leave watermarking? Useful, once you know its operating conditions. It hands a provider a specific statistical signal to test, and ordinary copying preserves that signal perfectly. Translation and rewriting mark the boundary: provenance stored in word choices lasts exactly as long as the word choices do, while readers go on recognizing the idea long after the evidence about its original wording has gone.

## Running it yourself

Start by opening the saved data. You don't need a key, model weights or Python to read `fresh-translation-2026-09-13/records.csv` in a spreadsheet. All code and saved data live in the companion `watermark-robustness` folder.

To try a new hosted transformation, sign in to the [Superlinked console](https://console.superlinked.com) and obtain an API key for your account. If hosted access is not enabled, [Superlinked's website](https://superlinked.com/) offers an inference-grant application. Check your account's access and available credits before running requests.

From the companion folder, with Python 3.12 and [`uv`](https://docs.astral.sh/uv/) installed:

```bash
uv sync --locked
uv run --frozen python run_paraphrase.py --ask-key --limit 1 \
  --arms rt_ar paraphrase_1x --output runs/my-sie-demo.json
```

The script asks for the key without displaying or saving it. Alternatively, supply `SIE_API_KEY` through your environment or secret manager. Never paste a real key into the published example.

This small demo takes the first saved watermarked paragraph, translates it into Arabic and back, and separately paraphrases the original once: three hosted generation calls in total. It scores both returned texts locally, using the original watermark key and tokenizer. The SIE API key authorizes requests; the numeric watermark key defines the statistical pattern. They are different keys with different jobs.

The endpoint is `https://api.superlinked.com`, configurable with `SIE_BASE_URL` or `--sie-url`. The scripts use `SIEClient.chat_completions`, the SDK method for SIE's [chat-completions endpoint](https://superlinked.com/docs/generate). The saved cloud comparison used `Qwen/Qwen3.6-27B`; the updated demo defaults to `Qwen/Qwen3.8-27B-FP8`, selectable with `SIE_GENERATOR_MODEL`. Check the endpoint's catalog before running: a new model is a new comparison, not the experiment behind the first table. A missing model, insufficient balance, timeout or incomplete response stops the demo instead of silently retrying it. I learned to respect the device detail the hard way, too: my first detector run scored every watermarked passage near zero because generation had run on a GPU and detection on the CPU, and the green-list generator quietly disagrees between the two. Generation and detection both use CPU here for that reason.

Results go to a new file, not into either published table. A rerun can differ because sampling and the hosted model configuration can change. One passage is a wiring check, not a new robustness study. The README also explains how to generate new local source texts and how to re-score the saved experiments without SIE calls.

The second table is a separate, local MADLAD experiment and needs no SIE key. Re-running it requires downloading the model assets and installing `ctranslate2==4.8.1`; the exact commands and a separate output folder are in the README. And if you want to see the semantic bottleneck applied to something you've just read, the companion folder includes this very article after its own English → Arabic → English round trip. The argument survives. The phrasing drifts on nearly every line.

---

*Experiment note, September 2026: These are synthetic test passages, including fictional topics and some incomplete or factually unreliable output. All watermark tests concern locally generated text under known experimental keys. The first table reports offline re-scoring of archived transformations. The second reports the fresh 21-prompt local translation experiment. The companion files preserve the new texts, settings, completeness flags, all 168 text-and-score records, and the sensitivity check. Neither experiment measures deployed commercial assistants.*

### Further reading

- Kirchenbauer et al., [*A Watermark for Large Language Models*](https://proceedings.mlr.press/v202/kirchenbauer23a.html), ICML 2023. The green-list scheme.
- Kirchenbauer et al., [*On the Reliability of Watermarks for Large Language Models*](https://arxiv.org/abs/2306.04634), ICLR 2024. Residual detection after rewriting.
- Dathathri et al., [*Scalable watermarking for identifying large language model outputs*](https://www.nature.com/articles/s41586-024-08025-4), Nature 634, 818–823, 2024. SynthID-Text.
- He et al., [*Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models*](https://arxiv.org/abs/2402.14007), ACL 2024. Translation attacks and X-SIR.
- Krishna et al., [*Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense*](https://arxiv.org/abs/2303.13408), NeurIPS 2023. DIPPER and retrieval.
- Mohamed and Gubri, [*Is Multilingual LLM Watermarking Truly Multilingual? Scaling Robustness to 100+ Languages via Back-Translation*](https://arxiv.org/abs/2510.18019), preprint, revised March 2026. Translation as part of detection.
