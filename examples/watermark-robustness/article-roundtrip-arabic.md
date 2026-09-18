# The article, after its own experiment

*This file is the watermark article translated English to Arabic and back to
English with the same hosted pipeline the experiment used (`Qwen/Qwen3.8-27B-FP8`,
temperature 0, block by block). Code blocks, tables, and formulas passed through
untranslated. Nothing here was edited afterward; the drift you notice is the
point. The original is `watermarks-and-the-semantic-bottleneck.md`.*

---

# Can translation erase a textual watermark for artificial intelligence?

*Sentences can retain their meaning while losing the statistical cues that point to their source. Here is how that happens, and what a small experiment can actually tell us.*

I wanted to know how watermarking on AI-generated text actually works. Not the version announced in press releases, but the mechanism itself. Labs announce it, regulators demand it, and detection tools promise they can read it, so I noticed that I had never tracked how the mark is supposed to persist within ordinary prose.

Part of the curiosity is old. When I was a student, every thesis had to pass through a plagiarism detection tool before submission, so the idea of a program making judgments about the origin of text is familiar territory for me. But a plagiarism detector compares your text against a library of pre-existing texts. A watermark, on the other hand, makes stranger promises: the text itself carries a quiet signature from the model that generated it, embedded at the moment of generation and visible only when someone with the right key looks for it. I wanted to see how the lab would implement this in freshly generated text, so I built a small version myself.

The entire trick hinges on a distinction we never make while reading: what the sentence says, and the specific words chosen to say it. This kind of watermark lives entirely in the latter. And this has a clear consequence worth testing. Any transformation that preserves what the text says while reselecting its words is a potential eraser.

So I tested it, using a small model, a watermark key that I controlled, and several generated clips. The experiment clearly demonstrates the mechanism. It also revealed that I had made a familiar mistake: one where a pattern in a few examples silently becomes a universal rule about all watermarks.

## The watermark is a small, recurring bias

Imagine a model that completes this sentence:

> The garden was unusual ___ for October.

Both "warm" and "calm" and "green" may be reasonable options, depending on what precedes them. The watermark shows a slight bias toward some of the available choices. A single guided choice does not tell you anything. But hundreds of these choices accumulate to form something measurable.

In the green list diagram you used, the key and the preceding token determine which dictionary entries receive a small probability boost. (The *token* is the unit of text in the model: sometimes a word, sometimes a part or punctuation mark.) The boosted entries form the green list, everything else is red, and the colors mean nothing. They are just labels. The diagram comes from Kirchenbauer et al.'s work [*Watermarking for Large Language Models*](https://proceedings.mlr.press/v202/kirchenbauer23a.html).

The list is reordered based on context. "Calm" might be green after one symbol and red after another. Anyone who has the same key and symbol hash can rerun those splits and count how many times the text appeared in green.

It is worth pausing on what this is not. A standard AI-writing classifier looks for learned patterns of machine style. A watermark detector, by contrast, checks for an intentionally embedded signal and requires the key to do so. Neither a familiar style nor a failed watermark test proves the identity of the passage’s author. I got a live demonstration of this distinction while drafting this article: I ran the article and its Arabic translation back and forth through a commercial AI-detection tool, and both came back as “100% AI.” The stylistic signal such tools read remains perfectly intact after translation. The keyed signal examined in this article is precisely the one that may not survive.

The designs also differ. Google's [SynthID-Text](https://www.nature.com/articles/s41586-024-08025-4) paper works by running a tournament among candidate tokens rather than boosting a list, and Google documents its [use in Gemini text](https://deepmind.google/models/synthid/). Everything below uses the simplest green-list watermarking scheme, on my device, under my key. None of them measure Gemini or any other commercial assistant.

## Sign Count

Say that the green list covers a quarter of the vocabulary. The non-water text should appear green about once every four times, purely by chance.

Let us denote the number of rated options as `T`, the number of green hits as `G`, and the green list ratio as `γ` (gamma). The standard score is:

```text
z = (G − γT) / √[Tγ(1 − γ)]
```

The numerator asks about the number of additional green hits that appeared. The denominator places this excess on the scale of ordinary random variation.

At `T = 200`, `γ = 0.25`, and `G = 70`: we expected 50 green hits and observed 20 additional hits, yielding `z ≈ 3.27`.

Here is the same calculation in pure Python. It starts from the enumerations; it does not split the text into tokens nor reconstruct a green list.

```python
from math import erfc, sqrt

T, G, gamma = 200, 70, 0.25
z = (G - gamma * T) / sqrt(T * gamma * (1 - gamma))
p_normal = 0.5 * erfc(z / sqrt(2))
print(f"z={z:.2f}, nominal p={p_normal:.5f}")
# z=3.27, nominal p=0.00055
```

I teach on results that exceed 3 in strictness. Under the standard normal approximation, the one-sided tail at the value 3 is approximately 0.00135, which corresponds to roughly one in 741. I want to be careful about this number. It is a *nominal* probability of a false alarm under the model's assumptions, not a measured guarantee for random texts; short reports, repetition, the chosen key, and testing many texts all affect it. Moreover, a p-value is not the probability that a specific author used artificial intelligence.

I ran into a small counting issue early on. With a single-token context, identical token pairs always receive the same color, so repeating them does not constitute a new random trial. The detector counts each distinct pair only once. Throughout this article, the symbol `T` refers to the options actually evaluated after deduplication, not the number of words.

## Why Preserving Words Is Only Part of the Story

Consider a pair of artificial sentences:

> The small boat crossed the calm lake.
>
> A small boat crossed the calm lake.

Most of the phrasing remains unchanged. However, changing one token before another can also alter the green list used in its evaluation. What matters in this watermark is the preservation of *token-context pairs*, not simply the proportion of copied words.

There is a useful accounting identity. Let us assume:

- `T₀` is the original number of scored pairs, and `T₁` is the number after rewriting.
- `R` is the number of scored pairs in the outputs that also appear in the original.
- `q_ret` and `q_new` are the green hit rates among retained pairs and new pairs, respectively.

Then the output result is exactly:

```text
z_after = [R(q_ret − γ) + (T₁ − R)(q_new − γ)]
          / √[T₁γ(1 − γ)]
```

There is nothing deep yet. It just splits the extra green strikes into two comets.

Now assume that the retained couples maintain approximately the original green infection rate, and that the new couples settle at the baseline level `γ` approximately. With `a = R/T₁`:

```text
z_after ≈ a × √(T₁/T₀) × z_before
```

And when the lengths are similar, it suffices to use `z_after ≈ a × z_before`. A useful heuristic, but conditional. Rewriting may favor keeping certain phrases, accidentally create new green pairs, shorten the segment, or alter its structure. There is no universal “translation factor.”

The consequences involve two outcomes. When the lengths are equal and `a = 0.5`, a score of 10 becomes approximately 5, while a score of 5 becomes approximately 2.5; thus, the same attenuation causes one segment to be labeled while pushing another below the threshold. Length also has an inherent effect: retaining only a quarter of the original distinct pairs without adding anything yields `a = 1` while `T₁/T₀ = 1/4`, which halves the expected score even though every surviving word remains untouched.

## Why Translation, of All Things

Durability research centers on two key stress tests: rewriting and translation. Google’s SynthID documentation notes that detection confidence can drop sharply when text is fully rewritten or translated into another language. A 2024 ACL conference paper posed the question in its title, *Can Watermarks Survive Translation?*, which we will return to later.

The intuitive idea is easy to state. Take an AI-generated paragraph, translate it into Arabic, then back-translate it into English. The wording changes. The argument mostly remains. Does the watermark also remain?

Sometimes. Less satisfying than a clear "yes" or "no," and far more interesting. The rest of this article deals with the source of that word "sometimes."

## What the saved experiments show

The main set contains eight watermarked outputs from `Qwen/Qwen2.5-1.5B-Instruct`, in addition to eight unwatermarked outputs from the same prompts. Generation was capped at 200 new tokens. The watermark used a green list ratio of 0.25, a bias of 4, and a context of one token.

The comparison below uses saved Arabic round-trip journeys and rephrased phrases generated via the SIE API using `Qwen/Qwen3.6-27B`. Translation was performed at temperature 0, and rephrasing at temperature 1. Scores were recomputed from the stored text after correcting the detector’s duplicate-pair count.

| Text | Passages | Mean z | Above z = 3 |
|---|---:|---:|---:|
| Original watermarked output | 8 | 10.86 | 8/8 |
| English → Arabic → English | 8 | 5.91 | 8/8 |
| One paraphrase pass | 8 | 0.69 | 0/8 |
| Two paraphrase passes | 8 | 1.10 | 0/8 |
| Unwatermarked model output | 8 | −0.09 | 0/8 |

The circular trips retained approximately 64% of the token pairs in their outputs compared to the original. The single-path rephrasing retained approximately 8%. These percentages describe only the exact overlap of token pairs, and nothing else; they are not ratios of meaning, words, or preserved watermark information.

In these examples, the pattern is perfectly clear. The circular trip in Arabic left a large signal. The rewriter replaced the phrasing it carried. And a second rewriting pass did not lower the average; rewriting is not a disk where each rotation reduces the degree.

But eight segments remain eight segments. Eight non-detections do not prove that rephrasing always works, and eight clean controls cannot determine a rare false-positive rate, especially since the controls are model-generated rather than human-written.

To see how the initial power changes the outcome, I conducted a separate experiment: 21 new prompts, three watermark keys, and four settings (no watermark, then biases with values of 2, 3, and 4). All 84 source segments underwent an English-to-Arabic-to-English round trip using the local translation model MADLAD-400-3B. This gives us 168 evaluated records, with each original text and intermediate text preserved. Text generation was capped at a maximum of 224 tokens.

What is actually in that dataset? Short synthetic paragraphs. Explanations about libraries, trees, backups, and bread; fictional scenes in cafés and coastal cities; debates about buses or cooking lessons. Each prompt requests 70 to 90 words, a constraint the model does not always respect. Deliberately ordinary writing tasks, not a sample of published human articles or a standard multilingual benchmark.

The arithmetic is `21 prompts × 4 settings × 2 text versions = 168 records`. Each source has a matching, re-returned English version. The intermediate Arabic version is also saved, but it does not constitute an additional row in the scores table. A single watermark key covers each group of seven prompts, and this assignment remains constant across settings, which helps compare settings within a single prompt but does not separate the effects of the key from those of the prompt.

| Watermark bias | Mean z before | Mean z after | Detected before | Detected after |
|---|---:|---:|---:|---:|
| 0: unwatermarked control | −0.13 | −0.11 | 0/21 | 0/21 |
| 2 | 4.48 | 2.48 | 19/21 | 8/21 |
| 3 | 7.11 | 3.56 | 21/21 | 15/21 |
| 4 | 9.23 | 4.07 | 21/21 | 15/21 |

Here is the direct effect of the initial force. At bias 2, translation pushed most segments below the threshold. At higher settings, most remained detectable. Note that moving from bias 3 to 4 did not increase the number of labeled items after translation in this sample; a higher mean is no guarantee for every segment. Also, none of these settings are established as standard for commercial publication.

The table includes every pair. Two source segments reached the generation threshold, and two pairs reached the translation threshold; these markers overlapped once. One translation collapsed into repetitive text, which is a failure of translation rather than evidence that faithful rewriting removed a marker. Excluding the three marked pairs, the post-translation findings for biases 2, 3, and 4 remain 8/21, 14/20, and 14/19, respectively. The pattern persists after inspection. However, passing the inspection does not prove that every detail of meaning was preserved.

It is also worth noting: these are 21 prompts reused across configurations, not 168 independent prompts. Furthermore, this new run uses a different translator than the previous cloud comparison, so the two tables answer two separate questions: how those specific alternative phrasings compared against round-trip translations, and how the strength of the initial watermark was affected in the local translation pipeline.

You can examine the evidence instead of relying on averages. In the accompanying dataset, `records.csv` contains every text with its prompt ID, watermark setting, number of green hits, number of rated pairs, and z-score; match the same `id` across both `arm` values to read a before-and-after pair. `originals.json` and `translations.json` preserve the source text, the Arabic step, and the returned English, while `results.json` adds length and completeness checks. A single library paragraph shifts from "easy and affordable" to "easier and more sustainable." Same topic, but a different claim. Reading the text is part of the experience, not an optional add-on.

## Translation can preserve meaning yet lose the mark

A translator attempts to convey a message into another language. There is no reason for them to preserve a preference tied between pairs of English symbols. In the reverse direction, some familiar phrases return. Others do not.

This is the *semantic bottleneck*: many different phrasings can carry nearly the same message. A transformation that preserves the message is free to discard details of the original phrasing, and the statistical pattern of the watermark is precisely such a detail.

It does not even require randomness to do so. A deterministic paraphrase can map several different sentences to a single canonical form, destroying information about the originally chosen phrasing. The converse is also true: sample-based paraphrasing may carry over enough of the original phrasing to leave a trace behind.

One-way translation and round-trip translation are different tests. In [*Can Watermarks Survive Translation?*](https://arxiv.org/abs/2402.14007), he and his colleagues found significant cross-lingual vulnerabilities in the methods they tested, and proposed X-SIR to improve cross-lingual consistency. Therefore, token differences do not make detection impossible by definition. The watermark design is what matters.

Paraphrasing has a similarly mixed track record. The DIPPER study by Krishna et al. [demonstrated significant drops in detector performance](https://arxiv.org/abs/2303.13408). However, the reliability study by Kirchenbauer et al. found residual watermarks that remained detectable after both human and machine paraphrasing; in their human-paraphrasing experiment, detection required on average about 800 observed tokens, with a reported false-positive rate of `10⁻⁵`. These results coexist because the transformations, lengths, and evaluation conditions differ. A weak residual signal may be inconclusive in a single paragraph yet devastating across a much larger sample.

The story continues to unfold. The [2026 revision of Muhammad and Gubrey's research](https://arxiv.org/abs/2510.18019) places inverse translation inside the *detector*, seeking translations that recover a weakened signal. The attacker's tool also works in favor of the defender.

## "Meaning Preservation" Requires Independent Verification

The eight-segment comparison used an embedding model to compare the original text with the translated text. The average cosine similarity was approximately 0.95 for the Arabic round-trip translation and 0.84 for a single stage of paraphrasing.

It is tempting to read that as "95% of the meaning was preserved" and "84% was preserved." Resist this temptation. Cosine similarity measures how close two model representations are to each other. It is not a measure of real-world accuracy, nor is it a percentage of anything.

The preserved examples show why this distinction is worth its weight. A single paragraph about night buses claims that reduced noise improves air quality; the faithful preservation of that dubious causal claim does not make it true by virtue of its faithful repetition. Elsewhere, a localized translation turns a reference to bus services into “these centers.” The topic remains recognizable while the details are silently lost.

The early-stage evaluation I conducted across seven languages suffered from a more severe version of this issue. Some outputs were significantly shorter; one round-trip translation pass in German reduced a 200-character passage to a 14-character sentence. The drop in detection score there reflects material loss as much as it reflects rephrasing, making that evaluation a weak basis for ranking languages by their effectiveness at removing the watermark.

If I were running the stronger version of this experiment, I would verify deletions, names, numbers, causal claims, and incomplete sentences in addition to the similarity metric, compare transformations at comparable output lengths, and use more commands, keys, and decryption seeds. Otherwise, a method that deletes half the argument can appear deceptively successful.

## Most Closely Related: Information and Choice Space

There is room for a watermark to work only where multiple plausible continuations exist. If the model is effectively forced into a single token, a small bias changes nothing. A stronger bias can enforce a different choice, but in that case maintaining answer quality becomes harder. The [SynthID-Text paper](https://www.nature.com/articles/s41586-024-08025-4) discusses both text length and next-token uncertainty as factors affecting detectability.

The precise way to talk about this is *Kullback–Leibler divergence*: how far the labeled probability distribution is from the unlabeled one. It is a standard quantity in information theory, and Leon Chlon’s [watermarking tool](https://github.com/leochlon/watermark-edfl-toolkit) is what prompted me to look at the experiment through this lens.

For a green-list watermark, the arithmetic is concise. Let `q` be the probability mass on the green list before boosting. Note that this differs from `γ`, the fraction of vocabulary entries on the list; it need not be the case that a quarter of the vocabulary occupies a quarter of the probability. If the watermark adds a bias `δ` to the green logits (the scores used to generate probabilities), then:

```text
q_watermarked = e^δ q / (1 − q + e^δ q)
```

The probabilities of choosing the green color are multiplied by `e^δ`. As a simple illustrative example, when `q = 0.25` and `δ = ln(2)`, these probabilities are doubled, raising the probability of the green color to 0.40.

The corresponding police KL is:

```text
KL = δ × q_watermarked − ln(1 − q + e^δ q)
```

This is the one-step distributional shift, in nats. It approaches zero when all probability mass is already on the green symbols, or when there is no probability mass on them. Color preference changes the results only where there is probability mass that can be transferred between groups.

You can sum these conditional KL values along a generated segment. However, the sum is not automatically a detection score, nor does it guarantee anything about survival after translation. The sequence-level exact KL average is taken over prefixes that the water model can generate:

```text
KL(Pw(sequence) || P(sequence))
  = E under Pw [sum over steps of KL(Pw(next | prefix) || P(next | prefix))]
```

Here, `P` refers to the same generator without its watermark, not all human writing. Matching some of the remaining word bigrams is not a measure of how much of this distributional information remains after applying the transformation.

This distinction led to a rewrite of this article. A previous draft had claimed that 87% of the watermark information was in uncertain locations, and that "retained information" predicted detection with a correlation of 0.90. The reconstruction of the likelihood behind those figures omitted the prompt and some sampling settings, and the retention metric was a weighted proxy for overlap. I retracted both claims. The revised code makes its assumptions explicit, which the original version should have done.

What remains is the acoustic part: applying the transformation itself to both the labeled and unlabeled text cannot increase the full-distribution KL divergence between them. This is the data-processing inequality. It permits a reformulation that eliminates the evidence. However, it does not determine how much evidence a given translation discards, nor does it force any individual z-score to decrease.

## What a Lower Score Actually Tells You

If the result is below the threshold, this means that this detector did not find sufficient evidence to distinguish this text under this key. This is the complete conclusion.

That does not prove the text was written by a human. Nor does it prove that all traces of digital watermarks have disappeared, or that the phrasing differs from the provider's previous outputs. Moreover, a moderately positive score below the threshold is not a distinct category of "obviously non-human" writing; texts free of digital watermarks fall within this range by chance all the time.

Residual evidence can also accumulate across independent segments, provided the detector correctly handles repeated material and multiple tests. A second source of full evidence is searching the provider's stored outputs. [The DIPPER paper](https://arxiv.org/abs/2303.13408) studies retrieval as a defense against paraphrasing. Its usefulness depends on the availability of logs, choosing a reasonable matching method, and calibrating false matches, so it is a tool rather than a guarantee.

Where does that place watermarking technology? It is useful, once you know its operating conditions. It gives the provider a specific statistical signal to test for, and ordinary copying preserves that signal perfectly. Translation and paraphrasing draw the line: the textual source stored in word choices lasts exactly as long as those choices do, while readers continue to recognize the idea long after the evidence of its original phrasing has disappeared.

## Run It Yourself

Start by opening the saved data. You do not need a key, model weights, or Python to read `fresh-translation-2026-09-13/records.csv` in a spreadsheet. The accompanying demo folder is `watermark-robustness/`; the original working version is `robustness-eval/`.

To convert a new hosted model, log in to the [Superlinked Console](https://console.superlinked.com) and obtain your account's API key. If hosted access is not enabled, the [Superlinked website](https://superlinked.com/) provides a request form to enable inference. Verify your account access and available credits before making requests.

From the utilities folder, with Python 3.12 and [`uv`](https://docs.astral.sh/uv/) installed:

```bash
uv sync --locked
uv run --frozen python run_paraphrase.py --ask-key --limit 1 \
  --arms rt_ar paraphrase_1x --output runs/my-sie-demo.json
```

The script requests the key without displaying or storing it. As an alternative, you can provide `SIE_API_KEY` via environment variables or your secrets manager. Never paste a real key into a published example.

This small prototype takes the first saved, watermarked paragraph, translates it into Arabic and then back to the original language, and separately rewrites the original text once: three total content-generation calls via hosting. It evaluates both returned texts locally, using the original watermark key and tokenizer. The SIE API key authorizes requests; the watermark numeric key defines the statistical pattern. Different keys, different tasks.

The endpoint is `https://api.superlinked.com`, and it can be configured using `SIE_BASE_URL` or `--sie-url`. The scripts use `SIEClient.chat_completions`, an SDK method for the SIE [chat-completions](https://superlinked.com/docs/generate) endpoint. The saved cloud comparison used `Qwen/Qwen3.6-27B`; whereas the updated demo defaults to `Qwen/Qwen3.8-27B-FP8`, which can be selected via `SIE_GENERATOR_MODEL`. Check the endpoint catalog before running: a new model means a new comparison, not the experiment behind the first table. The demo stops if the model is missing, the balance is insufficient, the timeout expires, or the response is incomplete, rather than silently retrying. I also learned the hard way to respect hardware details: on the first run of the detector, all watermarked segments received near-zero scores because generation was performed on the GPU while detection was performed on the CPU, and the generator in the green list differs silently between the two. For this reason, both generation and detection use the central processing unit (CPU) here.

The results are saved in a new file, not in either of the published tables. Reruns may differ due to changes in the sample and the hosted model settings. A single run is only a connectivity check, not a new robustness study. The README also explains how to generate new local source texts and how to re-evaluate saved experiments without SIE calls.

The second table is a local, independent MADLAD experiment and does not require an SIE key. Running it requires downloading the model assets and installing `ctranslate2==4.8.1`; the exact commands and a separate output directory are in the README file. If you want to see the semantic bottleneck applied to something you just read, the accompanying folder contains this very article after its own journey from English to Arabic and back to English. The argument remains intact. The phrasing, however, shifts on nearly every line.

---

*Experimental note, September 2026: These are synthetic test clips, including fictional topics and some outputs that are incomplete or unreliable in terms of real-world accuracy. All watermarking tests pertain to locally generated texts under known experimental keys. The first table presents an out-of-line re-evaluation of archived conversion operations. The second table presents a new local translation experiment comprising 21 prompts. The accompanying files retain the new texts, settings, completion flags, and all 168 records of texts, scores, and sensitivity checks. Neither experiment measures active commercial assistants.*

### Further Reading

- Kirchenbauer et al., [*A Watermark for Large Language Models*](https://proceedings.mlr.press/v202/kirchenbauer23a.html), ICML 2023. Green list scheme.
- Kirchenbauer et al., [*On the Reliability of Watermarks for Large Language Models*](https://arxiv.org/abs/2306.04634), ICLR 2024. Residual detection after rewriting.
- Dathathri et al., [*Scalable watermarking for identifying large language model outputs*](https://www.nature.com/articles/s41586-024-08025-4), Nature 634, 818–823, 2024. SynthID-Text.
- He et al., [*Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models*](https://arxiv.org/abs/2402.14007), ACL 2024. Translation attacks and X-SIR.
- Krishna et al., [*Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense*](https://arxiv.org/abs/2303.13408), NeurIPS 2023. DIPPER and retrieval.
- Mohamed and Gubri, [*Is Multilingual LLM Watermarking Truly Multilingual? Scaling Robustness to 100+ Languages via Back-Translation*](https://arxiv.org/abs/2510.18019), unpublished draft, revised in March 2026. Translation as part of detection.
