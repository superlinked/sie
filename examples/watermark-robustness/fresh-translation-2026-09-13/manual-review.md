# Qualitative checks on the completed run

All 84 source/translation pairs were checked automatically for detector
consistency, completion indicators and output-length ratios. This note
records inspection of the flagged cases and selected examples; it is not
a claim that all 84 pairs received human semantic-equivalence ratings.

Three distinct pairs trigger the predefined stop/length checks:

- **prompt-06-bias-3:** the source expands into a long list of backup risks
  and reaches the 224-token generation cap. The Arabic translation of the
  long sentence reaches its 256-token cap too. Its high embedding cosine
  (about 0.955) does not negate the incompleteness.
- **prompt-01-bias-4:** the seaside-scene source reaches the generation cap.
  Both translation directions finish normally, but the source's stopping
  condition prevents treating it as an unqualified complete-paragraph case.
- **prompt-02-bias-4:** a translation requested in Arabic produces repetitive
  English about trees, mixed with Arabic. Both directions have capped
  segments. The returned text is about 1.90 times the source token length,
  and cosine similarity is about 0.566. The z-score falls from about 10.76
  to 2.96, but this is a translation failure, not faithful watermark removal.
  Its embedding comparison also exceeds the embedding model's input limit.

The primary result table includes these cases so that the full pipeline's
outputs remain visible. The sensitivity table excludes the same predefined
flags, without selecting on detector score. For biases 2, 3 and 4, its
post-translation detections are 8/21, 14/20 and 14/19, respectively.

Even passing these checks is not proof of complete fidelity. For example,
the unwatermarked library paragraph describes learning as “accessible and
affordable,” while its round trip calls it “easier and more sustainable.”
That changes the affordability claim despite preserving the general topic.

No automated quality metric here establishes that the original source
claims are true. These are generated experimental passages, including
fictional material and some poor explanations. The watermark measurements
are measurements of token-choice patterns, not endorsements of their facts.
