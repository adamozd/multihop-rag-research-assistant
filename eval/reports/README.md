# Evaluation reports

`2026-10-05-partial-evaluation.json` is a sanitized summary of the first live watsonx benchmark attempt. It contains configuration, source hashes, per-question aggregate results and internal critic metrics. It contains no human support labels or raw paper excerpts.

**This is a partial run, not a completed benchmark or an accuracy claim.** Eleven of eighteen questions completed before token-quota exhaustion blocked the remainder. All three separate acceptance probes were blocked. Terminal logs supplied by the operator establish quota exhaustion; saved application errors are generic.

| Metric (completed questions only) | Draft | After critique |
| --- | ---: | ---: |
| Internally supported claims / total claims | 33/36 | 32/33 |
| Internal critic support proxy | 91.67% | 96.97% |
| Mean expected-paper recall (6 multi-hop questions) | 83.33% | 75.00% |
| Answerable response rate (11 questions) | 90.91% | 90.91% |
| Adversarial/no-answer abstention | Not measured | Not measured |

The internal support proxy rose 5.30 percentage points, but supported-claim count fell from 33 to 32, total claims fell from 36 to 33, and paper recall declined. These figures do not establish an improvement in answer quality. Two questions triggered revision. The answerable s01 question abstained. Human review is pending for all 69 draft/final claim items.

Raw run directories remain under ignored `eval/results/`. Keep them locally for provenance and human review. Do not publish the review-arm mapping before blinded review. Future public releases of raw artifacts should be deliberate, sanitized and assessed for excerpt redistribution rights; do not force-add the entire results directory.
