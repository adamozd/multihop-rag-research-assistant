# Frozen baseline and live acceptance

Version 1 contains 18 fixed questions: 5 single-hop, 9 multi-hop, and 4 adversarial/no-answer cases. The expected-paper and abstention labels were frozen before this benchmark's inference. They have not been independently human-validated. The comparison question was used during development, so this is a development benchmark, not a held-out generalization test.

Do not alter labels after seeing results. If a label is discovered to be wrong, document the issue and create a separately versioned benchmark; retain v1's results and disclose any correction.

From the repository root, with the existing `.env` configured:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -u -m eval.run_eval --questions eval/benchmarks/v1/questions.json
HF_HUB_OFFLINE=1 .venv/bin/python -u -m eval.run_acceptance
```

`HF_HUB_OFFLINE=1` keeps embeddings local; watsonx calls still use the network and consume quota. A complete benchmark normally needs roughly 72–126 reasoning calls, depending on abstentions, hops and revisions; JSON repairs can add calls. The acceptance probes add calls. Each command prints its output directory. A new invocation creates a separate run; do not rerun blindly or combine selected successes from different runs.

The benchmark compares each original draft with its own post-critique answer using the same evidence. It does not compare independently sampled end-to-end systems. The automatic report is an internal critic proxy, not human support accuracy. Source hashes, question hash, corpus-manifest hash, settings, per-question duration and completion status are recorded. SDK token usage/cost is not yet recorded.

## Human review

1. Choose a real human reviewer. Ideally use someone who has not seen the generated critique. Give them `human-review.json` and these instructions. Do not give them `run.json`, which reveals which claims are drafts versus revisions.
2. Preserve every `review_id`, question, claim and cited excerpt. Change only `label` and `notes`. Each label starts as null; no AI labels have been inserted.
3. Read the complete claim against its cited excerpts. Check all factual parts, numbers, qualifiers, optional versus required behavior, training versus inference, and the scope of comparisons. Valid-looking citation IDs alone are insufficient.
4. Set `label` to `supported` only when the entire claim follows from the cited evidence; `unsupported` for missing or partial support; `contradicted` when evidence directly conflicts with it. A comparison may follow from separately supported descriptions of each method; it need not appear word-for-word in one paper. Hedging does not automatically rescue an unsupported assertion.
5. Add a short justification in `notes`, especially for disputed claims. Do not use outside knowledge to rescue a claim whose cited excerpts do not support it. Invalid or missing cited evidence fails support.
6. Record reviewer name/identifier, review date and any adjudication separately. For stronger results, obtain two independent reviews and resolve disagreements without changing the original claims.
7. Return the completed JSON and score it:

```bash
.venv/bin/python -m eval.run_eval --score-run eval/results/RUN_ID/run.json --annotations eval/results/RUN_ID/human-review.json
```

The scorer requires every claim label. It cannot verify that a human supplied them: provenance must be documented honestly. `human-report.json` contains pooled support rates and their percentage-point difference, paper recall, answerable response rate and no-answer abstention rate. Zero claims produce an undefined support rate, not 100%. Also inspect answer completeness and limitations; these are not measured by claim support alone.

Review failures and completion status before reporting a full benchmark. Scores for a partial run must be labeled partial. Inspect `acceptance.json` separately: successful execution is not semantic acceptance. The controlled revision probe deliberately injects an incorrect claim and is not included in benchmark scores. Its automatic critic verdict still needs human review.
