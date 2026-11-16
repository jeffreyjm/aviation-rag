# aviation-rag

Precedent search and cited investigation drafting over 9,068 NASA ASRS aviation safety reports, with an evaluation harness that needs no hand-written gold set.

Given a free-text description of a new event, it:

1. retrieves the most similar past reports (hybrid BM25 + dense + cross-encoder rerank, with metadata filters such as flight phase and FAR part),
2. summarises what the analysts' structured labels say across those precedents, with **lift versus the corpus base rate** so distinctive factors stand out from merely common ones,
3. drafts a cited summary that treats contributing factors as hypotheses to check, and **abstains** when the best match is weak.

It surfaces precedent for a human investigator. It never asserts a cause.

```
$ aviation-rag investigate "On a night approach in IMC the crew got a TCAS RA just as the autopilot
    disconnected, and the captain overcorrected and descended below the assigned altitude." \
    --phase "Initial Approach" --far-part "Part 121"

PRECEDENTS
  [2] ASRS 1764056 | B737-800 | Initial Approach | Part 121 | 2020
      B737-800 Captain reported TCAS warning on approach.
  [4] ASRS 1148081 | B747-400 | Initial Approach | Part 121 | 2014
      A B747-400 flight crew descending for landing ... descended below their cleared altitude when the
      pilot flying failed to reset his altimeter to QNH passing the transition altitude.
  ...
FACTORS ACROSS PRECEDENTS (count / share / lift vs corpus)
  Anomaly: Deviation / Discrepancy - Procedural Clearance 3/60%/x2.9; Deviation - Track / Heading 2/40%/x4.7
  Result:  Flight Crew Returned To Clearance 4/80%/x11.8; ...
NOTE: Precedents are voluntary, self-reported and de-identified; absence of a factor is not evidence it was absent.
```

## Why it is built this way

Investigation work asks "have we seen this before, what contributed, and what was done about it?" That is a retrieval problem over messy narrative records with structured metadata, where wrong or unsupported answers are costly. So the design choices are about trust: every claim carries a validated citation, weak evidence triggers a refusal, labels are kept out of the index so retrieval quality can be measured honestly, and all of it is evaluated with confidence intervals and baselines.

(The structure resembles other deviation/incident investigation settings, but this project makes no claim about any regulated domain; it is a demonstration of the method on public aviation data.)

## Results

Everything below comes from `aviation-rag eval` (full output: [`eval/results/results.md`](eval/results/results.md)). 9,068 reports, 34,775 chunks.

### How the evaluation works (no hand-labelled gold set)

Every ASRS report has an analyst-written **synopsis** that the index never sees. So:

- **Known-item retrieval**: query = a report's synopsis, target = that report. Synopses paraphrase the narrative (lexical overlap only **0.47**), and 400 queries are sampled with a fixed seed, so there is no author bias.
- **Precedent quality**: for the same queries, drop the query's own report and ask whether the nearest *other* reports share its analyst labels. Labels are **never in the indexed text** (enforced by a unit test), so this is not leakage. Each number is shown beside a random-neighbour baseline.

### Known-item retrieval (n=400)

| mode | recall@1 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| bm25 | 0.510 [0.46, 0.56] | 0.698 [0.65, 0.74] | 0.745 [0.70, 0.79] | 0.588 [0.55, 0.63] |
| dense | 0.383 [0.34, 0.43] | 0.552 [0.51, 0.60] | 0.632 [0.59, 0.68] | 0.458 [0.41, 0.50] |
| hybrid (RRF) | 0.500 [0.45, 0.55] | 0.695 [0.65, 0.74] | 0.750 [0.71, 0.79] | 0.580 [0.54, 0.62] |
| hybrid + rerank | 0.527 [0.48, 0.57] | 0.720 [0.67, 0.76] | 0.772 [0.73, 0.81] | 0.609 [0.57, 0.65] |
| hybrid + rerank + phase filter* | **0.650** [0.60, 0.70] | **0.818** [0.78, 0.85] | **0.870** [0.83, 0.90] | **0.721** [0.68, 0.76] |

Cells are mean [95% bootstrap CI over queries].
\*The filter uses the target report's true flight phase, which a real user would supply from what they know, but it is an optimistic setting.

- **BM25 is hard to beat; dense retrieval alone is clearly worse** (recall@1 0.38 vs 0.51, intervals do not overlap). Pilot narratives are full of exact tokens (waypoints, callouts, aircraft types) that lexical matching handles well.
- **Fusion and reranking help only a little**: hybrid+rerank is +1.7 points recall@1 over BM25, well inside the intervals. I would not claim a difference at this sample size.
- **Metadata filtering is the biggest lever**: constraining to the right flight phase adds ~12 points recall@1 (0.527 → 0.650, intervals do not overlap).

### Precedent quality (n=368 labelled queries, top-5 neighbours, own report excluded)

| mode | primary-problem precision@5 | primary-problem vote accuracy | anomaly overlap@5 |
|---|---|---|---|
| bm25 | 0.426 [0.39, 0.46] | 0.524 [0.47, 0.58] | 0.727 [0.70, 0.75] |
| dense | 0.409 [0.38, 0.44] | 0.495 [0.45, 0.54] | 0.733 [0.70, 0.76] |
| hybrid | 0.432 [0.40, 0.46] | 0.549 [0.50, 0.60] | 0.752 [0.72, 0.78] |
| hybrid + rerank | 0.427 [0.40, 0.46] | 0.546 [0.50, 0.60] | 0.739 [0.71, 0.77] |
| hybrid + rerank + phase filter | 0.407 [0.38, 0.44] | 0.524 [0.48, 0.57] | 0.704 [0.68, 0.73] |

**Baselines:** random neighbours share the primary problem **0.223** of the time; always predicting the majority class ("Human Factors") is right **0.380**; two random reports share an anomaly category **0.398**.

- Retrieved precedents are meaningfully better than chance: ~1.9x the random primary-problem agreement, and anomaly overlap of ~0.74 vs 0.40.
- **But it is not a strong classifier**: a majority vote over five neighbours predicts the primary problem 55% of the time vs 38% for always guessing the most common class. Analyst primary-problem labels are partly subjective, and similar events do not always share one.
- **Retrieval modes are indistinguishable on this metric**, and the phase filter does not improve it (it improves finding a specific report, not label agreement). Reading these two tables together, the filter is a tool for narrowing, not for making precedents "more relevant".

### Abstention

The top reranker score separates in-domain synopses from 30 out-of-scope event descriptions written in the same voice (20 near-domain: rail, maritime, medical, industrial, grid; 10 far-domain): **AUROC 0.994** (near-domain 0.992, far-domain 1.000). A threshold calibrated on half the questions kept 98% of in-domain queries (n=200) and rejected 93% of out-of-scope ones (n=15) on the held-out half.

The held-out set is tiny (15 out-of-scope queries, one miss). The query that passes is informative: *"A chemical plant technician reported opening the wrong valve ... because the tag was missing"* matches an aircraft maintenance-procedure report. Near-domain procedural failures are where this score is weakest.

### Drafts

Offline extractive baseline over 100 in-domain synopses and the 30 out-of-scope queries: 99% of in-domain drafts produced, 96.7% of out-of-scope refused, citation validity 1.000, citation recall 0.707 (a cited report is the one the synopsis came from), groundedness 1.000 over 297 checked claims. The 1.000 groundedness is expected, since the baseline copies source sentences; it validates the harness, not the system. **The LLM generator has not been evaluated: there was no API key in the environment where this was built.** It is covered by unit tests against a fake client (task-specific prompts, citation handling, abstention token) only. `aviation-rag eval --llm` runs the real evaluation.

## How it works

```
event description (+ optional filters)
   |-- BM25 (80) ---------\
   |                       +-- RRF -- top 40 -- cross-encoder rerank -- collapse to reports -- top-k precedents
   |-- dense cosine (80) -/       (filters applied BEFORE ranking)                  |
                                                                  abstain if top score < threshold
                                                                                    |
                              factor aggregation (counts, lift) <-- precedents --> cited draft (LLM or extractive)
```

- **Chunking** (`chunk.py`): pilot narratives are shorthand, ALL-CAPS and sometimes unpunctuated, so chunks are packed to a 700-character budget with one sentence of overlap, and run-on text is hard-wrapped. Each chunk is prefixed with neutral context (aircraft | phase | FAR part | year).
- **No label leakage**: analyst labels (primary problem, anomaly, factors, results) live only in the report store, never in indexed text or filter metadata. This is what makes precedent-quality evaluation valid.
- **Filters before ranking** (`retrieve.py`): excluded chunks can never be returned, so a filtered search is a real constrained search and `k` results come back whenever they exist.
- **Factor aggregation with lift** (`investigate.py`): counts across the retrieved precedents' labels, compared with the corpus base rate. Labels carried by a single precedent are suppressed when there are enough precedents, because their lift is unstable (an early version showed "x16" for a label on one report).
- **Citations** (`generate.py`): unknown citation numbers are stripped and reported; the model's `INSUFFICIENT_EVIDENCE` becomes an abstention; the system prompt forbids stating a definitive cause.
- **Groundedness** (`evaluate.py`): claims are checked with an NLI model against single source sentences and two-sentence windows, taking the best support. Small NLI models label a claim "neutral" when the premise contains it plus other text (a verbatim sentence scored 0.002 against its own chunk), so whole-excerpt checking is unreliable.

## Quick start

```bash
make setup      # venv + deps
make data       # download ASRS reports (~63 MB) -> data/reports.jsonl
make index      # chunk + embed (~7 min on an idle CPU; ~16 min on a busy laptop)
make test       # 38 unit tests, <1s, no model downloads
make eval       # reproduce the tables above (~30-40 min on CPU)

.venv/bin/aviation-rag investigate "<describe the event>" --phase "Landing" --far-part "Part 121"
.venv/bin/aviation-rag ask "How do crews handle a stuck thrust lever on landing?"
ANTHROPIC_API_KEY=... .venv/bin/aviation-rag investigate "..." --generator llm
```

Filters: `--phase --far-part --operator --conditions --aircraft --year-min --year-max`.

## Data

NASA Aviation Safety Reporting System: public, de-identified voluntary reports. Obtained via the Hugging Face mirror [`elihoole/asrs-aviation-reports`](https://huggingface.co/datasets/elihoole/asrs-aviation-reports) (test + validation splits: 9,068 reports, 2012-2022). Reports without a narrative or synopsis are dropped. Raw data and the index are not committed; `make data` and `make index` rebuild them deterministically.

## Limitations

- **Known-item is a proxy.** The synopsis is a cleaner, better-formed query than an investigator's rough description, and the single target report may have equally valid near-duplicates, so recall@1 understates true quality while the clean query overstates it. The two effects do not cancel in any known way.
- **Label agreement is not usefulness.** Nobody has judged whether the precedents help an investigator; that needs a domain expert reviewing retrieved cases.
- **Primary-problem labels are noisy**, and the "Ambiguous" class is excluded from that metric.
- **Small out-of-scope set** (30 queries, 15 held out), and abstention is calibrated on the same sample type it is scored on, so treat 93% as indicative.
- **Phase-filter results use the target's true phase.**
- **LLM drafting is untested end to end.** Only the extractive baseline has run.
- **A 9k-report subset**, not the full ASRS database; self-reported narratives carry reporter bias.
- Hyperparameters (RRF k=60, 80 candidates, rerank top 40, 700-char chunks) were not tuned.

## Layout

```
src/aviation_rag/  data.py chunk.py index.py retrieve.py generate.py investigate.py evaluate.py report.py cli.py
eval/ood_queries.jsonl    30 out-of-scope event descriptions
eval/results/             latest evaluation output (json + markdown)
tests/                    38 tests using fake models (incl. a label-leakage guard)
```
