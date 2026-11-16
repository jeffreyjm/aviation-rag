### Known-item retrieval (query = analyst synopsis, n=400; synopsis/narrative lexical overlap 0.47)

| mode | recall@1 | recall@5 | recall@10 | mrr |
|---|---|---|---|---|
| bm25 | 0.510 [0.46, 0.56] | 0.698 [0.65, 0.74] | 0.745 [0.70, 0.79] | 0.588 [0.55, 0.63] |
| dense | 0.383 [0.34, 0.43] | 0.552 [0.51, 0.60] | 0.632 [0.59, 0.68] | 0.458 [0.41, 0.50] |
| hybrid | 0.500 [0.45, 0.55] | 0.695 [0.65, 0.74] | 0.750 [0.71, 0.79] | 0.580 [0.54, 0.62] |
| hybrid+rerank | 0.527 [0.48, 0.57] | 0.720 [0.67, 0.76] | 0.772 [0.73, 0.81] | 0.609 [0.57, 0.65] |
| hybrid+rerank + phase filter | 0.650 [0.60, 0.70] | 0.818 [0.78, 0.85] | 0.870 [0.83, 0.90] | 0.721 [0.68, 0.76] |

Cells are mean [95% bootstrap CI over queries]. The target is the single report the synopsis was written for.

### Precedent quality (top-5 neighbours excluding the query's own report; n=368 labelled queries)

| mode | primary-problem precision@5 | primary-problem vote accuracy | anomaly overlap@5 |
|---|---|---|---|
| bm25 | 0.426 [0.39, 0.46] | 0.524 [0.47, 0.58] | 0.727 [0.70, 0.75] |
| dense | 0.409 [0.38, 0.44] | 0.495 [0.45, 0.54] | 0.733 [0.70, 0.76] |
| hybrid | 0.432 [0.40, 0.46] | 0.549 [0.50, 0.60] | 0.752 [0.72, 0.78] |
| hybrid+rerank | 0.427 [0.40, 0.46] | 0.546 [0.50, 0.60] | 0.739 [0.71, 0.77] |
| hybrid+rerank + phase filter | 0.407 [0.38, 0.44] | 0.524 [0.48, 0.57] | 0.704 [0.68, 0.73] |

Baselines: random neighbours share the primary problem 0.223 of the time; always predicting 'Human Factors' is right 0.380 of the time; two random reports share an anomaly category 0.398 of the time.

### Abstention (top reranker score; in-domain synopses vs out-of-scope event descriptions)

- AUROC overall **0.994** (near_domain 0.992, far_domain 1.000)
- Threshold -0.28 calibrated on half; held-out half: 98% of in-domain kept (n=200), 93% of out-of-scope rejected (n=15).

### Investigation drafts: extractive baseline

- abstain_on_out_of_scope: 0.967
- answered_when_in_domain: 0.990
- citation_validity: 1.000
- citation_recall: 0.707
- groundedness_sentence_level: 1.000
- n_claims_checked: 297
