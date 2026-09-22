# Hybrid RAG search with a cross-encoder reranker — optional add-on module for NeXgen Engine

Retrieval for a Markdown knowledge base: vector search, BM25, and a title/filename signal fused with RRF, then re-scored by a quantized cross-encoder. CPU only, one container, no GPU, no vector database.

This is the search layer behind a set of AI coding agents that query a private knowledge base through MCP. It has been running in production on an ARM VM since June 2026 (4 cores until August 2026, 2 cores since). The code here is that code, not a reconstruction.

## Optional add-on for NeXgen Engine

This is the retrieval layer of [NeXgen Engine](https://github.com/matteopasseri407/NeXgen-Engine): the thing its agents query when they search a Markdown knowledge base. It is optional, and strongly recommended once that base grows past what plain keyword matching can find.

The engine runs without it: retrieval falls back to lexical search, nothing breaks. Install this container and the same corpus becomes searchable by meaning, with the fusion and the reranker described below, and no change on the engine side.

## Measured, on the production corpus

2,564 chunks across 373 files, scored against a 19-query gold set (September 2026, production VM with 2 ARM cores — same code, same corpus, reranker toggled by `RERANK_ENABLED`):

| Configuration | hits@1 | hits@5 | MRR | latency/query (p50) |
| --- | --- | --- | --- | --- |
| Fusion only (RRF + title boost) | 13/19 | 17/19 | 0.781 | ~6 ms |
| **+ cross-encoder reranker** | **15/19** | **19/19** | **0.879** | **~460 ms** |
| Reranker fed bare snippets | — | **12/20** | — | ~900 ms |

(June 2026 baseline, 1,284 chunks / 202 files / 4 cores / 20 queries: fusion 12/20 hits@1, 15/20 hits@5, MRR 0.662; +reranker 13/20, 16/20, 0.702 at 150–250 ms. The September rerun also includes two production fixes below: FTS terms are now double-quoted, and the synonym list grew from real misses.)

The reranker is the entire latency budget: the vector multiply, the SQL, and the fusion are all sub-millisecond.

That third row is the point of this repository. The first version of the reranker made retrieval **worse than no reranker at all** — 12/20 against a 15/20 baseline — while looking completely healthy: the model loaded, scores came back, results were plausibly ordered. Nothing but the gold set caught it.

The cause: the corpus leans heavily on the title/filename signal (`W_TITLE=0.6` is not a decorative weight), and a cross-encoder handed a bare body snippet cannot see that signal. Prefixing each candidate with `"{title} — {filename}"` before scoring — the same contextual framing already used when embedding — took it to 16/20. One line of string formatting, and the difference between a feature that helps and a feature that quietly hurts.

## Quick start

```bash
cp .env.example .env
CORPUS_PATH=/path/to/your/markdown docker compose up --build

curl 'http://127.0.0.1:8089/health'
curl -G 'http://127.0.0.1:8089/search' --data-urlencode 'q=your query' --data-urlencode 'k=5'
```

The reranker weights are not committed — 118 MB of ONNX does not belong in a git repository. To enable stage 4:

```bash
pip install torch transformers onnx onnxruntime optimum
python export_reranker.py          # writes reranker/model.int8.onnx
docker compose restart             # /health now reports "reranker": true
```

Without it the service still starts and serves; it reports `"reranker": false` and runs fusion-only. That is the same path taken if the model file is corrupt, so the degraded mode is exercised whether you plan for it or not.

## Architecture

Four stages. The first three always run; the fourth re-scores their winners.

**1 — Vector search.** Cosine similarity against `minishlab/potion-multilingual-128M` via `model2vec`: static embeddings, no attention, no GPU, sub-millisecond encode. The reason for choosing a static model is not quality, it's that a *full* reindex of the corpus costs 1–6 seconds, which makes reindexing-on-a-timer a valid design instead of a background job with its own failure modes.

**2 — BM25.** SQLite FTS5 over the same contextualized chunk text. Everything — embeddings as blobs, the lexical index, file metadata — lives in one SQLite file, so there is no second service to run, monitor, or lose.

**3 — Title/filename boost.** A third ranking that matches the query *only* against titles and filenames, never the body. This is what makes an exact-name query beat a semantically-similar-but-wrong note.

**Fusion — RRF.** A candidate at rank `r` in a list contributes `weight / (RRF_K + r)`. Production weights: `W_VEC=1.0`, `W_BM=0.4`, `W_TITLE=0.6`, `RRF_K=10`, `TOPN=50`. Setting any weight to `0` disables that signal — a first-class rollback lever, not a special case in the code.

**4 — Cross-encoder reranker.** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, multilingual and trained on mMARCO which includes Italian, exported to ONNX and dynamically quantized to int8 (471 MB → 118 MB). It re-scores the top `RERANK_POOL` candidates and cuts to `k` by its own order.

## Latency, on the hardware it actually runs on

Two findings, both measured on the production ARM VM rather than estimated (pool finding on 4 cores; latency re-measured at ~460 ms p50 after the August 2026 resize to 2 cores halved the ONNX thread count):

**Pool size and sequence length dominate.** Everything around the reranker — the vector multiply, the SQL, the fusion — is sub-millisecond. A pool of 20 candidates at 480-char snippets cost ~860–900 ms per query, too slow to sit in front of an interactive agent. Narrowing to **10 candidates at 240 chars** brought it to ~150–250 ms with *equal or better* quality: RRF already places the right candidates in the first ten, so the wider net bought latency and nothing else.

**ONNX Runtime does not use all your cores by default.** Setting `intra_op_num_threads` explicitly to the CPU count roughly halved inference time — 166 ms to 98–112 ms per batch. This is easy to miss because the service works perfectly without it, just at half speed.

## Failure behavior

The retrieval path is the dependency of an agent's answer, so it degrades rather than fails:

- Reranker fails to load at boot (missing file, corrupt export, OOM) → log it, serve fusion-only, report `"reranker": false` on `/health`. Never a failed boot.
- Reranker throws on a single query (degenerate input, transient memory pressure) → that query falls back to fusion order. The exception does not escape `search()`.
- BM25 query fails to parse → the lexical signal drops out for that query and vector plus title-boost still answer. Query terms are double-quoted before being sent to FTS5, so `:`, `-`, `*` and column-name-looking tokens can no longer kill the stage (this exact failure was observed in production logs and fixed September 2026).
- `RERANK_ENABLED=0` produces the same behavior as a failed load, so the fallback path is one env var away and gets exercised in normal operation instead of only during an incident.
- The corpus is mounted **read-only**. The service writes to its index volume and nowhere else.
- The container declares a memory limit. An uncapped container that OOMs can take the host down with it.

## Footprint

| Configuration | RAM |
| --- | --- |
| Fusion only | ~1.2 GB |
| With reranker | ~1.6 GB |

2 GB is the floor for a memory cap, and the whole thing fits on an always-free ARM tier alongside other services. No GPU, no managed vector store, no per-query API cost — retrieval quality here was bought with a quantized 118 MB model and a gold set, not with a bill.

## Interface

`GET /search?q=<query>&k=<int>` →

```json
{ "query": "...", "results": [ { "path": "...", "title": "...", "score": 0.87, "snippet": "..." } ] }
```

`GET /health` → `{ "status": "ok", "reranker": true, "chunks": 1284, "files": 202 }`

This contract is what an MCP server in front of it calls to expose a `semantic_search` tool to agent CLIs. Any backend that speaks it is a drop-in replacement; the [full build specification](https://github.com/matteopasseri407/NeXgen-Engine/blob/main/03-INFRA/deploy/semantic-search-recipe.md) documents every weight and model so the thing can be rebuilt from scratch.

## Evaluating a change

```bash
python eval/run_eval.py --url http://127.0.0.1:8089 --queries eval/queries.json --verbose
```

Reports hits@1, hits@5, MRR and latency. `eval/queries.example.json` shows the format and, more usefully, what a gold set needs to cover: a concept query whose answer never contains the words used to ask it, an exact-filename query that must win on the title signal alone, and a cross-language query. The gold set for the production corpus is not in this repository — a gold set is a description of its corpus, and this one describes a private one.

## What this is not

- Not a general-purpose RAG framework. It indexes Markdown files and returns ranked passages. There is no generation step, no agent loop, no chunk-level citation machinery.
- Not a benchmark contribution. Twenty queries on one 202-file corpus measures *this* system on *this* content. It is enough to catch a regression and not enough to generalize.
- The comments and log lines in `src/` are in Italian. This is the original production code, published as it runs.

## Sintesi in italiano

E' un modulo **opzionale e fortemente consigliato** del [NeXgen Engine](https://github.com/matteopasseri407/NeXgen-Engine): e' il layer di retrieval che gli agenti del motore interrogano via MCP. Il motore funziona anche senza (ricerca lessicale); con il modulo, la stessa base di conoscenza diventa interrogabile per significato.

Layer di retrieval ibrido per una base di conoscenza in Markdown: ricerca vettoriale, BM25 e un segnale su titolo/nome-file, fusi con RRF e poi riordinati da un cross-encoder quantizzato int8. Solo CPU, un container, nessun vector database. In produzione su una VM ARM a 4 core da giugno 2026.

La parte che conta è la terza riga della tabella in alto: la prima versione del reranker **peggiorava** i risultati (hits@5 12/20 contro un baseline di 15/20) sembrando perfettamente funzionante. Il modello caricava, i punteggi tornavano, l'ordine era plausibile. Se l'è mangiata solo il gold set. La causa era che il cross-encoder riceveva lo snippet nudo, senza il prefisso titolo/nome-file su cui questo corpus porta gran parte del segnale.

È il motivo per cui questo repo contiene un harness di valutazione e non solo del codice: su un sistema di retrieval, "adesso sembra migliore" non è una misura.

## License

PolyForm Noncommercial 1.0.0 — see [LICENSE](LICENSE).
