#!/usr/bin/env python3
"""Server semantico model2vec — potion-multilingual-128M da disco locale."""
import os, re, json, time, threading, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import rag_index as RI
import rag_rerank as RR

VAULT = os.environ.get("VAULT", "/vault")
DATA = os.environ.get("DATA", "/data")
DB = os.path.join(DATA, "index.db")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "minishlab/potion-multilingual-128M")
# Vuoto = scarica il modello per ID al primo boot e mettilo in cache in DATA.
# Impostalo a una directory se preferisci montare i pesi da disco (air-gapped).
MODEL_PATH = os.environ.get("MODEL_PATH", "") or EMBED_MODEL_ID
PORT = int(os.environ.get("PORT", "8080"))
REINDEX_SEC = int(os.environ.get("REINDEX_SEC", "900"))
TOPN = int(os.environ.get("TOPN", "50"))
RRF_K = int(os.environ.get("RRF_K", "10"))
W_VEC = float(os.environ.get("W_VEC", "1.0"))
W_BM = float(os.environ.get("W_BM", "0.4"))
W_TITLE = float(os.environ.get("W_TITLE", "0.6"))  # bonus se i termini query compaiono nel titolo/nome-file

# Reranker cross-encoder (query-time, non tocca l'indice): riordina il pool
# di candidati del fusion RRF prima del cutoff a k. Rollback morbido:
# RERANK_ENABLED=0. RERANK_POOL e' quanti candidati (per score fuso) entrano
# nel reranking; oltre quel numero i risultati restano nell'ordine RRF.
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "1") not in ("0", "false", "False", "")
# Latency is dominated by sequence length x pool size (measured live on the
# ARM VPS: 20 candidates x 480 chars ~=900ms, 10 x 240 ~=100-150ms), not by
# model load or the SQL/vector steps around it -- those are all sub-ms.
# 10/240 keeps a real margin over the k<=5 the retrieval protocol expects in
# practice while staying fast; raise both together if a deploy has RAM/CPU
# headroom to spare and wants a wider net.
RERANK_POOL = int(os.environ.get("RERANK_POOL", "10"))
RERANK_SNIPPET_CHARS = int(os.environ.get("RERANK_SNIPPET_CHARS", "240"))

# Espansione bilingue IT<->EN: le note tecniche sono spesso in inglese, le
# query in italiano parlato. La mappa e' SPECIFICA DEL CORPUS -- va scritta
# guardando le query che il tuo corpus sbaglia, non copiata. Quella di
# esempio in synonyms.example.json e' volutamente minima.
SYNONYMS_FILE = os.environ.get("SYNONYMS_FILE", "/app/synonyms.json")

def _load_synonyms(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return {k.lower(): list(v) for k, v in raw.items()}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[boot] synonyms non caricati da {path}: {e!r}", flush=True)
        return {}

SYNONYMS = _load_synonyms(SYNONYMS_FILE)

def expand_query(q):
    terms = re.findall(r"[0-9a-zàèéìòùáíóúäöüñç_]+", q.lower())
    extra = []
    for t in terms:
        for syn in SYNONYMS.get(t, []):
            if syn not in terms and syn not in extra:
                extra.append(syn)
    return (q + " " + " ".join(extra)).strip() if extra else q

STOPWORDS = set("""
il lo la i gli le un uno una di a da in con su per tra fra e ed o od ma se che chi cui
come dove quando perche perché non si mi ti ci vi ne se ho hai ha abbiamo avete hanno
sono sei siamo siete e del dello della dei degli delle al allo alla ai agli alle dal
dalla nel nello nella nei negli nelle sul sulla col coi questo questa questi queste
quello quella quelli quelle mio mia miei mie tuo tua suoi sue quale quali quanta quanto
quanti quante cosa cose piu meno molto poco anche solo gia gia ancora
the a an of to in on for and or how what which when where why is are was were be my your
this that these those with from as at by it its
""".split())

print(f"[boot] modello={EMBED_MODEL_ID} path={MODEL_PATH}", flush=True)
from model2vec import StaticModel
_model = StaticModel.from_pretrained(MODEL_PATH)

if RERANK_ENABLED:
    try:
        RR.load()
        print(f"[boot] reranker attivo: {RR.MODEL_DIR}", flush=True)
    except Exception as e:
        # Un reranker rotto non deve mai spegnere il server di ricerca:
        # degrada silenziosamente all'ordine RRF puro, come prima di questa
        # feature. search() controlla RR.available() a ogni chiamata.
        print(f"[boot] reranker disattivato (init fallita): {e!r}", flush=True)
else:
    print("[boot] reranker disattivato (RERANK_ENABLED=0)", flush=True)

def embed_docs(texts):
    return _model.encode(texts).astype(np.float32)

def embed_query(q):
    v = _model.encode([q]).astype(np.float32)[0]
    return v / (np.linalg.norm(v) + 1e-9)

_lock = threading.Lock()
_state = {"ids": np.array([], dtype=np.int64), "paths": np.array([], dtype=object),
          "M": np.zeros((0, 1), dtype=np.float32), "built_at": None, "n": 0, "files": 0}

def _reload_state():
    conn = RI.connect(DB)
    try:
        ids, paths, M = RI.load_matrix(conn)
        built = RI._meta_get(conn, "built_at")
        nfiles = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    finally:
        conn.close()
    with _lock:
        _state["ids"], _state["paths"], _state["M"] = ids, paths, M
        _state["built_at"] = built
        _state["n"] = int(M.shape[0])
        _state["files"] = int(nfiles)

def _index_once():
    conn = RI.connect(DB)
    try:
        RI.incremental_index(conn, VAULT, embed_docs, EMBED_MODEL_ID)
    finally:
        conn.close()
    _reload_state()

def _reindex_loop():
    while True:
        time.sleep(REINDEX_SEC)
        try:
            _index_once()
        except Exception as e:
            print(f"[reindex] errore: {e!r}", flush=True)

_FTS_CLEAN = re.compile(r"[^0-9a-zA-Zàèéìòùáíóúäöüñç_]+ ", re.UNICODE)

def _fts_query(q):
    toks = [t for t in _FTS_CLEAN.split(q) if len(t) >= 2]
    content = [t for t in toks if t.lower() not in STOPWORDS]
    use = content if content else toks
    return " OR ".join(use) if use else None

def search(q, k=5):
    with _lock:
        ids, paths, M = _state["ids"], _state["paths"], _state["M"]
    if M.shape[0] == 0:
        return []
    q = expand_query(q)   # query expansion bilingue IT<->EN (usata da vettore, BM25 e title-boost)
    qv = embed_query(q)
    sims = M @ qv
    order = np.argsort(-sims)
    vec_rank, seen = {}, 0
    for idx in order:
        p = paths[idx]
        if p not in vec_rank:
            seen += 1
            vec_rank[p] = seen
            if seen >= TOPN:
                break
    bm_rank = {}
    conn = RI.connect(DB)
    try:
        ftsq = _fts_query(q)
        if ftsq:
            try:
                rows = conn.execute(
                    "SELECT c.path FROM fts JOIN chunks c ON c.id=fts.rowid "
                    "WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?", (ftsq, TOPN)).fetchall()
                r = 0
                for (p,) in rows:
                    if p not in bm_rank:
                        r += 1
                        bm_rank[p] = r
            except Exception as e:
                print(f"[bm25] skip: {e!r}", flush=True)
        fused = {}
        for p, r in vec_rank.items():
            fused[p] = fused.get(p, 0.0) + W_VEC / (RRF_K + r)
        for p, r in bm_rank.items():
            fused[p] = fused.get(p, 0.0) + W_BM / (RRF_K + r)
        # --- title/filename boost: terzo segnale, premia chi ha i termini nel titolo o nel nome-file ---
        if W_TITLE > 0:
            qterms = [t for t in re.findall(r"[0-9a-zàèéìòùáíóúäöüñç_]+", q.lower())
                      if len(t) >= 2 and t not in STOPWORDS]
            if qterms:
                for (fp, ftitle) in conn.execute("SELECT path, title FROM files").fetchall():
                    hay = (os.path.basename(fp) + " " + (ftitle or "")).lower()
                    hits = sum(1 for t in qterms if t in hay)
                    if hits:
                        fused[fp] = fused.get(fp, 0.0) + W_TITLE * (hits / len(qterms))
        # Pool piu' largo di k per il reranker (se attivo): il fusion RRF e'
        # un buon RECALL, il cross-encoder e' un buon RANKING fine sopra
        # quel pool. Senza reranker il pool e' semplicemente k.
        pool_size = max(k, RERANK_POOL) if (RERANK_ENABLED and RR.available()) else k
        pool = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:pool_size]

        candidates = []
        for p, fused_score in pool:
            mask = (paths == p)
            snippet = ""
            if mask.any():
                local = np.where(mask)[0]
                best_local = local[np.argmax(sims[local])]
                cid = int(ids[best_local])
                row = conn.execute("SELECT text FROM chunks WHERE id=?", (cid,)).fetchone()
                if row:
                    snippet = row[0][:RERANK_SNIPPET_CHARS]
            trow = conn.execute("SELECT title FROM files WHERE path=?", (p,)).fetchone()
            title = trow[0] if trow else p
            candidates.append({"path": p, "title": title, "fused_score": fused_score, "snippet": snippet})

        if RERANK_ENABLED and RR.available() and candidates:
            try:
                # Titolo/nome-file PRIMA dello snippet: e' lo stesso contextual-
                # chunking gia' usato per l'embedding del corpus (rag_index.py),
                # e W_TITLE=0.6 nel fusion RRF conferma che per questo corpus
                # il titolo porta segnale forte da solo. Senza, il reranker
                # giudica lo snippet nudo e perde esattamente quel segnale.
                rerank_texts = [
                    f"{c['title']} — {os.path.basename(c['path'])}\n{c['snippet']}".strip()
                    for c in candidates
                ]
                rerank_scores = RR.score(q, rerank_texts)
                for c, rs in zip(candidates, rerank_scores):
                    c["score"] = round(float(rs), 5)
                candidates.sort(key=lambda c: c["score"], reverse=True)
            except Exception as e:
                # Stesso principio del boot: un fallimento a runtime del
                # reranker (OOM transitorio, input degenere) non deve mai
                # far fallire una ricerca. Ricadi sull'ordine RRF gia' noto.
                print(f"[rerank] skip su questa query: {e!r}", flush=True)
                for c in candidates:
                    c["score"] = round(float(c["fused_score"]), 5)
        else:
            for c in candidates:
                c["score"] = round(float(c["fused_score"]), 5)

        results = [
            {"path": c["path"], "title": c["title"], "score": c["score"], "snippet": c["snippet"]}
            for c in candidates[:k]
        ]
        return results
    finally:
        conn.close()

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a):
        pass
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/health":
            with _lock:
                self._send(200, {"status": "ok", "reranker": bool(RERANK_ENABLED and RR.available()),
                                 "chunks": _state["n"], "files": _state["files"],
                                 "model": EMBED_MODEL_ID, "built_at": _state["built_at"],
                                 "ok": True})
            return
        if u.path == "/search":
            qs = urllib.parse.parse_qs(u.query)
            q = (qs.get("q", [""])[0]).strip()
            if not q:
                self._send(400, {"error": "parametro q mancante"})
                return
            try:
                k = max(1, min(20, int(qs.get("k", ["5"])[0])))
            except ValueError:
                k = 5
            try:
                self._send(200, {"query": q, "results": search(q, k)})
            except Exception as e:
                self._send(500, {"error": repr(e)})
            return
        self._send(404, {"error": "not found"})

def main():
    os.makedirs(DATA, exist_ok=True)
    print("[boot] indicizzazione iniziale...", flush=True)
    _index_once()
    with _lock:
        print(f"[boot] pronto: {_state['n']} chunk, built_at={_state['built_at']}", flush=True)
    threading.Thread(target=_reindex_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[boot] in ascolto su :{PORT}", flush=True)
    srv.serve_forever()

if __name__ == "__main__":
    main()
