#!/usr/bin/env python3
"""
Indicizzazione incrementale di un corpus Markdown per la ricerca semantica.

Storage (in /data, volume persistente):
  - index.db : sqlite
      files(path PK, title, file_hash, mtime)
      chunks(id PK, path, ord, text, vec BLOB)   -- vec = embedding float32 del chunk
      fts(text)                                   -- FTS5, rowid = chunks.id  (BM25)
      meta(key, value)                            -- embed_model, dim, built_at
  - i vettori restano nel DB (BLOB per chunk): cosi' l'embedding e' VERAMENTE
    incrementale (si ri-embeddano solo i chunk dei file nuovi/cambiati).

Esclusioni: VCS/editor metadata sempre, piu' quelle configurate via env
  (EXCLUDE_PARTS / EXCLUDE_PREFIX). Escludi la tua directory di segreti e
  qualunque output generato che duplichi contenuto canonico sotto un secondo
  path -- un duplicato stantio confonde un giudice per-contenuto come il
  reranker, che lo legge come altrettanto rilevante.
"""
import os, re, glob, time, json, hashlib, sqlite3
import numpy as np

def _env_tuple(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return tuple(p.strip() for p in raw.split(",") if p.strip())

# Componenti di path esclusi a qualunque livello (es. la tua dir di segreti).
EXCLUDE_PARTS = _env_tuple("EXCLUDE_PARTS", (".git", ".obsidian"))
# sanitize/build/: generated pre-split engine snapshot, duplicate content of
# real canonical notes under a different path. Indexing it means every query
# that matches a canonical note ALSO matches its stale twin here, splitting
# score/rank between the two and confusing anything that judges by text
# content (a cross-encoder reranker especially, since the duplicate reads as
# equally relevant). Found while validating the reranker (2026-07-13); the
# duplication itself was already flagged as dead weight in the 2026-07-11
# architecture review, independent of the reranker.
# Prefissi di path esclusi (relativi alla radice del corpus).
EXCLUDE_PREFIX = _env_tuple("EXCLUDE_PREFIX", ())

CHUNK_WORDS = 250
OVERLAP_WORDS = 50
EMBED_BATCH = 64


def prefixes(model):
    m = (model or "").lower()
    if "e5" in m:
        return ("query: ", "passage: ")
    if "nomic" in m:
        return ("search_query: ", "search_document: ")
    return ("", "")


def rel(vault, path):
    return os.path.relpath(path, vault).replace("\\", "/")


def iter_notes(vault):
    for p in glob.glob(os.path.join(vault, "**", "*.md"), recursive=True):
        r = rel(vault, p)
        parts = r.split("/")
        if any(part in EXCLUDE_PARTS for part in parts):
            continue
        if any(r.startswith(pre) for pre in EXCLUDE_PREFIX):
            continue
        yield r, p


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def get_title(txt, fallback):
    m = re.search(r"^#\s+(.+)$", txt, re.MULTILINE)
    return (m.group(1).strip() if m else fallback)


def chunk_text(txt):
    paras = re.split(r"\n\s*\n", txt)
    chunks, cur, cur_w = [], [], 0
    for para in paras:
        w = para.split()
        if not w:
            continue
        if len(w) > CHUNK_WORDS:
            i = 0
            while i < len(w):
                chunks.append(" ".join(w[i:i + CHUNK_WORDS]))
                i += CHUNK_WORDS - OVERLAP_WORDS
            continue
        if cur_w + len(w) > CHUNK_WORDS and cur:
            chunks.append(" ".join(cur))
            tail = " ".join(cur).split()[-OVERLAP_WORDS:]
            cur, cur_w = list(tail), len(tail)
        cur.extend(w)
        cur_w += len(w)
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, title TEXT, file_hash TEXT, mtime REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT, ord INTEGER, text TEXT, vec BLOB)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(text, tokenize='unicode61 remove_diacritics 2')")
    conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    return conn


def _meta_get(conn, key, default=None):
    r = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return r[0] if r else default


def _meta_set(conn, key, value):
    conn.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))


def _delete_path(conn, path):
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE path=?", (path,)).fetchall()]
    for cid in ids:
        conn.execute("DELETE FROM fts WHERE rowid=?", (cid,))
    conn.execute("DELETE FROM chunks WHERE path=?", (path,))
    conn.execute("DELETE FROM files WHERE path=?", (path,))


def incremental_index(conn, vault, embed_docs, model_name, log=print):
    """embed_docs(list[str]) -> np.ndarray[float32] (gia' col prefisso 'passage/document').
    Ritorna dict di statistiche."""
    t0 = time.time()
    stats = {"added": 0, "updated": 0, "deleted": 0, "embedded_chunks": 0, "model": model_name}

    # cambio modello => l'indice e' invalido: ricostruzione totale
    prev_model = _meta_get(conn, "embed_model")
    if prev_model and prev_model != model_name:
        log(f"[index] cambio modello {prev_model} -> {model_name}: ricostruzione totale")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM fts")
        conn.execute("DELETE FROM files")
        conn.commit()

    on_disk = {}
    for r, p in iter_notes(vault):
        on_disk[r] = p

    db_files = {row[0]: (row[1], row[2]) for row in conn.execute("SELECT path, file_hash, mtime FROM files").fetchall()}

    # cancellati
    for path in list(db_files.keys()):
        if path not in on_disk:
            _delete_path(conn, path)
            stats["deleted"] += 1

    # nuovi / cambiati
    for path, abspath in on_disk.items():
        try:
            h = file_hash(abspath)
        except OSError:
            continue
        prev = db_files.get(path)
        if prev and prev[0] == h:
            continue  # invariato
        try:
            with open(abspath, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
        except OSError:
            continue
        _delete_path(conn, path)
        if not txt.strip():
            continue
        title = get_title(txt, os.path.basename(path))
        ctx = title + " " + os.path.basename(path)   # contesto da prependere (contextual chunking)
        mtime = os.path.getmtime(abspath)
        conn.execute("INSERT INTO files(path,title,file_hash,mtime) VALUES(?,?,?,?)", (path, title, h, mtime))
        for ordn, ch in enumerate(chunk_text(txt)):
            cur = conn.execute("INSERT INTO chunks(path,ord,text,vec) VALUES(?,?,?,NULL)", (path, ordn, ch))
            # BM25 sul testo contestualizzato; chunks.text resta grezzo per snippet puliti
            conn.execute("INSERT INTO fts(rowid,text) VALUES(?,?)", (cur.lastrowid, ctx + " " + ch))
        if prev:
            stats["updated"] += 1
        else:
            stats["added"] += 1
    conn.commit()

    # embedding dei soli chunk senza vettore (testo contestualizzato: titolo + nome-file + chunk)
    todo = conn.execute(
        "SELECT c.id, f.title, c.path, c.text FROM chunks c JOIN files f ON f.path=c.path "
        "WHERE c.vec IS NULL ORDER BY c.id").fetchall()
    for i in range(0, len(todo), EMBED_BATCH):
        batch = todo[i:i + EMBED_BATCH]
        texts = [(ttl + " " + os.path.basename(pth) + "\n" + txt2) for (_id, ttl, pth, txt2) in batch]
        vecs = embed_docs(texts).astype(np.float32)
        for (cid, _t, _p, _x), v in zip(batch, vecs):
            conn.execute("UPDATE chunks SET vec=? WHERE id=?", (v.tobytes(), cid))
        conn.commit()
        stats["embedded_chunks"] += len(batch)

    dim = None
    row = conn.execute("SELECT vec FROM chunks WHERE vec IS NOT NULL LIMIT 1").fetchone()
    if row:
        dim = len(np.frombuffer(row[0], dtype=np.float32))
    _meta_set(conn, "embed_model", model_name)
    if dim:
        _meta_set(conn, "dim", dim)
    _meta_set(conn, "built_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    conn.commit()

    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["total_chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    stats["total_files"] = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    log(f"[index] {stats}")
    return stats


def load_matrix(conn):
    """Carica (ids, paths, M_normalizzata) per la ricerca brute-force."""
    rows = conn.execute("SELECT id, path, vec FROM chunks WHERE vec IS NOT NULL ORDER BY id").fetchall()
    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=object), np.zeros((0, 1), dtype=np.float32)
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    paths = np.array([r[1] for r in rows], dtype=object)
    M = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows]).astype(np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    M = M / norms
    return ids, paths, M
