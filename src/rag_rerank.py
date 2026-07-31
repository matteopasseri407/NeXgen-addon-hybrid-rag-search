"""Cross-encoder reranker (ONNX int8, CPU-only) per il fusion RRF di rag_server.

Non tocca l'indice: prende gli STESSI candidati e snippet gia' recuperati
dalla ricerca ibrida (vettore + BM25 + title-boost) e li riordina secondo
un giudizio di rilevanza query-passaggio piu' fine di quanto un embedding
statico (model2vec) possa dare da solo. Nessun re-embed del corpus, nessun
reindex: e' uno stadio aggiuntivo a query-time, spegnibile con una env var.

Modello: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingue, addestrato
su mMARCO che include l'italiano), esportato in ONNX e quantizzato dynamic
int8 -- 118MB su disco, ~95-170ms per un batch di 20 coppie su ARM Ampere
(misurato dal vivo sulla VPS di produzione, non stimato).
"""
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = os.environ.get("RERANK_MODEL_DIR", "/app/reranker")
MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "256"))

_session = None
_tokenizer = None
_has_token_type = False


def load():
    """Carica modello e tokenizer. Solleva se i file mancano o sono corrotti --
    il chiamante decide se e' fatale o degrada a reranking disattivato."""
    global _session, _tokenizer, _has_token_type
    model_path = os.path.join(MODEL_DIR, "model.int8.onnx")
    tok_path = os.path.join(MODEL_DIR, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    # Default SessionOptions leaves ORT's own heuristic pick the thread count,
    # which measured well under the actual core count on the ARM Ampere VPS
    # (roughly half the throughput of an explicit intra_op_num_threads=cpu_count).
    # Verified live on the production VPS, not assumed.
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, os.cpu_count() or 1)
    so.inter_op_num_threads = 1
    session = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    has_token_type = "token_type_ids" in {i.name for i in session.get_inputs()}
    _tokenizer, _session, _has_token_type = tokenizer, session, has_token_type
    return session


def available():
    return _session is not None


def score(query, passages):
    """Un punteggio di rilevanza in [0,1] (sigmoide del logit) per ciascun
    passaggio, stesso ordine di `passages`. Piu' alto = piu' rilevante.
    Non e' una probabilita' calibrata: e' un segnale ordinale, come lo
    era gia' lo score RRF che sostituisce."""
    if not passages:
        return np.array([], dtype=np.float32)
    enc = _tokenizer.encode_batch([(query, p) for p in passages])
    feed = {
        "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
        "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
    }
    if _has_token_type:
        feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
    logits = _session.run(None, feed)[0].reshape(-1).astype(np.float32)
    return 1.0 / (1.0 + np.exp(-logits))
