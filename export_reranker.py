#!/usr/bin/env python3
"""Export the cross-encoder to ONNX and quantize it to int8.

Run this once, before `docker compose up`, to produce the ~118 MB
`reranker/model.int8.onnx` that stage 4 needs. The weights are deliberately
not committed -- a git repository is the wrong place for a 118 MB binary.

    pip install torch transformers onnx onnxruntime optimum
    python export_reranker.py

The service runs fine without this: `/health` reports `"reranker": false` and
retrieval degrades to pure RRF + title-boost.
"""
import os
import shutil
import subprocess
import sys

MODEL_ID = os.environ.get(
    "RERANK_MODEL_ID", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
OUT_DIR = os.environ.get("RERANK_MODEL_DIR", "reranker")
FP32_DIR = os.path.join(OUT_DIR, "_fp32")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. fp32 ONNX export via optimum, which knows the sequence-classification
    #    head's input signature so we don't hand-roll dummy inputs.
    print(f"[1/3] exporting {MODEL_ID} to ONNX")
    subprocess.run([
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", MODEL_ID,
        "--task", "text-classification",
        FP32_DIR,
    ], check=True)

    # 2. dynamic int8 quantization: 471 MB -> ~118 MB, and on CPU it is also
    #    faster. Dynamic (not static) because it needs no calibration set.
    print("[2/3] quantizing to int8")
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=os.path.join(FP32_DIR, "model.onnx"),
        model_output=os.path.join(OUT_DIR, "model.int8.onnx"),
        weight_type=QuantType.QInt8,
    )

    # 3. the tokenizer files rag_rerank.py loads at boot.
    print("[3/3] copying tokenizer")
    for name in ("tokenizer.json", "config.json", "special_tokens_map.json",
                 "tokenizer_config.json"):
        src = os.path.join(FP32_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUT_DIR, name))

    shutil.rmtree(FP32_DIR, ignore_errors=True)

    size_mb = os.path.getsize(os.path.join(OUT_DIR, "model.int8.onnx")) / 1e6
    print(f"\ndone: {OUT_DIR}/model.int8.onnx ({size_mb:.0f} MB)")
    print("mount it at /app/reranker and restart; /health should report "
          '"reranker": true')


if __name__ == "__main__":
    main()
