"""Stage 04: embed each film with Cohere embed-v4.0, once per embed-text variant.

Input per film is embed_text_<variant> from stage 03. input_type="clustering"
because the only downstream use is grouping/visualization (UMAP + Toponymy's
clusterer). One float32 vector per film. Checkpointed + resumable per variant.

Input:  data/films.parquet
Output: data/embeddings_<variant>.npz  (emb [N x dim] float32, film_id [N] object)
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from config import (
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    COHERE_INPUT_TYPE,
    COHERE_OUTPUT_DIM,
    EMBED_BATCH,
    EMBED_CHECKPOINT_EVERY,
    FILMS_PARQUET,
    VARIANTS,
    embeddings_npz,
)


def embed_with_retry(client, chunk, max_retries=5):
    for attempt in range(max_retries):
        try:
            resp = client.embed(
                model=COHERE_EMBED_MODEL,
                input_type=COHERE_INPUT_TYPE,
                texts=chunk,
                output_dimension=COHERE_OUTPUT_DIM,
                embedding_types=["float"],
            )
            return np.asarray(resp.embeddings.float_, dtype=np.float32)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt * 5, 60)
            print(f"  embed attempt {attempt + 1} failed ({type(e).__name__}: {e}); retry in {wait}s")
            time.sleep(wait)


def embed_variant(client, df: pd.DataFrame, variant: str) -> None:
    out_npz = embeddings_npz(variant)
    film_ids = df["film_id"].to_numpy()
    texts = df[f"embed_text_{variant}"].tolist()
    n = len(texts)
    print(f"[{variant}] embedding {n:,} films with {COHERE_EMBED_MODEL} (dim={COHERE_OUTPUT_DIM})")

    sig = f"{film_ids[0]}_{film_ids[-1]}_{n}_{COHERE_EMBED_MODEL}_{COHERE_OUTPUT_DIM}_{variant}"
    if out_npz.exists():
        cached = np.load(out_npz, allow_pickle=True)
        if str(cached["sig"]) == sig:
            print(f"  [{variant}] reusing cached embeddings ({n:,})")
            return

    emb = np.zeros((n, COHERE_OUTPUT_DIM), dtype=np.float32)
    done = 0
    prog_path = str(out_npz) + ".progress.npz"
    if os.path.exists(prog_path):
        p = np.load(prog_path, allow_pickle=True)
        if int(p["n"]) == n and int(p["dim"]) == COHERE_OUTPUT_DIM:
            emb = p["emb"]
            done = int(p["done"])
            print(f"  [{variant}] resuming from {done:,}/{n:,}")

    batch_i = 0
    for start in range(done, n, EMBED_BATCH):
        chunk = texts[start : start + EMBED_BATCH]
        emb[start : start + len(chunk)] = embed_with_retry(client, chunk)
        batch_i += 1
        if batch_i % EMBED_CHECKPOINT_EVERY == 0:
            np.savez(prog_path + ".tmp.npz", n=n, dim=COHERE_OUTPUT_DIM, done=start + len(chunk), emb=emb)
            os.replace(prog_path + ".tmp.npz", prog_path)
            print(f"  [{variant}] embedded {start + len(chunk):,}/{n:,}")

    tmp = str(out_npz) + ".tmp.npz"
    np.savez(tmp, sig=sig, emb=emb, film_id=film_ids)
    os.replace(tmp, out_npz)
    if os.path.exists(prog_path):
        os.unlink(prog_path)
    print(f"  [{variant}] wrote {out_npz} ({out_npz.stat().st_size / 1e6:.1f} MB)")


def main():
    import cohere

    if not CO_API_KEY:
        raise RuntimeError("CO_API_KEY not set; add it to .env (see .env.example)")
    client = cohere.ClientV2(api_key=CO_API_KEY)

    df = pd.read_parquet(FILMS_PARQUET, columns=["film_id"] + [f"embed_text_{v}" for v in VARIANTS])
    for variant in VARIANTS:
        embed_variant(client, df, variant)


if __name__ == "__main__":
    main()
