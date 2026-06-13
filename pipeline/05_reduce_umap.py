"""Stage 05: reduce each variant's embeddings to 2D with UMAP for the map layout.

Same params as the sibling map projects (cosine, n_neighbors=15, min_dist=0.05)
for shape consistency. random_state fixed for reproducibility.

Input:  data/embeddings_<variant>.npz
Output: data/umap_coords_<variant>.npz  (coords [N x 2] float32, aligned film_id)
"""

from __future__ import annotations

import os

import numpy as np
import umap
from config import UMAP_MIN_DIST, UMAP_N_NEIGHBORS, UMAP_RANDOM_STATE, VARIANTS, embeddings_npz, umap_npz


def main():
    for variant in VARIANTS:
        d = np.load(embeddings_npz(variant), allow_pickle=True)
        emb = d["emb"].astype(np.float32)
        film_id = d["film_id"]
        print(f"[{variant}] loaded {emb.shape[0]:,} embeddings x {emb.shape[1]}")

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=UMAP_RANDOM_STATE,
            verbose=True,
        )
        coords = reducer.fit_transform(emb).astype(np.float32)

        out = umap_npz(variant)
        tmp = str(out) + ".tmp.npz"
        np.savez(tmp, coords=coords, film_id=film_id)
        os.replace(tmp, out)
        print(f"[{variant}] wrote {out}")


if __name__ == "__main__":
    main()
