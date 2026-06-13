"""Stage 06: hierarchical region labels via Toponymy + Claude, per variant.

Toponymy names *regions of the embedding space* (place-naming), not individual
films. The 2D UMAP coords are the substrate the named regions sit on
(clusterable_vectors); the film embeddings carry the semantic content used while
clustering (embedding_vectors). Toponymy's own keyphrase embedder is INDEPENDENT
of these vectors (its own search_query space) — model/dim need not match ours.
Films in unnamed space come back "Unlabelled": a gap in the map (signal), not a
labeling failure. cluster_layers_ / topic_name_vectors_ are FINEST-FIRST, which
is also DataMapPlot's *label_layers order, so layers pass through unchanged.

Inputs:  data/embeddings_<variant>.npz, data/umap_coords_<variant>.npz, data/films.parquet
Output:  data/toponymy_labels_<variant>.parquet  (film_id + label_layer_0..k, finest first)
"""

from __future__ import annotations

import os

import nest_asyncio
import numpy as np
import pandas as pd
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MAX_CONCURRENCY,
    ANTHROPIC_MODEL_NAMING,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    FILMS_PARQUET,
    VARIANTS,
    embeddings_npz,
    labels_parquet,
    umap_npz,
)

nest_asyncio.apply()

MAX_DOC_CHARS = 2_000


def label_variant(variant: str) -> None:
    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.embedding_wrappers import CohereEmbedder
    from toponymy.llm_wrappers import AsyncAnthropicNamer

    crd = np.load(umap_npz(variant), allow_pickle=True)
    coords = crd["coords"].astype(np.float32)
    film_id = crd["film_id"]

    ed = np.load(embeddings_npz(variant), allow_pickle=True)
    row = {c: i for i, c in enumerate(ed["film_id"])}
    idx = np.array([row[c] for c in film_id], dtype=np.int64)
    embeddings = ed["emb"][idx].astype(np.float32)

    text_by_id = pd.read_parquet(FILMS_PARQUET, columns=["film_id", f"embed_text_{variant}"]).set_index("film_id")[
        f"embed_text_{variant}"
    ]
    documents = text_by_id.reindex(film_id).fillna("").str.slice(0, MAX_DOC_CHARS).tolist()
    print(f"[{variant}] {len(documents):,} films; embeddings {embeddings.shape}")

    llm = AsyncAnthropicNamer(
        api_key=ANTHROPIC_API_KEY,
        model=ANTHROPIC_MODEL_NAMING,
        max_concurrent_requests=ANTHROPIC_MAX_CONCURRENCY,
    )
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)
    clusterer = ToponymyClusterer(min_clusters=6)

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="movies and TV titles from a video rental catalog",
        corpus_description=(
            "the rental catalog of Movie Madness, a Portland video store with one of the world's "
            "largest physical-media collections; each title is given with its year and plot synopsis"
            + (" and the store shelf section it lives in" if variant == "shelf" else "")
        ),
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    np.random.seed(42)
    topic_model.fit(objects=documents, embedding_vectors=embeddings, clusterable_vectors=coords)

    n_layers = len(topic_model.topic_name_vectors_)
    if n_layers == 0:
        raise ValueError("Toponymy produced 0 cluster layers")
    print(f"[{variant}] Toponymy produced {n_layers} cluster layer(s)")

    out = {"film_id": film_id}
    for i, names in enumerate(topic_model.topic_name_vectors_):
        out[f"label_layer_{i}"] = names

    df = pd.DataFrame(out)
    target = labels_parquet(variant)
    tmp = str(target) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, target)
    print(f"[{variant}] wrote {target} ({n_layers} layers)")


def main():
    for variant in VARIANTS:
        label_variant(variant)


if __name__ == "__main__":
    main()
