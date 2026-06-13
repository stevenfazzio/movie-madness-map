"""Central config for the movie-madness-map pipeline. Every stage does `from config
import ...` (the stage's own dir is on sys.path when run as `python pipeline/XX.py`).
Edit constants here for smoke tests rather than adding CLI args."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR = PROJECT_ROOT / "docs"

load_dotenv(PROJECT_ROOT / ".env")
CO_API_KEY = os.environ.get("CO_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

PROJECT_NAME = "Movie Madness Map"
PROJECT_TAGLINE = "Every title on the shelves of Portland's Movie Madness, laid out by what it's about"

# --- Stage 00: fetch the rental catalog (public WordPress REST API) ---
# Movie Madness (moviemadness.org, a nonprofit run by the Hollywood Theatre)
# exposes its full rental catalog as the `rental` custom post type: ~98.8k SKU
# records, plus six taxonomies (format / rating / date / genre / location /
# language). `location` is the store's hand-curated shelf-section system
# (572 terms) — the soul of the dataset. Crawl gently: this is a small
# nonprofit's website. ~990 pages at REQUEST_INTERVAL_S is ~12 min.
MM_BASE_URL = "https://www.moviemadness.org"
MM_API_URL = f"{MM_BASE_URL}/wp-json/wp/v2"
MM_USER_AGENT = "movie-madness-map/0.1 (personal data-visualization project)"
REQUEST_INTERVAL_S = 0.7  # sleep between catalog requests (be a polite guest)
RENTAL_TAXONOMIES = ["format", "rating", "date", "genre", "location", "language"]
MAX_CATALOG_PAGES = None  # smoke-test knob: e.g. 5 fetches only 500 SKUs; None = all
CATALOG_RAW_PARQUET = DATA_DIR / "catalog_raw.parquet"
TAXONOMY_TERMS_PARQUET = DATA_DIR / "taxonomy_terms.parquet"

# --- Stage 01: prepare films (SKU -> film dedupe) ---
# One node = one *film* (normalized title + year), not one rental SKU. The same
# movie on DVD/Blu-ray/VHS collapses to one row carrying a formats list; TV
# seasons stay separate rows (season number is part of the title). Non-film
# formats (e.g. Book) are kept and filterable downstream.
FILMS_BASE_PARQUET = DATA_DIR / "films_base.parquet"

# --- Stage 02: TMDB enrichment (optional but recommended) ---
# Matches each film against The Movie Database by normalized title + year for
# synopsis gap-fill (~31% of catalog SKUs have no synopsis), director/cast,
# poster URLs, and extra colormap axes. ~2 calls per film; throttled. Skipped
# formats (books etc.) and unmatched films flow through with empty TMDB fields.
TMDB_API_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"
TMDB_REQUEST_INTERVAL_S = 0.08  # ~12 req/s; TMDB tolerates ~50/s but be modest
TMDB_SKIP_FORMATS = {"Book"}  # formats that should never be matched against TMDB
TMDB_CHECKPOINT_EVERY = 200  # films between checkpoint writes
TMDB_PARQUET = DATA_DIR / "tmdb.parquet"

# --- Stage 03: build the embed/display corpus ---
# Two embed-text variants, so "what shapes the layout" can be decided by
# comparing finished maps (see CLAUDE.md):
#   synopsis — title + year + synopsis only: the map organizes purely by what
#              the films are about; shelf sections become a colormap overlay.
#   shelf    — additionally includes the store's location/section path, so the
#              hand-curated shelving partially shapes the neighborhoods.
# Stages 04-07 loop over VARIANTS; trim to one entry to halve cost/time.
VARIANTS = ["synopsis", "shelf"]
# The variant published as the GitHub Pages root (docs/index.html). Decided
# 2026-06-12 in favor of "shelf": its national-cinema / director regions
# (French Cinema, Japanese Action, Nordic Indie, Hong Kong) preserve the store's
# curatorial personality. The other variant ships at docs/<variant>.html.
PRIMARY_VARIANT = "shelf"
FILMS_PARQUET = DATA_DIR / "films.parquet"
# Smoke-test knob: cap to a deterministic random subset of films for a fast,
# cheap end-to-end dry run of stages 04-07. None = the full corpus.
MAX_FILMS = None
SUBSET_SEED = 42

# --- Stage 04: embed films (Cohere embed-v4.0) ---
# input_type="clustering" because the only downstream use is grouping /
# visualization (UMAP + Toponymy's clusterer). Toponymy's INTERNAL keyphrase
# embedder lives in its own (search_query) space and is never compared against
# these vectors, so our input_type and output_dimension are free choices.
COHERE_EMBED_MODEL = "embed-v4.0"
COHERE_INPUT_TYPE = "clustering"
COHERE_OUTPUT_DIM = 1024  # Matryoshka dim; 256/512/1024/1536 allowed
EMBED_BATCH = 96  # Cohere embed max texts per call
EMBED_CHECKPOINT_EVERY = 50  # batches between progress checkpoints


def embeddings_npz(variant: str) -> Path:
    return DATA_DIR / f"embeddings_{variant}.npz"


# --- Stage 05: UMAP layout ---
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_RANDOM_STATE = 42


def umap_npz(variant: str) -> Path:
    return DATA_DIR / f"umap_coords_{variant}.npz"


# --- Stage 06: Toponymy region labels (the costliest stage at full scale) ---
ANTHROPIC_MODEL_NAMING = "claude-haiku-4-5-20251001"
ANTHROPIC_MAX_CONCURRENCY = 24


def labels_parquet(variant: str) -> Path:
    return DATA_DIR / f"toponymy_labels_{variant}.parquet"


# --- Stage 07: DataMapPlot visualization ---
def map_html(variant: str) -> Path:
    return DATA_DIR / f"movie_map_{variant}.html"


def docs_html(variant: str) -> Path:
    # The primary variant is the Pages root (index.html); the other keeps its name.
    return DOCS_DIR / ("index.html" if variant == PRIMARY_VARIANT else f"{variant}.html")
