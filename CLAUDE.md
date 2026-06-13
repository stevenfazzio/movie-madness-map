# CLAUDE.md

Agent notes for this repo. Read `README.md` first for the project overview and
pipeline list; this file adds conventions, decisions, and gotchas.

## What this is

An interactive 2D semantic map of the Movie Madness rental catalog (~98.8k SKUs
→ ~one-dot-per-film), sibling of `../jeopardy-map` (the lean spine this is
ported from), `../taskmaster-map`, `../steam-atlas`, `../huggingface-dataset-map`.
Spine: fetch → dedupe → TMDB-enrich → corpus → embed (Cohere) → UMAP →
Toponymy (Claude naming) → DataMapPlot → GitHub Pages (`docs/`).

## Decisions (made with Steven, 2026-06-12)

- **One dot per film, not per SKU.** Same title+year across DVD/Blu-ray/VHS
  collapses to one row (`formats` list survives; "VHS only" is a signal worth
  filtering on). TV seasons stay separate (season is part of the title).
- **TMDB enrichment: yes** — synopsis gap-fill (~31% of SKUs have no catalog
  synopsis), director/cast, posters, genres. Catalog synopsis wins when present.
  No LLM-generated synopses ever: the obscure tail is exactly where hallucinated
  plots would go undetected.
- **Two embed-text variants, decided empirically later**: `synopsis` (pure
  "what films are about") vs `shelf` (store sections in the embed text shape
  neighborhoods). `config.VARIANTS` drives stages 04–07; artifacts are suffixed
  `_<variant>`. Trim the list to one to halve cost.
- **No LLM-extracted per-film fields in v1** (no facets/taglines à la
  steam-atlas). LLM use is confined to Toponymy region naming.
- **Be a polite guest**: moviemadness.org is a small nonprofit's WP Engine site.
  Catalog crawl is throttled (`REQUEST_INTERVAL_S`), resumable, honest UA; don't
  hammer it for iteration — `data/catalog_raw.parquet` is the cache. The map
  carries an attribution footer (Movie Madness + TMDB-not-endorsed lines).

## Data source map (learned by probing, 2026-06)

- Catalog = WordPress custom post type `rental` at
  `https://www.moviemadness.org/wp-json/wp/v2/rental` (~98,834 SKUs; standard WP
  pagination, `per_page=100`, `orderby=id&order=asc` for stable paging;
  `X-WP-Total` header carries the count; robots.txt is fully permissive).
- Six taxonomies on it: `format`, `rating`, `date` (year-as-term — it SHADOWS
  WP's core post `date` field in responses), `genre` (~mirrors top-level
  section), `location` (the famous hand-curated shelf sections, 572 terms),
  `language`. Term ids are unique across WP taxonomies, so one global id→name
  map is safe.
- **Director/actor are NOT in the REST API** (they render only in the site's
  htmx search dialogs, backed by a "movie-madness-maven" plugin + `data-mmdb-id`
  attrs). We get cast/crew from TMDB instead; don't bother scraping the dialogs.
- Posters in the catalog are nearly all placeholders; TMDB is the poster source.
- Catalog titles encode inventory in parentheses: `"Fighter, The (1952) (DVD-R)"`,
  `"Totem (2023) (Latin American) (DVD)"`. `pipeline/normalize.py` strips ALL
  trailing parentheticals (year captured; unknown ones kept as `qualifiers`),
  un-inverts articles incl. mid-title (`"Conqueror, The: Hollywood Fallout"`),
  and splits season suffixes. It has real tests — `make test` after touching it.

## Conventions

- `film_id` is the alignment key across every stage — it's the content-derived
  string `"<norm title>|<year>"` (stable across catalog refetches, so the TMDB
  checkpoint cache keyed on it survives). Stages 04+ carry it through npz/parquet
  and stage 07 left-merges everything back on it, asserting row counts.
- All knobs live in `pipeline/config.py` (no CLI args). Smoke tests: set
  `MAX_CATALOG_PAGES` (stage 00) or `MAX_FILMS` (stage 03 subset, flows through
  04–07). Per-variant paths come from config helper functions.
- Atomic writes everywhere (tmp + verify + `os.replace`); stages 00/02/04 are
  checkpointed + resumable. Treat `data/*.parquet` / `*.npz` as expensive.
- Plain `.py` scripts, `uv`, `ruff` (line-length 120, E/F/I). Keys via env or
  `.env` (CO_API_KEY, ANTHROPIC_API_KEY, TMDB_API_KEY); CO/ANTHROPIC are already
  exported in Steven's shell.

## Gotchas learned in this repo

- **Stage 02's fail-soft `except` records `error:<Type>` rows; the resume path
  DROPS those so they retry, and a >30% error-rate circuit breaker aborts the
  run.** Both exist because a column-name typo once burned 2,200 rows as
  `error:KeyError` while the run kept cheerfully going. If you change the
  films_base schema, re-check every `row[...]` access in stage 02.
- **TV vs movie matching**: `looks_tv` requires a Season/Series suffix or a TV
  shelf section. A bare "Part 2"/"Vol. 3" is NOT television — feature films use
  those (e.g. "Friday The 13th Part 2" must not match the 1987 *Friday the 13th*
  TV series). TV searches use `title_base` (compound season suffixes are peeled
  repeatedly), movies use the display `title`, and TV scoring is year-lenient
  (catalog year is often a mid-run season's year, not first-air).
- **Match-scorer heuristics** (all year-gated, tested): digit↔word number
  spelling ("100 Men…" ↔ "One Hundred Men…"), prefix-extension with an 8-char
  guard ("12 Rounds 2" ↔ "12 Rounds 2: Reloaded"), original-title comparison
  for foreign films. Threshold `ACCEPT_THRESHOLD=0.93`; sub-threshold best
  scores are recorded on `no_match` rows for audit.
- Observed timings: catalog crawl ~24 min at 0.7 s politeness interval;
  TMDB ~5 films/s (~2 req/film) → ~4.7 h full pass. Film-level catalog synopsis
  coverage is 54% (the 69% SKU-level sample flattered it).
- The catalog has typo years ("1000", "7079"); stage 01 range-guards to
  1888–2027 and falls back through the date-taxonomy candidates.

## Gotchas inherited from siblings (still apply here)

- `transformers` is a required dep even though no HF model runs (Toponymy
  imports it at module level); the "PyTorch not found" import notice is fine.
- `fast-hdbscan==0.2.2` pin: toponymy 0.5.0 breaks against 0.3.x.
- Toponymy is place-naming, not topic-modeling: `Unlabelled` points are gaps in
  the map (signal), and layer 0 is FINEST — pass through to DataMapPlot unchanged.
- Cohere `input_type="clustering"`; Toponymy's internal keyphrase embedder is a
  separate space — never "fix" a perceived mismatch (see jeopardy-map CLAUDE.md).
- UMAP `random_state=42` for reproducibility (single-threaded is fine here).
- Preview HTML over `python3 -m http.server --bind 127.0.0.1`, never `file://`
  (Chrome extension rewrites it).
