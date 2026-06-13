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
  WP's core post `date` field in responses), `genre`, `location`, `language`.
  Term ids are unique across WP taxonomies, so one global id→name map is safe.
- **The store's two-level "MM LOCATION" ("Horror › Stalker Films") is split
  across TWO taxonomies**: `genre` = the COARSE level ("Horror", 75 values in
  use / 78 published), `location` = the FINE shelf ("Stalker Films", 557 in use
  / 572 published). It is NOT a clean tree — 61% of fine shelves map to one
  coarse genre, but cross-cutting shelves ("New Arrival" spans 64 genres, "Staff
  Picks", "Curated") and name collisions break it; the website breadcrumb is a
  per-rental display of one (genre, location) pair, not a global parent. The
  coarse `genre` also mixes in inventory states (`*New Release`, `Library`,
  `Head Cleaner`…) — stage 03 stoplists 11 of them. Stage 07 colors by coarse
  genre (top-15 + greyed "Other"); stage 03 embeds coarse+fine as one deduped
  `Categories:` line in the shelf variant.
- **Director/actor are NOT in the REST API** (they render only in the site's
  htmx search dialogs, backed by a "movie-madness-maven" plugin + `data-mmdb-id`
  attrs). We get cast/crew from TMDB instead; don't bother scraping the dialogs.
- Posters in the catalog are nearly all placeholders; TMDB is the poster source.
- Catalog titles encode inventory in parentheses: `"Fighter, The (1952) (DVD-R)"`,
  `"Totem (2023) (Latin American) (DVD)"`. `pipeline/normalize.py` strips trailing
  parentheticals (year captured; unknown ones kept as `qualifiers`) EXCEPT a bare
  `(N)` volume marker which stays in the title so anime "X (5)"/"X (6)" don't
  merge; un-inverts articles incl. mid-title (`"Conqueror, The: Hollywood Fallout"`),
  and splits season suffixes. It has real tests — `make test` after touching it.

## Decisions & QA (2026-06-13 — colormap/genre rerun)

After a 4-reviewer QA pass (data integrity, TMDB matches, embed-text, viz),
these landed in one rerun:
- **Colormap recolored to coarse genre** (top-15 + greyed "Other"), multi-genre
  films colored by their globally-RAREST genre so "Foreign" (13.5k) doesn't
  swallow everything (`pick_coarse_genre`, stage 07). Coarse genre also added to
  the hover (deduped vs the fine `Shelf:` line) and to the shelf embed text.
- **Year false-split collapse** (stage 01 `canonical_year_map`): a film's
  re-release years (Army of Darkness 1992+1993) merge to one node; remakes
  (Of Mice and Men 1939/1968) stay separate via a ≥3yr gap. ~1,300 films merged.
  Per-SKU year now prefers an in-title `(YYYY)` over the `date` taxonomy.
- **TMDB prefix-match guard** (stage 02): the prefix-extension bonus was matching
  short catalog titles to longer unrelated works ("The Experiment"→"The
  Experimental Film", "Zeitgeist"→"Zeitgeist Stammheim"). Now requires a word
  boundary AND (a subtitle separator `:`/`-` OR `vote_count>2`). ~700 FPs fixed.
- **Digit↔Roman numeral matching** (stage 02 `_to_arabic`): recovers sequels
  ("A Better Tomorrow 2" ↔ "II"). Roman only for multi-char numerals (ii, iii…)
  to avoid clobbering "X"/"I, Robot".
- **Stoplists** (stage 03): 11 inventory pseudo-genres dropped from `genre`;
  store-shorthand qualifiers (`Si`, `G/L`…) dropped from shelf text; synopsis
  run through `unescape + whitespace-collapse`.
- **Rating colormap bug** (stage 07): hand-picked green→red palette was being
  overridden by the glasbey fallback (merge order) — fixed; `N/`→`N/R`.
- **Decisions deferred/declined**: adult content stays fully visible (no NSFW
  filter — user's call); About page + OpenGraph/social meta still deferred (a
  later render-only pass). TMDB backfill for title-only films NOT pursued (low
  yield, risky per QA). (The filter panel, once deferred, was built — see below.)

## Stage 07 interactive layer (post-render, added 2026-06-13)

Everything here is **render-only**: change it and re-run `07_visualize.py`
(~2 min, both variants); no embed/UMAP/Toponymy rerun. `07` renders with
DataMapPlot, then `postprocess_html` (attribution footer + scroll-zoom bump to
`ZOOM_SPEED=0.04`, both regex patches on the minified output) and
`inject_filter_panel` patch the HTML.

- **Advanced Filters panel** is vendored at `pipeline/filter_panel.html` (split
  by `<!-- SECTION: css/html/js -->`), ported from `../steam-atlas` /
  `../huggingface-dataset-map`. We re-themed it by **self-defining its CSS vars**
  (`--brass`/`--cyan`/`--ink*`…) in a `:root` block — the steam-atlas host page
  defined them, our DataMapPlot HTML doesn't. The container needs
  `container-box interactive-element` or the `pointer-events:none` UI layer
  eats clicks. `PARAM_KEY_MAP` + the match-count label were localized.
- **Injection** (`inject_filter_panel`): the panel bootstraps off a
  `datamapReady` CustomEvent we dispatch right after the unique anchor
  `const hoverData = parsedData;` (deferred a tick via setTimeout so
  `addMetaData` + legends are ready). **This anchor is datamapplot-version-
  specific** — if a datamapplot upgrade changes the loader, the `assert` fires
  and you re-find the anchor. Panel HTML is inserted after `#search-container`,
  CSS into `<head>`, JS before `</html>`.
- **Filters read per-point values from dedicated `*_filter` columns in
  `extra_point_data`** (NOT the colormap rawdata; those are separate `_r/_g/_b/_a`
  columns). `build_filter_config` defines them.
- **multiValue gotcha (cost real debugging):** a checkbox filter over data where
  a film has SEVERAL values (format, genre) MUST be multi-value — encode the set
  as a `"|"`-joined string and the panel matches if ANY value is checked
  (`multiValue:true`). Single-bucket coercion (`format_bucket`, rarest-genre) is
  for the COLORMAP only (one color/point); using it for a checkbox hides
  multi-format films (Parenthood on 4K+Blu-ray+DVD vanished under a Blu-ray-only
  filter). `rating` is genuinely single-value, so it stays scalar.
- **Range sliders** (year / runtime / popularity / editions): missing values
  encode as a sentinel strictly **below the slider min** (0 for year & runtime,
  -1 for popularity) so undated/unmatched films show at reset but drop out once
  the slider is touched. Heavy-tailed fields (runtime, popularity) cap the slider
  at the 99th percentile (`capLabel` → "180+"). Years use `plainInt:true` so 1894
  renders "1894" not "2K"/"1,894".
- **Search corpus** (`hover_text`, DataMapPlot substring-matches it):
  title + director + cast + shelf sections + edition qualifiers (Criterion/A24).
  Synopsis deliberately excluded — common words would flood the unranked match.
- **Genre filter lists all 65 coarse genres** (not the colormap's top-15+Other),
  so it's decoupled from the colormap legend; only `format`/`rating` are
  legend-synced via `colormapFieldToFilterId`.

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
