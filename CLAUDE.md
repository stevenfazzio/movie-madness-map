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
`ZOOM_SPEED=0.04`, both regex patches on the minified output),
`inject_filter_panel`, and `inject_mobile_support` patch the HTML.

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

### Mobile support — Phases 1 & 2 (`inject_mobile_support`, 2026-06-13/14)

DataMapPlot's default output isn't touch-friendly two ways: (1) no viewport meta,
so phones render into a ~980px layout viewport and shrink the legend/filters
off-screen; (2) the hover tooltip is the only way to see a film's details and it
doesn't exist on touch — a tap instead fires `on_click` (`onClickFunction`, a
top-level deck prop) and navigates straight to moviemadness.org, so the whole
info layer is unreachable on a phone. Phase 1 fixes both; ported/re-themed from
`../semantic-github-map` (light-mode + has a nav bar — we're dark, no nav).

- **What it injects**: the viewport meta; a `@media (max-width:768px)` block
  (heights → plain `100dvh` so the bottom-row stacks clear the URL bar — *no*
  `-44px` nav offset like semantic-github; title scaled 18pt/10pt; colormap
  width-clamp; hide `.deck-tooltip`); and a touch-only **bottom-sheet info card**.
- **DataMapPlot's OWN `@media` is hostile to our injected UI (this was the v1
  regression).** At ≤768px it caps `.top-left` to a `30vh` scrollbox — so
  title+search+the filter panel become a sliver that scrolls as ONE unit — and
  `display:none`s 3 of the 4 stacks, hiding the **colormap selector**
  (`.bottom-left`) and **legend** (`.top-right`). Our block overrides the two that
  matter: `.stack.top-left{max-height:none;overflow-y:visible}` (filter body keeps
  its own `updateFilterBodyHeight` cap, now measured against the visible colormap)
  and `.stack.bottom-left{display:flex}` (colormap back). Legend (`.top-right`)
  stays hidden; Phase 2 mirrors it into a popover (below). **Subtle cause**:
  adding the viewport meta is what *activated*
  datamapplot's mobile `@media` — without it phones render at ~980px and it never
  fires, so the desktop layout (scaled) was what shipped pre-mobile.
- **Touch card** (`isTouchDevice` gate; desktop is untouched): catches the same
  `datamapReady` event the filter panel uses, then `datamap.deckgl.setProps`
  overrides the desktop deck props — `getTooltip:null`, and `onClick`/`onHover`
  show the card instead of navigating. Card fields are read from
  `datamap.metaData[<field>][idx]` (same `extra_point_data` cols as the hover
  template: poster/title/year/byline/synopsis/genre/shelf/formats/rating/cast),
  and the store link moves to a "Find it at the store" button. Card CSS reuses the
  filter panel's dark `:root` vars via `var(--ink-2/--brass/--text…, fallback)`.
- **Gotchas**: uses `onHover` for the tap (deck.gl's `onClick` misses points in
  the upper screen area) — it would pop the card while panning, so Phase 2 gates
  it on an `isDrag` flag (below). A 400ms guard stops the opening tap from self-closing
  the card via the document outside-click handler. `setPointHighlight` clones the
  point layer with `highlightedObjectIndex` so the highlight survives the card
  overlay's `pointerleave`.
- **Phase 2 (added 2026-06-14)**: (1) **mobile legend popover** — a bottom-right
  toggle (`#mobile-legend-popover`, injected before `</body>`) that mirrors the
  active child of the hidden `#legend-container` via a MutationObserver + a
  `datamapReady` sync; hidden when the colormap has no legend (e.g. Clusters),
  shown otherwise (semantic-github's pattern, re-themed dark). Always-run, not
  touch-gated — the `@media` block controls its visibility. (2) **attribution
  overlap** — on mobile the footer becomes a centered, full-width line lifted to
  `bottom:72px` so it clears the colormap selector (both had sat bottom-left).
  (3) **filter touch targets** — fuller-width panel (`100vw-16px`), 20px range
  thumbs (wrapper 28px, track re-centred to 12px), taller checkbox rows. (4)
  **drag-suppression** — pointer-move tracking sets an `isDrag` flag; the card's
  `onHover` ignores hovers once a gesture passes ~10px, so panning no longer pops
  the card (`onClick` is unaffected — deck only fires it on a real tap).
- **Phase 2 follow-up (footer vs. colormap dropdown, same day)**: the repositioned
  footer (`bottom:72px`) sat exactly where the colormap dropdown's options expand
  *upward* when open, so its bottom option ("Runtime") was un-tappable. Two causes:
  the options list was capped at `max-height:50dvh` and scrolled, leaving the
  bottom option clipped behind the `.color-map-selected` header; and even
  un-clipped, the footer covers the options through a **stacking-context trap** —
  the options live in `.content-wrapper` › `.bottom-left` (which has a `transform`,
  = its own context), so their high *local* z-index is still globally below the
  footer (a `body` child). Fix: `max-height:85dvh` (all options fit, no
  scroll/clip) + hide the footer while the dropdown is open (`body.cm-open`,
  toggled by a MutationObserver on `#colorMapOptions` style.display — wired on
  `datamapReady`, because datamapplot builds that element *after* our script runs).
  Also **removed "Genre (TMDB)" from the colormap** (redundant with the store's
  coarse `store_genre`; trims the dropdown to 8 options).
- **Verifying mobile here**: the chrome-extension `resize_window` can't drop the
  viewport below ~1368px, so the `@media` breakpoint won't trigger normally. Flip
  it on at the current width via the CSSOM (`rule.media.mediaText='all'` for every
  `max-width<=768` media rule) to exercise the real cascade, and rely on DOM
  measurements + `elementFromPoint` over screenshots (the dark UI reads as
  near-black on the dark map, and the extension's viewport drifts between calls).
  Final visual check is on a real phone.

### Per-point title labels on zoom (`inject_point_labels`, added 2026-06-14)

Film titles that fade in as you zoom (like `../semantic-github-map`), as a deck.gl
`TextLayer` injected on the `datamapReady` event. semantic-github (10k pts) renders
EVERY label into one TextLayer; at our 82k that's ~1.7M glyphs — too heavy, esp.
mobile. Instead we **viewport-cull + declutter** on each view-state change:

- keep only points inside the current viewport (bounds test against the
  `datamap.pointLayer.props.data.attributes.getPosition.value` buffer);
- greedily place them on a screen-space grid (`CELL_W×CELL_H`), **popularity-first**
  (`metaData.popularity_filter`), capped at `MAX_LABELS` (200);
- rebuild the TextLayer with that subset; **opacity is gated on in-view DENSITY,
  not zoom** — it ramps 0→1 as the in-view point count drops from `CAND_HIGH`
  (600) to `CAND_LOW` (200).

So the layer stays a few hundred labels regardless of the 82k total (~3ms/update
measured, rAF-throttled inside a wrapped `onViewStateChange` that still calls the
original). White SDF text + dark halo, sized in pixels, placed just above each dot
(`getAlignmentBaseline:'bottom'`; offset = `-(dotRadius+2)` recomputed per update —
the dot's screen radius is `getRadius*radiusScale*pixels-per-world-unit`, capped at
`radiusMaxPixels`≈24, so a *fixed* offset can't clear it at high zoom). **Tuning
fights, on-device (2026-06-14):** a zoom-delta gate kept showing labels too early —
`initialViewState.zoom` is baked at render time and the load-time `baseZoom` didn't
match the real overview on large screens, so "+N zoom levels" fired with ~13k
points still in view. Switched to the **density gate** above (screen-size-adaptive,
no base-zoom needed; calibrated — on a 1152px screen, +6 zoom ≈ 584 in view, +7 ≈
176). Separately, the label offset must track the dot radius (above) or labels
overlap the dots once they hit `radiusMaxPixels`. The layer is **not** pickable, so
clicks still hit the dot for the hover/card. Titles are already in `metaData.title`
→ no file-size cost. Inserted right after `dataPointLayer` so the Toponymy region
labels (`labelLayer`) stay on top. Works on all devices (zoom-driven, not
touch-gated). **Filter-aware (2026-06-14):** a label is drawn only for a point
that survives the active filters, read from `datamap.selected` (the per-point
visibility Float32Array `highlightPoints` maintains: `1.0`=shown, `-1.0`=dimmed,
all `1.0` when unfiltered). That one array is the universal signal — the search
box (`searchText`), Advanced Filters (`addSelection`), and colormap-legend clicks
(native `addSelection 'legend'`) all funnel through `highlightPoints` into it, so
no per-mechanism hooks are needed. Two parts: the viewport-cull skips points with
`selected[i] <= 0.5`; and `highlightPoints` is wrapped to re-cull on selection
change (filters/search/legend fire it WITHOUT a view-state change, so
`onViewStateChange` alone would leave stale labels). Because the in-view count
drives the opacity fade, the cull also makes a narrow filter/search surface its
labels early (don't have to zoom into a sparse region first). We wrap
FilterPanel's own `highlightPoints` wrapper (its empty-match fix), so both run.

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
