"""Stage 07: render the interactive DataMapPlot, one HTML per variant.

Layout: one point per film, positioned by stage 05's UMAP of that variant's
embeddings. Hover shows poster (when TMDB matched), title/year, director, the
synopsis, coarse genre, the store shelf section, formats, and rating; clicking a
point opens the film's search page on moviemadness.org ("go rent it"). A colormap
dropdown covers year, coarse genre, format, rating, editions, and TMDB
popularity/runtime. Toponymy region names float on top.

Post-render patches (see postprocess_html / inject_filter_panel /
inject_mobile_support / inject_point_labels): attribution footer, faster
scroll-zoom, the composable "Advanced Filters" panel (format / decade / genre /
rating / editions) vendored from filter_panel.html, a mobile layer (viewport meta
+ touch info card + legend popover), and per-point film-title labels that fade in
on zoom (viewport-culled + decluttered, so they scale to ~82k points).

Inputs:  data/umap_coords_<variant>.npz, data/films.parquet, pipeline/filter_panel.html,
         [optional] data/toponymy_labels_<variant>.parquet
Output:  data/movie_map_<variant>.html  (+ copied to docs/<variant>.html)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from html import escape
from pathlib import Path

import datamapplot
import glasbey
import numpy as np
import pandas as pd
from config import (
    FILMS_PARQUET,
    PROJECT_NAME,
    PROJECT_TAGLINE,
    VARIANTS,
    docs_html,
    labels_parquet,
    map_html,
    umap_npz,
)
from matplotlib.colors import to_hex

FILTER_PANEL_TEMPLATE = Path(__file__).resolve().parent / "filter_panel.html"

VARIANT_BLURB = {
    "synopsis": "layout: what the films are about",
    "shelf": "layout: synopsis + the store's shelf sections",
}

FORMAT_COLORS = {
    "VHS": "#c0392b",
    "4K UHD": "#8e44ad",
    "Blu-Ray": "#2471a3",
    "DVD": "#7f8c8d",
    "Other": "#b7950b",
}

RATING_COLORS = {
    "G": "#27ae60",
    "PG": "#82e0aa",
    "PG-13": "#f4d03f",
    "R": "#e67e22",
    "NC-17": "#cb4335",
    "X": "#78281f",
    "N/R": "#95a5a6",
}

ATTRIBUTION_HTML = """
<div id="mm-attribution" style="position:fixed;bottom:8px;left:10px;z-index:10;font-size:11px;
     font-family:Roboto,sans-serif;opacity:.75;pointer-events:auto;max-width:60vw">
  Catalog: <a href="https://www.moviemadness.org/" target="_blank" rel="noopener">Movie Madness</a>,
  a nonprofit video store in Portland, OR (Hollywood Theatre) &nbsp;&middot;&nbsp;
  Film metadata &amp; posters: <a href="https://www.themoviedb.org/" target="_blank" rel="noopener">TMDB</a>
  (this product uses the TMDB API but is not endorsed or certified by TMDB) &nbsp;&middot;&nbsp;
  Click a film to find it at the store
</div>
"""


# DataMapPlot hardcodes the deck.gl scroll-zoom at speed 0.01 (datamap.js),
# which feels sluggish; bump it. 0.04 = 4x the default — snappy but still smooth.
ZOOM_SPEED = 0.04


def postprocess_html(html: str) -> str:
    """Apply our post-render patches to a DataMapPlot HTML string: attribution
    footer + faster scroll-zoom. Shared so already-rendered files can be patched
    in place without a full re-render."""
    html, n = re.subn(r"</body>", ATTRIBUTION_HTML + "</body>", html, count=1)
    assert n == 1, "no </body> found for attribution injection"
    # Minified controller config: scrollZoom:{speed:0.01,smooth:true}
    html, n = re.subn(r"(scrollZoom:\{speed:)[0-9.]+", rf"\g<1>{ZOOM_SPEED}", html, count=1)
    assert n == 1, "scroll-zoom speed token not found (datamapplot internals changed?)"
    return html


def _alpha_other_last(values) -> list[str]:
    out = sorted(set(map(str, values)))
    if "Other" in out:
        out.remove("Other")
        out.append("Other")
    return out


def _range_filter(name, field, label, lo, hi, slider_max, cap_label, plain_int=False) -> dict:
    return {
        "type": "range",
        "name": name,
        "filterId": f"filter-{name}",
        "label": label,
        "field": field,
        "min": int(lo),
        "max": int(hi),
        "sliderMax": int(slider_max),
        "step": 1,
        "compact": True,
        "capLabel": cap_label,
        "plainInt": plain_int,
    }


def build_filter_config(df: pd.DataFrame, format_cats, genre_cats, rating_cat) -> dict:
    """Assemble the JSON consumed by FilterPanel in filter_panel.html. Each
    filter's `field` is an extra_point_data column the panel reads per-point.
    Checkbox: format / genre (all 65) / rating. format and genre are MULTI-VALUE
    (a film matches if ANY of its values is checked). Range: year / runtime /
    popularity / editions. `colormapFieldToFilterId` syncs a filter to its
    colormap legend (genre is excluded — its filter is finer than the colormap)."""
    fmt_order = ["4K UHD", "Blu-Ray", "DVD", "VHS", "Other"]
    present_fmt = {c for cats in format_cats for c in cats}
    format_values = [f for f in fmt_order if f in present_fmt]

    genre_values = _alpha_other_last({g for cats in genre_cats for g in cats})

    rating_order = ["G", "PG", "PG-13", "R", "NC-17", "X", "N/R"]
    present_rt = set(rating_cat)
    rating_values = [r for r in rating_order if r in present_rt] + sorted(present_rt - set(rating_order))

    year = df["year"].dropna().astype(int)
    runtime = pd.to_numeric(df["runtime"], errors="coerce")
    runtime = runtime[runtime > 0]
    pop = pd.to_numeric(df["popularity"], errors="coerce")
    pop = pop[pop > 0]
    editions = df["sku_count"].astype(int)

    sections = [
        {
            "label": "Categories",
            "filters": [
                {
                    "name": "format",
                    "filterId": "filter-format",
                    "label": "Format",
                    "field": "format_filter",
                    "values": format_values,
                    "multiValue": True,
                },
                {
                    "name": "genre",
                    "filterId": "filter-genre",
                    "label": "Genre",
                    "field": "genre_filter",
                    "values": genre_values,
                    "multiValue": True,
                },
                {
                    "name": "rating",
                    "filterId": "filter-rating",
                    "label": "MPAA Rating",
                    "field": "rating_filter",
                    "values": rating_values,
                },
            ],
        },
        {
            "label": "Ranges",
            "filters": [
                # Year: full linear range (no heavy tail to cap); plain int label
                # so 1894 doesn't render as "2K" or "1,894".
                _range_filter(
                    "year",
                    "year_filter",
                    "Release year",
                    year.min(),
                    year.max(),
                    year.max(),
                    cap_label=False,
                    plain_int=True,
                ),
                # Runtime & popularity are heavy-tailed -> cap the slider at the
                # 99th percentile; the max handle at cap means "include above".
                _range_filter(
                    "runtime",
                    "runtime_filter",
                    "Runtime (min)",
                    1,
                    runtime.max(),
                    np.percentile(runtime, 99),
                    cap_label=True,
                ),
                _range_filter(
                    "popularity",
                    "popularity_filter",
                    "TMDB popularity",
                    0,
                    pop.max(),
                    np.percentile(pop, 99),
                    cap_label=True,
                ),
                _range_filter(
                    "editions",
                    "editions_filter",
                    "Editions in stock",
                    editions.min(),
                    editions.max(),
                    editions.max(),
                    cap_label=False,
                ),
            ],
        },
    ]
    colormap_to_filter = {"format": "filter-format", "rating": "filter-rating"}
    return {
        "totalCount": int(len(df)),
        "sections": sections,
        "colormapFieldToFilterId": colormap_to_filter,
        "filterIdToColormapField": {v: k for k, v in colormap_to_filter.items()},
    }


def inject_filter_panel(html: str, config: dict) -> str:
    """Inject the Advanced Filters panel: dispatch a `datamapReady` event once
    datamap+hoverData are live, then patch the panel's CSS/HTML/JS into the page.
    Ported from steam-atlas / huggingface-dataset-map (see CLAUDE.md)."""
    # 1. Bootstrap: fire datamapReady after metadata is attached. `const hoverData
    #    = parsedData;` is the unique point where both globals are in scope; defer
    #    a tick so datamap.addMetaData (and its colormap legends) finish first.
    anchor = "const hoverData = parsedData;"
    dispatch = (
        anchor + "\n      setTimeout(function(){ try { window.dispatchEvent(new CustomEvent("
        "'datamapReady', { detail: { datamap: datamap, hoverData: hoverData } })); } "
        "catch(e){ console.error('datamapReady dispatch failed', e); } }, 0);"
    )
    assert html.count(anchor) == 1, "filter injection: hoverData anchor not found/unique"
    html = html.replace(anchor, dispatch, 1)

    template = FILTER_PANEL_TEMPLATE.read_text()
    parts = re.split(r"<!-- SECTION: (\w+) -->", template)
    section = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}

    html = html.replace("</head>", section["css"] + "\n</head>", 1)

    search_re = re.compile(r'(<div id="search-container" class="container-box[^"]*">\s*<input[^>]*/>\s*</div>)')
    m = search_re.search(html)
    assert m, "filter injection: search-container div not found"
    html = html[: m.end()] + "\n      " + section["html"] + "\n" + html[m.end() :]

    js = section["js"].replace("__FILTER_CONFIG_JSON__", json.dumps(config))
    html = html.replace("</html>", js + "\n</html>", 1)
    return html


# ── Mobile support (Phase 1: viewport + touch info card) ──────────────────────
# DataMapPlot's default output isn't touch-friendly in two ways. (1) No viewport
# meta tag, so phones render into a ~980px layout viewport and shrink everything
# (the colormap/legend end up off-screen or microscopic). (2) The only way to see
# a film's details is the hover tooltip — which doesn't exist on touch, where a
# tap instead fires on_click and navigates straight to moviemadness.org, so the
# whole info layer (poster, synopsis, ...) is unreachable on a phone.
#
# Fix: inject the viewport meta + a max-width:768px stylesheet (dvh heights so
# the bottom stacks clear the URL bar, scaled title, width clamps), and on touch
# devices intercept the tap to show a bottom-sheet "info card" with the same
# fields as the hover tooltip plus an explicit "find it at the store" button.
# Ported/re-themed from ../semantic-github-map (which is light-mode and has a top
# nav bar — we're dark-mode with no nav, so the card reuses the filter panel's
# dark :root vars and heights are a plain 100dvh). Render-only (see CLAUDE.md).
# Phase 2 (mobile legend popover, filter-panel drawer, drag-suppression) is TODO.

VIEWPORT_META = '<meta name="viewport" content="width=device-width, initial-scale=1">'

MOBILE_CSS = """<style>
@media (max-width: 768px) {
  #title-container span:first-child { font-size: 18pt !important; line-height: 1.1 !important; }
  #title-container span:last-of-type { font-size: 10pt !important; line-height: 1.2 !important; }
  #title-container { margin: 4px 8px !important; padding: 8px 10px !important; }
  .container-box { margin: 3px 8px !important; padding: 8px 10px !important; }
  /* Mobile 100vh bug: vh counts the area behind the URL bar, pushing the bottom
     row of stacks (colormap, legend) under the browser chrome. dvh tracks the
     visible viewport. No nav bar here, so it's a plain 100dvh (semantic-github
     subtracts its 44px nav). */
  body { height: 100dvh !important; }
  #deck-container { height: 100dvh !important; }
  .content-wrapper { height: 100dvh !important; min-height: 100dvh !important; }
  /* DataMapPlot's OWN @media(max-width:768px) breaks our injected UI two ways:
     it caps `.top-left` to `max-height:30vh; overflow-y:auto` — cramming
     title+search+the filter panel into a sliver that scrolls as ONE unit — and
     `display:none`s `.top-right`/`.bottom-left`/`.bottom-right`, which hides our
     colormap selector (it lives in `.bottom-left`). Undo the two that hurt: let
     `.top-left` grow so the filter body keeps its own scroll cap (set by
     updateFilterBodyHeight, measured against the colormap), and bring the
     colormap back. The legend (`.top-right`) stays hidden — Phase 2 popover. */
  .stack.top-left { max-height: none !important; overflow-y: visible !important; }
  .stack.bottom-left { display: flex !important; }
  #colormap-selector-container { max-width: calc(100vw - 24px); }
  /* Roomy enough for all options so the upward-opening dropdown doesn't scroll
     and clip its bottom option (which then renders behind the .color-map-selected
     header — the header is z100, options z101/z102, so an UN-clipped option wins,
     but a scrolled-out one isn't painted). 50dvh was too short. */
  .color-map-options { max-height: 85dvh !important; }
  /* The deck.gl hover tooltip is replaced by the bottom-sheet card on touch. */
  .deck-tooltip { display: none !important; }
  /* Phase 2: show the mobile legend popover (the desktop legend in .top-right
     stays hidden — we mirror its content into the popover). */
  #mobile-legend-popover { display: block !important; }
  /* Phase 2: fuller-width filter panel + finger-sized controls. The whole
     checkbox row is a <label>, so taller padding alone widens the tap target;
     the 14px box is left as-is (its :checked checkmark is positioned for 14px). */
  #filter-container { max-width: calc(100vw - 16px) !important; }
  #filter-container.open { width: calc(100vw - 16px) !important; }
  .filter-range-wrapper { height: 28px !important; }
  .filter-range-track, .filter-range-fill { top: 12px !important; }
  .filter-range-wrapper input[type="range"] { height: 28px !important; }
  .filter-range-wrapper input[type="range"]::-webkit-slider-thumb { width: 20px !important; height: 20px !important; }
  .filter-range-wrapper input[type="range"]::-moz-range-thumb { width: 20px !important; height: 20px !important; }
  .filter-checkbox-item { padding: 6px 2px !important; }
  /* Phase 2: the attribution and the colormap selector both sit bottom-left and
     overlapped on short viewports. Lift the attribution into a centered,
     full-width line just above the colormap (which stays at the very bottom). */
  #mm-attribution {
    left: 0 !important; right: 0 !important; bottom: 72px !important;
    max-width: none !important; text-align: center !important;
    font-size: 9px !important; line-height: 1.35 !important; padding: 0 10px !important;
  }
  /* The colormap dropdown opens upward, over where the footer now sits; hide the
     footer while it's open (body.cm-open, toggled by JS) so all options stay
     tappable — otherwise the footer covers the bottom option(s). */
  body.cm-open #mm-attribution { display: none !important; }
}

/* Touch info card — dark bottom sheet. Reuses the filter panel's :root palette;
   the var() fallbacks keep it self-contained if that CSS ever isn't present. */
#mobile-info-card {
  display: none;
  position: fixed;
  bottom: 0; left: 0; right: 0;
  z-index: 300;
  background: var(--ink-2, #17171b);
  border-top: 1px solid var(--rule, #34343c);
  border-radius: 14px 14px 0 0;
  box-shadow: 0 -6px 28px rgba(0, 0, 0, 0.55);
  padding: 16px 18px calc(env(safe-area-inset-bottom, 8px) + 14px);
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  color: var(--text, #f1f1f3);
  pointer-events: auto;
  max-height: 62dvh;
  overflow-y: auto;
  -webkit-overflow-scrolling: touch;
}
#mobile-info-card.visible { display: block; }
#mobile-info-card .mic-close {
  position: absolute; top: 6px; right: 10px;
  background: none; border: none;
  font-size: 26px; line-height: 1;
  color: var(--text-faint, #8b8b93);
  cursor: pointer; padding: 4px 10px;
}
#mobile-info-card .mic-visit {
  display: inline-block; margin-top: 12px;
  padding: 10px 18px;
  background: var(--brass, #e0b34e);
  color: #1a1407;
  font-weight: 600; font-size: 14px;
  text-decoration: none; border-radius: 8px;
}
#mobile-info-card .mic-visit:active { background: #c99a30; }

/* Phase 2: mobile legend popover — mirrors the desktop #legend-container, which
   datamapplot hides on mobile (it's in .top-right). Hidden on desktop; the
   @media block shows it. Fixed bottom-right, opposite the colormap selector. */
#mobile-legend-popover {
  display: none;
  position: fixed;
  right: 10px;
  bottom: calc(env(safe-area-inset-bottom, 0px) + 10px);
  z-index: 150;
  pointer-events: auto;
}
#mobile-legend-toggle {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 7px 12px;
  border: 1px solid var(--rule, #34343c);
  border-radius: 9px;
  background: var(--ink-2, #17171b);
  color: var(--text-dim, #c6c6cc);
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
  font-size: 12px; font-weight: 500;
  cursor: pointer; pointer-events: auto;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
}
.mobile-legend-content {
  display: none;
  position: absolute;
  bottom: calc(100% + 8px); right: 0;
  background: var(--ink-2, #17171b);
  border: 1px solid var(--rule, #34343c);
  border-radius: 11px;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.6);
  padding: 10px 12px;
  max-height: 60dvh; overflow-y: auto;
  -webkit-overflow-scrolling: touch;
  min-width: 150px;
  color: var(--text, #f1f1f3);
}
/* Re-apply sizing to the copied legend content — datamapplot's own legend CSS is
   scoped under #legend-container, so the innerHTML copy loses it. */
.mobile-legend-content .legend-label { font-size: 11px !important; }
.mobile-legend-content .color-swatch-box { width: 11px !important; height: 11px !important; }
.mobile-legend-content .colorbar-container { height: 140px !important; padding: 6px !important; }
.mobile-legend-content .colorbar { height: 140px !important; }
.mobile-legend-content .colorbar-tick-container { height: 140px !important; }
.mobile-legend-content .colorbar-tick-label { font-size: 10px !important; }
</style>"""

MOBILE_INFOCARD_HTML = """<div id="mobile-info-card">
  <button class="mic-close" type="button" aria-label="Close">×</button>
  <div class="mic-body"></div>
</div>"""

MOBILE_LEGEND_HTML = """<div id="mobile-legend-popover">
  <div class="mobile-legend-content"></div>
  <button id="mobile-legend-toggle" type="button">Legend +</button>
</div>"""

MOBILE_JS = """<script>
(function() {
  // Touch only — desktop keeps the hover tooltip and click-to-store behavior.
  var isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
  if (!isTouchDevice) return;

  var card = document.getElementById('mobile-info-card');
  if (!card) return;
  var cardBody = card.querySelector('.mic-body');
  var closeBtn = card.querySelector('.mic-close');
  var lastOpen = 0;
  var clearHighlight = function() {};

  function hideCard() {
    if (!card.classList.contains('visible')) return;
    card.classList.remove('visible');
    clearHighlight();
  }
  closeBtn.addEventListener('click', function(e) { e.stopPropagation(); hideCard(); });

  // Dismiss when tapping non-canvas UI (legend, filters, chrome). The 400ms
  // guard stops the very tap that opened the card from also closing it (the
  // same gesture bubbles a DOM click to document right after).
  document.addEventListener('click', function(e) {
    if (card.contains(e.target)) return;
    if (Date.now() - lastOpen < 400) return;
    hideCard();
  });

  // Drag-suppression (Phase 2): onHover fires continuously during a pan, which
  // would pop the card for every point the finger crosses. Track movement since
  // pointerdown and treat anything past ~10px as a pan. onClick is unaffected —
  // deck.gl only fires it on a genuine tap, not a drag.
  var dragStart = null, isDrag = false;
  window.addEventListener('pointerdown', function(e) {
    dragStart = { x: e.clientX, y: e.clientY };
    isDrag = false;
  }, true);
  window.addEventListener('pointermove', function(e) {
    if (!dragStart) return;
    var dx = e.clientX - dragStart.x, dy = e.clientY - dragStart.y;
    if (dx * dx + dy * dy > 100) isDrag = true;
  }, true);
  window.addEventListener('pointerup', function() { dragStart = null; }, true);

  window.addEventListener('datamapReady', function(e) {
    var datamap = e.detail && e.detail.datamap;
    var hoverData = e.detail && e.detail.hoverData;
    if (!datamap || !hoverData) return;

    clearHighlight = function() { setPointHighlight(-1); };

    // Persist the point highlight ourselves: when the card slides up over the
    // touch point it triggers a pointerleave on the canvas, which would clear
    // deck.gl's autoHighlight.
    function setPointHighlight(idx) {
      var layer = datamap.pointLayer;
      if (!layer || !datamap.layers || !datamap.deckgl) return;
      var layerIdx = datamap.layers.indexOf(layer);
      if (layerIdx === -1) return;
      var updated = layer.clone({ highlightedObjectIndex: idx });
      datamap.layers[layerIdx] = updated;
      datamap.pointLayer = updated;
      datamap.deckgl.setProps({ layers: datamap.layers.slice() });
    }

    function field(name, idx) {
      return hoverData[name] ? hoverData[name][idx] : '';
    }

    function showMobileCard(idx) {
      // poster / genre / cast are pre-built HTML fragments (may be empty); the
      // rest are escaped text. Mirrors the desktop hover template.
      var html = '<div style="overflow:hidden">'
        + field('poster', idx)
        + '<div style="font-weight:700;font-size:15px;line-height:1.3">'
        + field('title', idx)
        + ' <span style="font-weight:400;opacity:.7">(' + field('year', idx) + ')</span></div>';
      var byline = field('byline', idx);
      if (byline) html += '<div style="margin-top:3px;font-size:11px;opacity:.8">' + byline + '</div>';
      var synopsis = field('synopsis', idx);
      if (synopsis) html += '<div style="margin-top:6px;font-size:12.5px;line-height:1.45">' + synopsis + '</div>';
      html += field('genre', idx)
        + '<div style="margin-top:5px;font-size:11px;color:#c79a3a">Shelf: ' + field('shelf', idx) + '</div>'
        + '<div style="margin-top:3px;font-size:11px;opacity:.7">'
        + field('formats', idx) + ' &nbsp;·&nbsp; ' + field('rating', idx) + '</div>'
        + field('cast', idx)
        + '</div>';
      var url = field('mm_url', idx);
      if (url) {
        html += '<a class="mic-visit" href="' + url + '" target="_blank" rel="noopener">'
          + 'Find it at the store →</a>';
      }
      cardBody.innerHTML = html;
      card.scrollTop = 0;
      card.classList.add('visible');
      lastOpen = Date.now();
      setPointHighlight(idx);
    }

    // Override the deck props set for desktop:
    //  - getTooltip: drop the hover tooltip (the card replaces it).
    //  - onClick: was window.open(mm_url) — now show the card, or dismiss on an
    //    empty-space tap; the store link lives on the card's button instead.
    //  - onHover: on touch this fires reliably on tap across the whole canvas,
    //    where onClick alone often misses points in the upper area (deck.gl
    //    synthetic-click timing). It also fires during a pan, so it's gated on
    //    the isDrag flag (above) to avoid popping the card mid-pan.
    datamap.deckgl.setProps({
      getTooltip: null,
      onClick: function(info) {
        if (info && info.picked) showMobileCard(info.index);
        else hideCard();
      },
      onHover: function(info) {
        if (isDrag) return;
        if (info && info.picked) showMobileCard(info.index);
      }
    });
  });
})();
</script>
<script>
// Mobile legend popover (Phase 2): mirror the desktop #legend-container — which
// datamapplot hides on mobile — into a bottom-right toggle. Not touch-gated; the
// @media block controls visibility, so it works on any narrow viewport.
(function() {
  var popover = document.getElementById('mobile-legend-popover');
  if (!popover) return;
  var toggle = document.getElementById('mobile-legend-toggle');
  var content = popover.querySelector('.mobile-legend-content');
  var open = false;

  toggle.addEventListener('click', function(e) {
    e.stopPropagation();
    open = !open;
    content.style.display = open ? 'block' : 'none';
    toggle.innerHTML = open ? 'Legend −' : 'Legend +';
  });
  document.addEventListener('click', function(e) {
    if (open && !popover.contains(e.target)) {
      open = false;
      content.style.display = 'none';
      toggle.innerHTML = 'Legend +';
    }
  });

  // datamapplot keeps one child per colormap in #legend-container and shows only
  // the active one (inline display); mirror that child. Hide the toggle if the
  // active colormap has no legend.
  function sync() {
    var src = document.getElementById('legend-container');
    if (!src) return;
    var vis = null;
    for (var i = 0; i < src.children.length; i++) {
      if (src.children[i].style.display !== 'none') { vis = src.children[i]; break; }
    }
    if (vis && vis.innerHTML.trim()) {
      content.innerHTML = vis.innerHTML;
      toggle.style.removeProperty('display');
    } else {
      content.innerHTML = '';
      open = false;
      content.style.display = 'none';
      toggle.innerHTML = 'Legend +';
      toggle.style.setProperty('display', 'none', 'important');
    }
  }
  var src = document.getElementById('legend-container');
  if (src) {
    new MutationObserver(sync).observe(src, {
      childList: true, subtree: true, attributes: true, attributeFilter: ['style']
    });
  }
  window.addEventListener('datamapReady', function() { setTimeout(sync, 300); });
  setTimeout(sync, 1000);  // fallback if datamapReady already fired

  // Toggle body.cm-open while the colormap dropdown is open, so the @media rule
  // hides the footer. (The footer sits in front of the dropdown's options via a
  // stacking-context trap — content-wrapper is nested and the footer is a body
  // child — so hiding it is the reliable fix.) datamapplot builds #colorMapOptions
  // AFTER this script runs (and may rebuild it), so wire the observer on
  // datamapReady + a fallback, keyed by a per-element flag, not at parse time.
  function setupCmOpen() {
    var o = document.getElementById('colorMapOptions');
    if (!o || o.__cmObserved) return;
    o.__cmObserved = true;
    var t = function() { document.body.classList.toggle('cm-open', o.style.display !== 'none'); };
    new MutationObserver(t).observe(o, { attributes: true, attributeFilter: ['style'] });
    t();
  }
  window.addEventListener('datamapReady', function() { setTimeout(setupCmOpen, 350); });
  setTimeout(setupCmOpen, 1100);
  setupCmOpen();
})();
</script>"""


def inject_mobile_support(html: str) -> str:
    """Mobile patches (Phase 1 + 2): viewport meta + a max-width:768px stylesheet
    + a touch-only bottom-sheet info card (tap a point -> card with the hover
    fields and a 'find it at the store' button, instead of navigating away) + a
    mobile legend popover (mirrors the desktop legend datamapplot hides on
    mobile). The card's onHover is drag-suppressed so panning doesn't pop it.
    String-in/string-out like postprocess_html, so it can also patch an
    already-rendered file. Desktop is untouched (the card JS early-returns off
    touch; the legend popover is hidden by CSS above 768px). Run after
    inject_filter_panel — it relies on that function's datamapReady event and on
    the attribution patch having re-added </body>."""
    assert html.count("</head>") == 1, "mobile: expected exactly one </head>"
    html = html.replace("</head>", VIEWPORT_META + "\n" + MOBILE_CSS + "\n</head>", 1)
    assert html.count("</body>") == 1, "mobile: expected exactly one </body>"
    html = html.replace("</body>", MOBILE_INFOCARD_HTML + "\n" + MOBILE_LEGEND_HTML + "\n</body>", 1)
    assert html.count("</html>") == 1, "mobile: expected exactly one </html>"
    html = html.replace("</html>", MOBILE_JS + "\n</html>", 1)
    return html


# ── Per-point title labels (fade in on zoom) ──────────────────────────────────
# DataMapPlot shows only the Toponymy *region* names, not per-film titles. This
# adds film titles that fade in once you zoom in, like ../semantic-github-map —
# but that map (10k pts) renders ALL labels into one TextLayer; at our 82k that's
# ~1.7M glyphs (heavy, esp. mobile). Instead we viewport-CULL: each view-state
# change, keep only points in the current viewport that survive the active
# filters (datamap.selected), then greedily DECLUTTER on a screen grid (popular
# titles win slots) and draw at most MAX_LABELS. So the layer stays a few hundred
# labels regardless of total (≈3ms/update, measured). White SDF text with a dark
# halo, placed just above each dot; not pickable, so clicks still hit the dot.
# Titles are already in metaData -> no file-size cost. Filter-aware: the search
# box, Advanced Filters, and colormap-legend clicks all gate the labels (the
# in-view count also drives the fade, so a narrow filter surfaces labels early).
POINT_LABELS_JS = """<script>
(function() {
  window.addEventListener('datamapReady', function(e) {
    var datamap = e.detail.datamap;
    if (!datamap || !datamap.pointLayer || !datamap.deckgl || typeof deck === 'undefined') return;
    var meta = datamap.metaData || {};
    var titles = meta.title;
    if (!titles) return;
    var pl = datamap.pointLayer;
    var positions = pl.props.data.attributes.getPosition.value;
    var pop = meta.popularity_filter || null;   // popular titles win the declutter
    var n = titles.length;
    // Filter-awareness: datamap.selected is the per-point visibility array that
    // highlightPoints maintains — 1.0 = shown, -1.0 = dimmed/filtered out, and
    // all 1.0 when nothing is filtered. Every selection path writes it (Advanced
    // Filters via addSelection, the search box via searchText, colormap-legend
    // clicks via addSelection 'legend'), so reading it makes the labels respect
    // all three controls at once, regardless of which one the user touched.
    var selected = datamap.selected || null;
    // Point radius config, so a label can sit just ABOVE its dot. The dot's screen
    // radius grows with zoom (world radius * pixels-per-world-unit) and caps at
    // radiusMaxPixels (~24px); a fixed offset can't clear it, so we compute it.
    var R_MAX = pl.props.radiusMaxPixels || 24, R_MIN = pl.props.radiusMinPixels || 0;
    var R_SCALE = pl.props.radiusScale || 1;
    var R_BASE = typeof pl.props.getRadius === 'number' ? pl.props.getRadius : 0.05;

    var charSet = new Set();
    for (var i = 0; i < n; i++) {
      var t = titles[i]; if (!t) continue;
      for (var j = 0; j < t.length; j++) charSet.add(t[j]);
    }
    var characterSet = Array.from(charSet);

    // Gate on in-view DENSITY, not a zoom number: labels fade in as the count of
    // points inside the viewport drops from CAND_HIGH to CAND_LOW. A zoom-delta
    // gate kept misfiring on large screens (the baked initialViewState.zoom and
    // the load-time zoom didn't match the real overview), and density is what we
    // actually want — "show labels once the view is focused enough to read them."
    // Adapts to screen size for free (a bigger viewport holds more points, so it
    // gates labels until you zoom further in).
    var CAND_HIGH = 600, CAND_LOW = 200;   // >HIGH in view: hidden; <LOW: full opacity
    var MAX_LABELS = 200;             // hard cap on labels drawn at once
    var CELL_W = 96, CELL_H = 18;     // declutter grid cell (screen px)

    function viewport() { var v = datamap.deckgl.getViewports(); return v && v[0]; }

    function inViewCandidates(vp) {
      var c1 = vp.unproject([0, 0]), c2 = vp.unproject([vp.width, vp.height]);
      var minX = Math.min(c1[0], c2[0]), maxX = Math.max(c1[0], c2[0]);
      var minY = Math.min(c1[1], c2[1]), maxY = Math.max(c1[1], c2[1]);
      var cand = [];
      for (var i = 0; i < n; i++) {
        if (selected && selected[i] <= 0.5) continue;   // filtered / searched out
        var x = positions[i * 2], y = positions[i * 2 + 1];
        if (x < minX || x > maxX || y < minY || y > maxY || !titles[i]) continue;
        cand.push(i);
      }
      return cand;
    }

    function declutter(vp, cand) {
      if (pop) cand.sort(function(a, b) { return pop[b] - pop[a]; });
      var occ = {}, out = [];
      for (var k = 0; k < cand.length && out.length < MAX_LABELS; k++) {
        var idx = cand[k];
        var sp = vp.project([positions[idx * 2], positions[idx * 2 + 1]]);
        var key = ((sp[0] / CELL_W) | 0) + ',' + ((sp[1] / CELL_H) | 0);
        if (occ[key]) continue;
        occ[key] = 1;
        out.push({ text: titles[idx], position: [positions[idx * 2], positions[idx * 2 + 1]] });
      }
      return out;
    }

    function buildLayer(data, opacity, offY) {
      return new deck.TextLayer({
        id: 'pointLabelLayer', data: data, opacity: opacity,
        getText: function(d) { return d.text; },
        getPosition: function(d) { return d.position; },
        getSize: 12, sizeUnits: 'pixels',
        getColor: [245, 245, 248],
        getTextAnchor: 'middle', getAlignmentBaseline: 'bottom', getPixelOffset: [0, offY],
        fontFamily: 'IBM Plex Sans, system-ui, sans-serif', fontWeight: 500,
        characterSet: characterSet,
        fontSettings: { sdf: true, radius: 12, buffer: 4 },
        outlineWidth: 2.5, outlineColor: [8, 8, 11, 255],
        background: false, parameters: { depthTest: false }
      });
    }

    function update() {
      var vp = viewport(); if (!vp) return;
      var cand = inViewCandidates(vp);
      var opacity = Math.max(0, Math.min(1, (CAND_HIGH - cand.length) / (CAND_HIGH - CAND_LOW)));
      // dot screen radius -> place the label just above it
      var ppwu = Math.abs(vp.project([1, 0])[0] - vp.project([0, 0])[0]);
      var dotR = Math.min(R_MAX, Math.max(R_MIN, R_BASE * R_SCALE * ppwu));
      var layer = buildLayer(opacity > 0 ? declutter(vp, cand) : [], opacity, -(dotR + 2));
      var i = datamap.layers.findIndex(function(l) { return l.id === 'pointLabelLayer'; });
      var dp = datamap.layers.findIndex(function(l) { return l.id === 'dataPointLayer'; });
      if (i !== -1) datamap.layers[i] = layer;
      else if (dp !== -1) datamap.layers.splice(dp + 1, 0, layer);
      else datamap.layers.push(layer);
      datamap.deckgl.setProps({ layers: [].concat(datamap.layers) });
    }

    var pending = false;
    function scheduleUpdate() {
      if (pending) return;
      pending = true;
      requestAnimationFrame(function() { pending = false; update(); });
    }

    update();
    var orig = datamap.deckgl.props.onViewStateChange || null;
    datamap.deckgl.setProps({
      onViewStateChange: function(params) {
        var r = orig ? orig(params) : undefined;
        scheduleUpdate();
        return r;
      }
    });
    // Re-cull when the SELECTION changes, not just the view. A filter toggle, a
    // search keystroke, or a legend click fires highlightPoints WITHOUT a
    // view-state change, so onViewStateChange alone would leave stale labels on
    // now-hidden points until the next pan/zoom. highlightPoints is the single
    // chokepoint every selection path goes through; FilterPanel has already
    // wrapped it (its empty-match fix), and we wrap that wrapper so both run.
    var prevHighlight = datamap.highlightPoints;
    if (typeof prevHighlight === 'function') {
      datamap.highlightPoints = function(itemId) {
        prevHighlight.call(datamap, itemId);
        scheduleUpdate();
      };
    }
  });
})();
</script>"""


def inject_point_labels(html: str) -> str:
    """Inject the per-point title labels (POINT_LABELS_JS). Runs after
    inject_filter_panel — it listens on that function's datamapReady event."""
    assert html.count("</html>") == 1, "point labels: expected exactly one </html>"
    return html.replace("</html>", POINT_LABELS_JS + "\n</html>", 1)


def categorical_color_mapping(values, default=None, default_color="#c9ccd1"):
    uniques = sorted(set(map(str, values)))
    others = [v for v in uniques if v != default]
    palette = glasbey.create_palette(palette_size=max(len(others), 1))
    mapping = {v: to_hex(palette[i]) for i, v in enumerate(others)}
    if default is not None and default in uniques:
        mapping[default] = default_color
    return mapping


def _fill_nonfinite(a):
    a = np.asarray(a, dtype=float)
    finite = a[np.isfinite(a)]
    fill = np.median(finite) if finite.size else 0.0  # all-NaN (e.g. tiny smoke subset) -> 0
    return np.where(np.isfinite(a), a, fill)


def pick_coarse_genre(genre_lists: pd.Series) -> pd.Series:
    """One coarse genre per film for the colormap. Films carry 0-4 genres (99.4%
    have 1); when >1, pick the GLOBALLY RAREST so a film isn't swallowed by the
    huge 'Foreign' bucket (a Foreign+Horror film colors as Horror). Empty -> Other."""
    counts = Counter(g for gs in genre_lists for g in (gs if isinstance(gs, (list, np.ndarray)) else []))

    def pick(gs):
        gs = list(gs) if isinstance(gs, (list, np.ndarray)) else []
        return min(gs, key=lambda g: (counts[g], g)) if gs else "Other"

    return genre_lists.map(pick)


def format_bucket(formats: list) -> str:
    """Single bucket per film — for the COLORMAP (one color per point). Picks the
    BEST available format on a 4K > Blu-Ray > DVD > VHS ladder; the colormap is
    labeled "Format (best available)" to match."""
    fset = set(formats)
    if "4K UHD" in fset:
        return "4K UHD"
    if any("Blu-Ray" in f for f in fset):
        return "Blu-Ray"
    if any(f.startswith("DVD") for f in fset):
        return "DVD"
    if "VHS" in fset:
        return "VHS"
    return "Other"


_FORMAT_COVERED = {"4K UHD", "Blu-Ray", "Blu-Ray 3D", "DVD", "DVD-R", "VHS"}


def format_categories(formats: list) -> list[str]:
    """The SET of format buckets a film qualifies for — for the multi-select
    FILTER, so a film on 4K+Blu-ray+DVD+VHS matches any of those checkboxes (not
    just the colormap's priority bucket). Every format is a plain contains-match."""
    fset = set(formats)
    cats = []
    if "4K UHD" in fset:
        cats.append("4K UHD")
    if any("Blu-Ray" in f for f in fset):
        cats.append("Blu-Ray")
    if any(f.startswith("DVD") for f in fset):
        cats.append("DVD")
    if "VHS" in fset:
        cats.append("VHS")
    if not cats or (fset - _FORMAT_COVERED):  # Book/Laserdisc/etc., or nothing matched
        cats.append("Other")
    return cats


def bucket_top_n(values: pd.Series, n: int, other="Other") -> pd.Series:
    top = values.value_counts().nlargest(n).index
    return values.where(values.isin(top), other)


def render_variant(variant: str, films: pd.DataFrame) -> None:
    crd = np.load(umap_npz(variant), allow_pickle=True)
    coords = crd["coords"].astype(np.float32)
    layout = pd.DataFrame({"film_id": crd["film_id"], "x": coords[:, 0], "y": coords[:, 1]})

    df = layout.merge(films, on="film_id", how="left")
    assert len(df) == len(layout), "join changed row count — film_id not unique"
    print(f"[{variant}] map points: {len(df):,}")
    coords_xy = df[["x", "y"]].to_numpy()

    label_layers = []
    lp = labels_parquet(variant)
    if lp.exists():
        topo = pd.read_parquet(lp)
        merged = df[["film_id"]].merge(topo, on="film_id", how="left")
        layer_cols = sorted(
            (c for c in topo.columns if c.startswith("label_layer_")),
            key=lambda s: int(s.rsplit("_", 1)[1]),
        )
        label_layers = [merged[c].fillna("Unlabelled").to_numpy() for c in layer_cols]
        print(f"  [{variant}] using {len(label_layers)} Toponymy label layer(s)")
    else:
        print(f"  [{variant}] no Toponymy labels; rendering without region names")

    # --- hover fields (raw HTML is interpolated into the template; pre-escape) ---
    year_str = df["year"].map(lambda y: str(int(y)) if pd.notna(y) else "?")
    poster_html = [
        f'<img src="{p}" style="width:92px;float:right;margin:0 0 6px 10px;border-radius:3px" loading="lazy">'
        if p
        else ""
        for p in df["poster_url"].fillna("")
    ]
    synopsis = df["synopsis"].fillna("").map(lambda s: escape(s if len(s) <= 320 else s[:317] + "…"))
    byline = [
        " · ".join(
            x
            for x in (
                escape(d) if d else "",
                f"{int(r)} min" if pd.notna(r) and r else "",
            )
            if x
        )
        for d, r in zip(df["director"].fillna(""), df["runtime"])
    ]
    shelf_str = df["sections"].map(lambda s: escape("; ".join(s)) if len(s) else "—")
    cast_html = [
        f'<div style="margin-top:4px;font-size:11px;opacity:.65">{escape(c)}</div>' if c else ""
        for c in df["cast_str"].fillna("")
    ]
    # Coarse-genre hover line, deduped against the fine shelf sections (drop a
    # coarse token already shown as a section, e.g. "Comedy"); omit if nothing left.
    genre_html = []
    for genres, sections in zip(df["genres_coarse"], df["sections"]):
        sec_lower = {s.lower() for s in sections}
        coarse = [g for g in (genres if isinstance(genres, (list, np.ndarray)) else []) if g.lower() not in sec_lower]
        genre_html.append(
            f'<div style="margin-top:6px;font-size:11px;color:#7a9a6a">Genre: {escape("; ".join(coarse))}</div>'
            if coarse
            else ""
        )
    rating_clean = df["rating"].fillna("N/R").replace({"": "N/R", "N/": "N/R", "Unrated": "N/R"})

    extra = pd.DataFrame(
        {
            "poster": poster_html,
            "title": [escape(t) for t in df["title"].fillna("")],
            "year": year_str.to_numpy(),
            "byline": byline,
            "synopsis": synopsis.to_numpy(),
            "shelf": shelf_str.to_numpy(),
            "formats": [escape(f) for f in df["formats_str"].fillna("")],
            "rating": [escape(r) for r in rating_clean],
            "mm_url": df["mm_url"].fillna("").to_numpy(),
        }
    )

    hover_template = (
        '<div style="max-width:360px;overflow:hidden">'
        "{poster}"
        '<div style="font-weight:700;font-size:14px;line-height:1.3">{title} '
        '<span style="font-weight:400;opacity:.7">({year})</span></div>'
        '<div style="margin-top:3px;font-size:11px;opacity:.8">{byline}</div>'
        '<div style="margin-top:5px;font-size:12px;line-height:1.4">{synopsis}</div>'
        "{genre}"
        '<div style="margin-top:4px;font-size:11px;color:#9a6a00">Shelf: {shelf}</div>'
        '<div style="margin-top:3px;font-size:11px;opacity:.7">{formats} &nbsp;·&nbsp; {rating}</div>'
        "{cast}"
        "</div>"
    )
    extra["cast"] = cast_html
    extra["genre"] = genre_html

    # Search corpus (DataMapPlot substring-matches this): title + people + the
    # store's shelf sections, so "noir"/"kung fu"/"criterion" find the right
    # shelf. Synopsis is deliberately excluded — its common words ("love",
    # "war") would flood the substring match and dilute title/person lookup.
    section_search = df["sections"].map(lambda s: " ".join(s) if len(s) else "")
    qual_search = df["qualifiers"].map(lambda q: " ".join(q) if len(q) else "")  # Criterion, A24, Arrow...
    hover_text = (
        df["title"].fillna("")
        + " "
        + df["director"].fillna("")
        + " "
        + df["cast_str"].fillna("")
        + " "
        + section_search
        + " "
        + qual_search
    ).to_numpy()

    # --- colormaps ---
    year_num = _fill_nonfinite(df["year"].to_numpy(dtype=float))
    # One coarse genre per film. The COLORMAP buckets to top-15 + "Other" (a
    # 65-color glasbey legend is unreadable); the FILTER keeps all 65 so every
    # genre is selectable, so the two intentionally differ (no legend sync).
    genre_single = pick_coarse_genre(df["genres_coarse"])
    genre_cat = bucket_top_n(genre_single, 15).to_numpy()
    fmt_cat = df["formats"].map(format_bucket).to_numpy()
    rating_cat = rating_clean.to_numpy()
    editions = df["sku_count"].to_numpy(dtype=float).clip(1, 10)

    # Per-point values the Advanced Filters panel reads (extra_point_data cols,
    # distinct from the colormap _r/_g/_b/_a columns). Range-filter fields encode
    # "missing" as a sentinel strictly below the slider min (0 for year/runtime,
    # -1 for popularity) so undated/unmatched films are shown at reset but drop
    # out the moment that slider is touched — see build_filter_config.
    runtime_num = pd.to_numeric(df["runtime"], errors="coerce")
    pop_num = pd.to_numeric(df["popularity"], errors="coerce")
    # format & genre are MULTI-VALUE: a film can be on several formats / in
    # several genres, so the checkbox filter must match ANY of them. Encode the
    # set as a "|"-joined string the panel splits (see filter_panel.html).
    format_cats = df["formats"].map(format_categories)
    genre_cats = df["genres_coarse"].map(lambda gs: list(gs) if len(gs) else ["Other"])
    extra["format_filter"] = format_cats.map("|".join).to_numpy()
    extra["genre_filter"] = genre_cats.map("|".join).to_numpy()
    extra["rating_filter"] = rating_cat
    extra["editions_filter"] = df["sku_count"].to_numpy(dtype=int)
    extra["year_filter"] = df["year"].fillna(0).astype(int).to_numpy()
    extra["runtime_filter"] = runtime_num.where(runtime_num > 0).fillna(0).astype(int).to_numpy()
    extra["popularity_filter"] = pop_num.where(pop_num > 0).fillna(-1.0).round(2).to_numpy()

    # Legend follows MPAA severity, not alphabetical: datamapplot's ColorLegend
    # renders colorMapping in dict-insertion order, so build the dict in that
    # order. RATING_COLORS spread LAST so the hand-picked green->red palette wins
    # over the glasbey fallback (which only needs to cover stray values).
    rating_legend_order = ["G", "PG", "PG-13", "R", "NC-17", "X", "N/R"]
    _rating_colors = {**categorical_color_mapping(rating_cat, default="N/R"), **RATING_COLORS}
    rating_color_mapping = {
        r: _rating_colors[r]
        for r in rating_legend_order + [v for v in _rating_colors if v not in rating_legend_order]
        if r in _rating_colors
    }

    rawdata = [year_num, genre_cat, fmt_cat, rating_cat, editions]
    metadata = [
        {"field": "year", "description": "Release year", "kind": "continuous", "cmap": "viridis"},
        {
            "field": "store_genre",
            "description": "Genre",
            "kind": "categorical",
            "color_mapping": categorical_color_mapping(genre_cat, default="Other"),
        },
        {
            "field": "format",
            "description": "Format (best available)",
            "kind": "categorical",
            "color_mapping": FORMAT_COLORS,
        },
        {
            "field": "rating",
            "description": "MPAA rating",
            "kind": "categorical",
            "color_mapping": rating_color_mapping,
        },
        {"field": "editions", "description": "Editions in stock (1-10+)", "kind": "continuous", "cmap": "YlGnBu"},
    ]

    # TMDB-derived colormaps only exist on an enriched corpus (stage 02 run).
    if df["matched"].fillna(False).any():
        pop = df["popularity"].to_numpy(dtype=float)
        pop_log = np.log10(np.where(np.isfinite(pop) & (pop > 0), pop, np.nan))
        finite_pop = pop_log[np.isfinite(pop_log)]
        pop_log = np.where(np.isfinite(pop_log), pop_log, finite_pop.min() if finite_pop.size else 0.0)
        runtime_num = _fill_nonfinite(df["runtime"].to_numpy(dtype=float)).clip(40, 240)
        rawdata += [pop_log, runtime_num]
        # "Genre (TMDB)" is intentionally omitted from the colormap — the store's
        # coarse genre ("store_genre") is the canonical genre coloring, and the
        # TMDB genre was a near-duplicate that just crowded the dropdown.
        metadata += [
            {"field": "popularity", "description": "TMDB popularity (log)", "kind": "continuous", "cmap": "inferno"},
            {"field": "runtime", "description": "Runtime (min)", "kind": "continuous", "cmap": "cividis"},
        ]

    print(f"  [{variant}] rendering DataMapPlot...")
    fig = datamapplot.create_interactive_plot(
        coords_xy,
        *label_layers,
        hover_text=hover_text,
        hover_text_html_template=hover_template,
        extra_point_data=extra,
        on_click="window.open(`{mm_url}`, '_blank')",
        colormap_rawdata=rawdata,
        colormap_metadata=metadata,
        title=PROJECT_NAME,
        sub_title=f"{PROJECT_TAGLINE} · {VARIANT_BLURB[variant]}",
        enable_search=True,
        font_family="IBM Plex Sans",
        tooltip_font_family="IBM Plex Sans",
        darkmode=True,
    )
    out = map_html(variant)
    fig.save(str(out))

    # Post-render: attribution + faster zoom, the Advanced Filters panel, then
    # the mobile layer (viewport meta + touch info card).
    html = postprocess_html(out.read_text())
    config = build_filter_config(df, format_cats, genre_cats, rating_cat)
    html = inject_filter_panel(html, config)
    html = inject_mobile_support(html)
    html = inject_point_labels(html)
    out.write_text(html)
    n_filters = sum(len(s["filters"]) for s in config["sections"])
    print(f"  [{variant}] saved {out} ({out.stat().st_size / 1e6:.1f} MB) + {n_filters} filters")

    dh = docs_html(variant)
    dh.parent.mkdir(exist_ok=True)
    dh.write_bytes(out.read_bytes())
    print(f"  [{variant}] copied to {dh}")


def main():
    films = pd.read_parquet(FILMS_PARQUET)
    for variant in VARIANTS:
        render_variant(variant, films)


if __name__ == "__main__":
    main()
