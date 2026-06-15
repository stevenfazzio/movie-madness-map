# The recycled-record quirk

A note on a quirk in the raw Movie Madness catalog, for anyone digging through
it (and as a possible aside if the project ever gets written up publicly).

Movie Madness reuses its rental records. When a title leaves the shelves, an old
record gets re-titled for whatever takes its place — but the original URL slug
and synopsis stay behind. So the record now called "The Imitation Game" still
lives at an address ending in `silicon-valley-season-6-dvd`, and in the public
data it still carries Silicon Valley's plot. A few thousand entries in the raw
catalog describe the wrong movie this way.

The store's own search reads synopses from a separate system, so nobody browsing
the site ever sees this — the staleness lives only in the raw catalog data and
old permalinks. The pipeline detects and corrects it (see
`flag_contaminated_catalog_synopsis` in stage 03 and the audits alongside this
file); the recovered set is catalogued in
`movie_madness_recycled_records.xlsx`.

It's a fun artifact more than a problem to fix.
