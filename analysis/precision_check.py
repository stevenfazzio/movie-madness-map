"""(a) Precision spot-check of the confirmed wrong-synopsis set.

For each confirmed-wrong film, look at the OTHER films sharing its exact synopsis
and find the 'owner' -- the sibling whose own TMDB overview best matches that
shared synopsis. If a sibling's overview matches well, we can NAME the film the
synopsis really belongs to => two-sided proof (certain true positive). Otherwise
the flag rests only on the one-sided 'overview-disagrees' gate => needs review.

Emits a stratified random sample (proven + review) with full evidence to adjudicate.
"""

import re

import pandas as pd

MIN_LEN = 60
WRONG_JAC = 0.06  # a film's own catalog-synopsis vs its overview (the confirm gate)

TITLE_STOP = {"the", "a", "an", "of", "and", "de", "la", "le", "el", "il", "part", "vol",
              "los", "las", "un", "une", "season", "series", "disc", "collection", "complete"}
EN_STOP = set(
    "the a an of and or to in is are was were on for with at by from as it its his her their "
    "this that these those who what which into out up down over under after before during while "
    "when where why how he she they we you i him them us not no new film movie story series season "
    "about all two one".split()
)


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def ttok(t):
    return {w for w in norm(t).split() if w and w not in TITLE_STOP and not w.isdigit()}


def cw(s):
    return {w for w in norm(s).split() if len(w) > 2 and w not in EN_STOP}


def jac(a, b):
    return len(a & b) / len(a | b) if (a and b) else 0.0


F = pd.read_parquet("data/films.parquet")
cat = F[F.synopsis_source == "catalog"].copy()
cat["sn"] = cat.synopsis_catalog.map(norm)
cat = cat[cat.sn.str.len() >= MIN_LEN].drop_duplicates("film_id")

rows = []
for sn, g in cat.groupby("sn"):
    if g.film_id.nunique() < 2:
        continue
    members = list(g.itertuples())
    synt = cw(sn)
    ov = {m.film_id: (jac(synt, cw(m.overview)) if m.matched else float("nan")) for m in members}
    freq = {}
    for m in members:
        for t in ttok(m.title):
            freq[t] = freq.get(t, 0) + 1
    dom = {t for t, c in freq.items() if c >= 2}  # franchise signature
    for m in members:
        if ttok(m.title) & dom:
            continue  # legit franchise/variant member
        v = ov[m.film_id]
        if not (m.matched and v == v and v < WRONG_JAC):
            continue  # not confirmed-wrong
        # owner = best-matching OTHER member
        others = [(o, ov[o.film_id]) for o in members
                  if o.film_id != m.film_id and ov[o.film_id] == ov[o.film_id]]
        owner, owner_v = (max(others, key=lambda x: x[1]) if others else (None, float("nan")))
        rows.append(dict(
            film_id=m.film_id, title=m.title, year=m.year, group_size=len(members),
            own_overlap=round(v, 3),
            owner_title=(owner.title if owner is not None else ""),
            owner_year=(owner.year if owner is not None else ""),
            owner_overlap=(round(owner_v, 3) if owner_v == owner_v else float("nan")),
            wrong_synopsis=str(m.synopsis_catalog)[:200],
            true_overview=str(m.overview)[:200],
        ))

C = pd.DataFrame(rows)
print(f"confirmed-wrong films: {len(C)}")
print("\nhow many are PROVABLE (a sibling owns the synopsis) at various thresholds:")
for thr in (0.08, 0.10, 0.12, 0.15, 0.20):
    n = int((C.owner_overlap >= thr).sum())
    print(f"  owner_overlap >= {thr:.2f}: {n}  ({100*n/len(C):.0f}%)")

THR = 0.12
C["stratum"] = ["proven" if v >= THR else "review" for v in C.owner_overlap.fillna(-1)]
print("\nstratum @0.12:\n", C.stratum.value_counts().to_string())

samp = pd.concat([
    C[C.stratum == "proven"].sample(min(20, (C.stratum == "proven").sum()), random_state=7),
    C[C.stratum == "review"].sample(min(30, (C.stratum == "review").sum()), random_state=7),
])
samp.to_csv("analysis/precision_sample.csv", index=False)
C.to_csv("analysis/confirmed_wrong_full.csv", index=False)

print("\n================ REVIEW SAMPLE (adjudicate each) ================")
for _, r in samp[samp.stratum == "review"].iterrows():
    print(f"\n[{r.stratum}] {r.title!r} ({r.year})  own_overlap={r.own_overlap} group_x{r.group_size}"
          f"  owner?={r.owner_title!r}({r.owner_overlap})")
    print(f"   catalog-syn: {r.wrong_synopsis!r}")
    print(f"   TMDB-overv:  {r.true_overview!r}")

print("\n================ PROVEN SAMPLE (sibling owns synopsis) ================")
for _, r in samp[samp.stratum == "proven"].iterrows():
    print(f"  {r.title!r} ({r.year})  <-- synopsis really belongs to {r.owner_title!r} "
          f"({r.owner_year})  [owner_overlap={r.owner_overlap}]")
