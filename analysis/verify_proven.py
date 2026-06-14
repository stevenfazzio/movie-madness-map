"""Verify the 'proven' stratum: is a confirmed film's synopsis really a DIFFERENT
film's text (true positive), or could it be a dedup-miss (same film, variant
title -> synopsis actually correct)? Pull full text + title disjointness."""

import re

import pandas as pd

MIN_LEN = 60
WRONG_JAC = 0.06
OWNER_THR = 0.12
TITLE_STOP = {"the", "a", "an", "of", "and", "de", "la", "le", "el", "il", "part", "vol",
              "los", "las", "un", "une", "season", "series", "disc", "collection", "complete"}
EN_STOP = set(
    "the a an of and or to in is are was were on for with at by from as it its his her their this "
    "that these those who what which into out up down over under after before during while when where "
    "why how he she they we you i him them us not no new film movie story series season about all two one".split())


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

proven = []
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
    dom = {t for t, c in freq.items() if c >= 2}
    for m in members:
        if ttok(m.title) & dom:
            continue
        v = ov[m.film_id]
        if not (m.matched and v == v and v < WRONG_JAC):
            continue
        others = [o for o in members if o.film_id != m.film_id and ov[o.film_id] == ov[o.film_id]]
        if not others:
            continue
        owner = max(others, key=lambda o: ov[o.film_id])
        if ov[owner.film_id] < OWNER_THR:
            continue
        proven.append(dict(
            film_id=m.film_id, title=m.title, year=m.year,
            owner_title=owner.title, owner_year=owner.year, owner_overlap=round(ov[owner.film_id], 3),
            title_jac=round(jac(ttok(m.title), ttok(owner.title)), 3),
            same_year=(str(m.year) == str(owner.year)),
            syn=str(m.synopsis_catalog), owner_overview=str(owner.overview),
        ))

P = pd.DataFrame(proven)
print(f"proven films: {len(P)}")
print(f"owner_overlap distribution: min={P.owner_overlap.min()} median={P.owner_overlap.median()} "
      f"frac==1.0: {(P.owner_overlap == 1.0).mean():.2f}")
print(f"title-token disjoint from owner (different films): {(P.title_jac == 0).mean():.2%}")
print(f"possible dedup-miss (title overlap>0 AND same year): {((P.title_jac > 0) & P.same_year).sum()}")

dar = F[F.film_id == "the darjeeling limited|2007"]
print("\nDarjeeling in proven?:", "the darjeeling limited|2007" in set(P.film_id))

print("\n=== 12 proven, full text (does flagged film's synopsis describe the OWNER, not itself?) ===")
for _, r in P.sample(min(12, len(P)), random_state=3).iterrows():
    print(f"\n  FLAGGED: {r.title!r} ({r.year})   OWNER: {r.owner_title!r} ({r.owner_year})  "
          f"[ovl={r.owner_overlap} titleJac={r.title_jac} sameYr={r.same_year}]")
    print(f"    flagged's catalog synopsis: {r.syn[:170]!r}")
    print(f"    owner's TMDB overview:      {r.owner_overview[:170]!r}")
