"""Stage 02: enrich films with TMDB matches (synopsis gap-fill, director/cast,
poster URLs, extra metadata).

Reads  data/films_base.parquet
Writes data/tmdb.parquet — one row per film_id, INCLUDING non-matches (matched=False
       with a skip/fail reason), so stage 03 can left-join one file and report coverage.

Matching: /search/movie with title+year (TV-looking films use /search/tv without
year — season N of a show airs years after first_air_date), scored by normalized
title similarity + year proximity. Accepted matches get one details call with
append_to_response=credits. ~2 calls/film, throttled => full catalog is an
overnight-ish background run (~4 h). Resumable: checkpointed parquet keyed on
film_id; rerunning skips films already processed.

Requires TMDB_API_KEY in .env (v3 key or v4 read token both work).
"""

import os
import tempfile
from difflib import SequenceMatcher

import pandas as pd
from config import (
    FILMS_BASE_PARQUET,
    TMDB_API_KEY,
    TMDB_API_URL,
    TMDB_CHECKPOINT_EVERY,
    TMDB_PARQUET,
    TMDB_REQUEST_INTERVAL_S,
    TMDB_SKIP_FORMATS,
)
from http_util import ThrottledGetter
from normalize import TV_SEASON_RE, norm_key
from tqdm import tqdm

EMPTY = {
    "tmdb_id": None,
    "media_type": None,
    "tmdb_title": None,
    "tmdb_year": None,
    "overview": "",
    "poster_path": None,
    "tmdb_genres": [],
    "director": "",
    "cast_top": [],
    "runtime": None,
    "vote_average": None,
    "vote_count": None,
    "popularity": None,
    "original_language": None,
    "match_score": None,
}


def make_getter() -> ThrottledGetter:
    headers = {}
    if TMDB_API_KEY and TMDB_API_KEY.startswith("eyJ"):  # v4 read access token
        headers["Authorization"] = f"Bearer {TMDB_API_KEY}"
    return ThrottledGetter(TMDB_REQUEST_INTERVAL_S, "movie-madness-map/0.1", headers)


def api_params(extra: dict | None = None) -> dict:
    params = dict(extra or {})
    if TMDB_API_KEY and not TMDB_API_KEY.startswith("eyJ"):  # v3 key
        params["api_key"] = TMDB_API_KEY
    return params


# Digit tokens -> words, so "100 Men and a Girl" meets TMDB's
# "One Hundred Men and a Girl". Bounded on purpose; rare numbers stay digits.
_NUM_WORDS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six",
    "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
    "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen",
    "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen",
    "20": "twenty", "30": "thirty", "40": "forty", "50": "fifty", "60": "sixty",
    "70": "seventy", "80": "eighty", "90": "ninety", "100": "one hundred",
    "1000": "one thousand",
}  # fmt: skip


def _spell_numbers(key: str) -> str:
    return " ".join(_NUM_WORDS.get(tok, tok) for tok in key.split())


def _strip_article(key: str) -> str:
    for art in ("the ", "a ", "an "):
        if key.startswith(art) and len(key) > len(art) + 3:
            return key[len(art) :]
    return key


def similarity(a: str, b: str) -> float:
    ka, kb = norm_key(a), norm_key(b)
    if not ka or not kb:
        return 0.0
    if ka == kb:
        return 1.0
    sim = SequenceMatcher(None, ka, kb).ratio()
    # The catalog drops/keeps leading articles inconsistently ("Andy Griffith
    # Show" vs TMDB "The Andy Griffith Show") — equal-modulo-article is a match.
    if _strip_article(ka) == _strip_article(kb):
        sim = max(sim, 0.95)
    sa, sb = _spell_numbers(ka), _spell_numbers(kb)
    if (sa, sb) != (ka, kb):
        sim = max(sim, 1.0 if sa == sb else SequenceMatcher(None, sa, sb).ratio())
    # One title extending the other ("12 rounds 2" vs "12 rounds 2 reloaded"):
    # near-certain same film when the year also agrees (the year gate still
    # applies in score_candidate). Length guard keeps "10" from claiming
    # "10,000 Saints".
    if min(len(ka), len(kb)) >= 8 and (ka.startswith(kb) or kb.startswith(ka)):
        sim = max(sim, 0.95)
    return sim


def score_candidate(cand: dict, title: str, year, is_tv: bool) -> float:
    cand_title = cand.get("name") if is_tv else cand.get("title")
    cand_orig = cand.get("original_name") if is_tv else cand.get("original_title")
    sim = max(similarity(title, cand_title or ""), similarity(title, cand_orig or ""))
    date = cand.get("first_air_date") if is_tv else cand.get("release_date")
    cand_year = int(date[:4]) if date and len(date) >= 4 and date[:4].isdigit() else None
    if pd.isna(year) or year is None or cand_year is None:
        year_adj = -0.05  # unknown year: lean on title alone, slightly stricter
    else:
        delta = abs(int(year) - cand_year)
        # TV: catalog year is often a mid-run season's year, so be lenient.
        year_adj = {0: 0.10, 1: 0.06}.get(delta, 0.0 if (is_tv and delta <= 12) else -0.30)
    return sim + year_adj


def best_match(getter: ThrottledGetter, title: str, year, is_tv: bool) -> tuple[dict | None, float]:
    """Try search passes from most to least specific; return (candidate, score)."""
    passes = []
    if is_tv:
        passes.append(("/search/tv", {"query": title}))
        if ":" in title:
            passes.append(("/search/tv", {"query": title.split(":")[0].strip()}))
    else:
        if year is not None and not pd.isna(year):
            passes.append(("/search/movie", {"query": title, "year": int(year)}))
        passes.append(("/search/movie", {"query": title}))
        if ":" in title:
            passes.append(("/search/movie", {"query": title.split(":")[0].strip()}))

    best, best_score = None, 0.0
    for path, params in passes:
        params = api_params({**params, "include_adult": "true"})
        results = getter.get(f"{TMDB_API_URL}{path}", params=params).json().get("results", [])
        for cand in results[:8]:
            s = score_candidate(cand, title, year, is_tv)
            if s > best_score:
                best, best_score = cand, s
        if best_score >= 1.05:  # exact title + year agreement; stop early
            break
    return best, best_score


ACCEPT_THRESHOLD = 0.93  # exact title needs year sanity; fuzzy title needs year support


def fetch_details(getter: ThrottledGetter, tmdb_id: int, is_tv: bool) -> dict:
    kind = "tv" if is_tv else "movie"
    d = getter.get(f"{TMDB_API_URL}/{kind}/{tmdb_id}", params=api_params({"append_to_response": "credits"})).json()
    if is_tv:
        director = ", ".join(p["name"] for p in d.get("created_by", []) or [])
        date = d.get("first_air_date")
        runtime = (d.get("episode_run_time") or [None])[0]
        title = d.get("name")
    else:
        crew = (d.get("credits") or {}).get("crew", [])
        director = ", ".join(p["name"] for p in crew if p.get("job") == "Director")
        date = d.get("release_date")
        runtime = d.get("runtime")
        title = d.get("title")
    cast = [p["name"] for p in ((d.get("credits") or {}).get("cast", []) or [])[:6]]
    return {
        "tmdb_id": tmdb_id,
        "media_type": kind,
        "tmdb_title": title,
        "tmdb_year": int(date[:4]) if date and date[:4].isdigit() else None,
        "overview": d.get("overview") or "",
        "poster_path": d.get("poster_path"),
        "tmdb_genres": [g["name"] for g in d.get("genres", []) or []],
        "director": director,
        "cast_top": cast,
        "runtime": runtime,
        "vote_average": d.get("vote_average"),
        "vote_count": d.get("vote_count"),
        "popularity": d.get("popularity"),
        "original_language": d.get("original_language"),
    }


def atomic_write(df: pd.DataFrame, path) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path)
        assert len(pd.read_parquet(tmp_path)) == len(df)
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def looks_tv(row) -> bool:
    """TV needs a television-ish signal: a 'Season/Series' suffix or a TV shelf.
    'Part 2' / 'Vol. 3' alone do NOT count — feature films use those too (e.g.
    'Friday The 13th Part 2' must not match the 1987 Friday the 13th TV series)."""
    season = row["season"]
    if season is not None and not pd.isna(season) and TV_SEASON_RE.search(str(season)):
        return True
    return any("TV" in s or "(Series)" in s for s in row["sections"])


def main() -> None:
    assert TMDB_API_KEY, "TMDB_API_KEY missing — add it to .env (see .env.example)"
    films = pd.read_parquet(FILMS_BASE_PARQUET)
    films = films.sort_values("film_id")  # deterministic order for resume

    done_rows: list[dict] = []
    done_ids: set[str] = set()
    if TMDB_PARQUET.exists():
        prior = pd.read_parquet(TMDB_PARQUET)
        prior = prior[prior["film_id"].isin(set(films["film_id"]))]  # drop orphans of older corpora
        n_err = prior["match_method"].str.startswith("error:").sum()
        if n_err:
            print(f"  dropping {n_err} prior error rows so they retry")
            prior = prior[~prior["match_method"].str.startswith("error:")]
        done_rows = prior.to_dict("records")
        done_ids = set(prior["film_id"])
        print(f"resuming: {len(done_ids)} films already processed")

    getter = make_getter()
    todo = films[~films["film_id"].isin(done_ids)]
    print(f"{len(todo)} films to match against TMDB ({len(films)} total)")

    since_checkpoint = 0
    n_errors = 0
    n_processed = 0
    for _, row in tqdm(todo.iterrows(), total=len(todo), desc="TMDB"):
        base = {"film_id": row["film_id"], "matched": False, "match_method": "", **EMPTY}
        try:
            if set(row["formats"]) and set(row["formats"]) <= TMDB_SKIP_FORMATS:
                base["match_method"] = "skipped_format"
            else:
                is_tv = looks_tv(row)
                # TV matches against the show name (title_base, season suffix
                # stripped); movies against the full display title.
                query_title = row["title_base"] if is_tv else row["title"]
                cand, score = best_match(getter, query_title, row["year"], is_tv)
                if cand is not None and score >= ACCEPT_THRESHOLD:
                    details = fetch_details(getter, cand["id"], is_tv)
                    base.update(details)
                    base.update(matched=True, match_score=round(score, 3), match_method="tv" if is_tv else "movie")
                else:
                    base["match_method"] = "no_match"
                    base["match_score"] = round(score, 3) if cand is not None else None
        except Exception as e:  # one bad film shouldn't kill an overnight run
            base["match_method"] = f"error:{type(e).__name__}"
            n_errors += 1
        done_rows.append(base)
        n_processed += 1
        # Circuit breaker: scattered errors are fine (they retry next run), but a
        # high error RATE means a systematic bug — stop before burning hours.
        if n_processed >= 50 and n_errors / n_processed > 0.3:
            atomic_write(pd.DataFrame(done_rows), TMDB_PARQUET)
            raise RuntimeError(
                f"{n_errors}/{n_processed} films errored this run — systematic failure, aborting. "
                f"Last error method: {base['match_method']}"
            )
        since_checkpoint += 1
        if since_checkpoint >= TMDB_CHECKPOINT_EVERY:
            atomic_write(pd.DataFrame(done_rows), TMDB_PARQUET)
            since_checkpoint = 0

    out = pd.DataFrame(done_rows)
    atomic_write(out, TMDB_PARQUET)
    n_matched = out["matched"].sum()
    print(f"matched {n_matched}/{len(out)} films ({n_matched / len(out):.0%}) -> {TMDB_PARQUET}")
    for method, n in out["match_method"].value_counts().items():
        print(f"  {method or 'matched'}: {n}")


if __name__ == "__main__":
    main()
