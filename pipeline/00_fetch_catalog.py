"""Stage 00: fetch the Movie Madness rental catalog via the public WordPress REST API.

Writes:
  data/catalog_raw.parquet     one row per rental SKU (~98.8k), taxonomy fields as term-id lists
  data/taxonomy_terms.parquet  one row per term across the six rental taxonomies

Resumable: each completed page of 100 SKUs is appended to data/.catalog_pages.jsonl;
rerunning skips pages already on disk and re-assembles the parquet at the end.
Gentle by design: one request every REQUEST_INTERVAL_S, retries with backoff that
honor Retry-After, and an honest User-Agent. This is a small nonprofit's website.
"""

import json
import os
import tempfile
import time

import pandas as pd
import requests
from config import (
    CATALOG_RAW_PARQUET,
    DATA_DIR,
    MAX_CATALOG_PAGES,
    MM_API_URL,
    MM_USER_AGENT,
    RENTAL_TAXONOMIES,
    REQUEST_INTERVAL_S,
    TAXONOMY_TERMS_PARQUET,
)
from tqdm import tqdm

PAGES_JSONL = DATA_DIR / ".catalog_pages.jsonl"

PER_PAGE = 100
RENTAL_FIELDS = "id,slug,modified,title,content,format,rating,date,genre,location,language"
TERM_FIELDS = "id,name,slug,count,taxonomy"

session = requests.Session()
session.headers["User-Agent"] = MM_USER_AGENT

_last_request_t = 0.0


def get_with_retry(url: str, params: dict | None = None, max_retries: int = 6, timeout: int = 60) -> requests.Response:
    """Throttled GET with exponential backoff. Treats 429/5xx as retryable and
    honors Retry-After when the server sends one."""
    global _last_request_t
    for attempt in range(max_retries):
        wait_throttle = REQUEST_INTERVAL_S - (time.monotonic() - _last_request_t)
        if wait_throttle > 0:
            time.sleep(wait_throttle)
        try:
            _last_request_t = time.monotonic()
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(2**attempt * 5, 120)
                if attempt == max_retries - 1:
                    resp.raise_for_status()
                print(f"  HTTP {resp.status_code} on {url}, backing off {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt * 5, 120)
            print(f"  {type(e).__name__} on {url}, retrying in {wait:.0f}s...")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def atomic_write_parquet(df: pd.DataFrame, path) -> None:
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".parquet.tmp")
    os.close(tmp_fd)
    try:
        df.to_parquet(tmp_path)
        verify = pd.read_parquet(tmp_path)
        assert len(verify) == len(df), f"row count mismatch: {len(verify)} vs {len(df)}"
        os.replace(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def fetch_taxonomy_terms() -> pd.DataFrame:
    rows = []
    for tax in RENTAL_TAXONOMIES:
        page = 1
        while True:
            resp = get_with_retry(
                f"{MM_API_URL}/{tax}",
                params={"per_page": 100, "page": page, "_fields": TERM_FIELDS},
            )
            batch = resp.json()
            rows.extend(batch)
            total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
            if page >= total_pages:
                break
            page += 1
        print(f"  taxonomy '{tax}': {sum(1 for r in rows if r['taxonomy'] == tax)} terms")
    df = pd.DataFrame(rows)[["taxonomy", "id", "name", "slug", "count"]]
    return df


def load_done_pages() -> dict[int, list[dict]]:
    """Read completed pages from the JSONL checkpoint, tolerating a torn final
    line from a previous crash."""
    done: dict[int, list[dict]] = {}
    if not PAGES_JSONL.exists():
        return done
    with open(PAGES_JSONL) as f:
        for line in f:
            try:
                rec = json.loads(line)
                done[rec["page"]] = rec["rows"]
            except (json.JSONDecodeError, KeyError):
                continue  # torn line; that page will be re-fetched
    return done


def fetch_rentals() -> pd.DataFrame:
    params = {
        "per_page": PER_PAGE,
        "page": 1,
        "orderby": "id",
        "order": "asc",
        "_fields": RENTAL_FIELDS,
    }
    first = get_with_retry(f"{MM_API_URL}/rental", params=params)
    total = int(first.headers["X-WP-Total"])
    total_pages = int(first.headers["X-WP-TotalPages"])
    if MAX_CATALOG_PAGES is not None:
        total_pages = min(total_pages, MAX_CATALOG_PAGES)
    print(f"catalog: {total} rental SKUs across {total_pages} pages to fetch")

    done = load_done_pages()
    if done:
        print(f"  resuming: {len(done)} pages already on disk")
    if 1 not in done:
        with open(PAGES_JSONL, "a") as f:
            f.write(json.dumps({"page": 1, "rows": first.json()}) + "\n")
        done[1] = first.json()

    with open(PAGES_JSONL, "a") as f:
        for page in tqdm(range(1, total_pages + 1), desc="catalog pages"):
            if page in done:
                continue
            params["page"] = page
            rows = get_with_retry(f"{MM_API_URL}/rental", params=params).json()
            f.write(json.dumps({"page": page, "rows": rows}) + "\n")
            f.flush()

    done = load_done_pages()
    all_rows = [r for page in sorted(done) for r in done[page] if page <= total_pages]
    df = pd.json_normalize(all_rows)
    df = df.rename(columns={"title.rendered": "title_raw", "content.rendered": "content_html"})
    df = df.drop_duplicates(subset="id", keep="first")  # pages can shift if SKUs are added mid-crawl
    keep = ["id", "slug", "modified", "title_raw", "content_html"] + RENTAL_TAXONOMIES
    df = df[[c for c in keep if c in df.columns]]
    return df


def main() -> None:
    print("fetching taxonomy terms...")
    terms = fetch_taxonomy_terms()
    atomic_write_parquet(terms, TAXONOMY_TERMS_PARQUET)
    print(f"wrote {len(terms)} terms -> {TAXONOMY_TERMS_PARQUET}")

    rentals = fetch_rentals()
    assert len(rentals) > 0, "no rentals fetched"
    assert rentals["id"].is_unique
    atomic_write_parquet(rentals, CATALOG_RAW_PARQUET)
    print(f"wrote {len(rentals)} SKUs -> {CATALOG_RAW_PARQUET}")

    if MAX_CATALOG_PAGES is None:
        PAGES_JSONL.unlink(missing_ok=True)  # full run complete; next run starts fresh


if __name__ == "__main__":
    main()
