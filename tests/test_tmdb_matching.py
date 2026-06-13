import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

tmdb = importlib.import_module("02_fetch_tmdb")


def movie(title, date, **kw):
    return {"title": title, "release_date": date, "original_title": kw.get("original_title", title)}


def test_exact_title_and_year_scores_above_threshold():
    s = tmdb.score_candidate(movie("Alien", "1979-05-25"), "Alien", 1979, is_tv=False)
    assert s >= 1.05  # early-stop tier


def test_wrong_year_same_title_penalized():
    # "Crash" 1996 (Cronenberg) vs "Crash" 2004 (Haggis): year keeps them apart
    good = tmdb.score_candidate(movie("Crash", "1996-10-04"), "Crash", 1996, is_tv=False)
    bad = tmdb.score_candidate(movie("Crash", "2004-09-10"), "Crash", 1996, is_tv=False)
    assert good > bad
    assert bad < tmdb.ACCEPT_THRESHOLD


def test_fuzzy_title_needs_year_support():
    # Slightly different title, right year -> accepted; no year info -> rejected
    near = movie("The Fighter", "1952-05-01")
    with_year = tmdb.score_candidate(near, "Fighter", 1952, is_tv=False)
    without_year = tmdb.score_candidate(movie("The Fighter", ""), "Fighter", None, is_tv=False)
    assert without_year < with_year


def test_original_title_matches_foreign_films():
    cand = {"title": "Seven Samurai", "original_title": "七人の侍", "release_date": "1954-04-26"}
    s = tmdb.score_candidate(cand, "Seven Samurai", 1954, is_tv=False)
    assert s >= 1.05


def test_tv_year_leniency():
    # Catalog year = a season's year, far from first_air_date: TV stays acceptable
    cand = {"name": "The Sopranos", "original_name": "The Sopranos", "first_air_date": "1999-01-10"}
    s = tmdb.score_candidate(cand, "The Sopranos", 2004, is_tv=True)
    assert s >= tmdb.ACCEPT_THRESHOLD


def test_movie_far_year_not_lenient():
    cand = movie("The Sopranos", "1999-01-10")
    s = tmdb.score_candidate(cand, "The Sopranos", 2004, is_tv=False)
    assert s < tmdb.ACCEPT_THRESHOLD


def test_punctuation_and_ampersand_insensitive():
    cand = movie("Fast & Furious", "2009-04-03")
    s = tmdb.score_candidate(cand, "Fast and Furious", 2009, is_tv=False)
    assert s >= 1.05


def test_digit_vs_word_numbers():
    cand = movie("One Hundred Men and a Girl", "1937-09-05")
    s = tmdb.score_candidate(cand, "100 Men And A Girl", 1937, is_tv=False)
    assert s >= tmdb.ACCEPT_THRESHOLD


def test_tmdb_side_subtitle_extension():
    cand = movie("12 Rounds 2: Reloaded", "2013-06-04")
    s = tmdb.score_candidate(cand, "12 Rounds 2", 2013, is_tv=False)
    assert s >= tmdb.ACCEPT_THRESHOLD


def test_short_prefix_does_not_overclaim():
    # "10" must not claim "10,000 Saints" via the prefix rule
    assert tmdb.similarity("10", "10,000 Saints") < 0.9


def test_leading_article_insensitive():
    cand = {"name": "The Andy Griffith Show", "original_name": "The Andy Griffith Show", "first_air_date": "1960-10-03"}
    s = tmdb.score_candidate(cand, "Andy Griffith Show", 1963, is_tv=True)
    assert s >= tmdb.ACCEPT_THRESHOLD


def test_article_strip_needs_substance():
    # "The Fly" vs "A Fly" must not become equal via article stripping... and
    # very short titles keep their article ("The It" != "It")
    assert tmdb.similarity("The It", "It") < 0.95


def test_prefix_rule_still_year_gated():
    # "Alien" vs "Aliens" 1986: prefix rule fires but the year penalty rejects
    cand = movie("Aliens", "1986-07-18")
    s = tmdb.score_candidate(cand, "Alien", 1979, is_tv=False)
    assert s < tmdb.ACCEPT_THRESHOLD
