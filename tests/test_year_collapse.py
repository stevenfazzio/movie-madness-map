import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

prepare = importlib.import_module("01_prepare_films")
cym = prepare.canonical_year_map


def test_adjacent_rerelease_years_merge():
    # Army of Darkness: 1992 (5 SKUs) + 1993 (2 SKUs) -> one film, canonical 1992
    m = cym([1992, 1992, 1992, 1992, 1992, 1993, 1993])
    assert set(m.values()) == {1992}


def test_remake_years_stay_separate():
    # Of Mice and Men 1939 / 1968 / 1992 are genuinely different films
    m = cym([1939, 1968, 1992])
    assert m == {1939: 1939, 1968: 1968, 1992: 1992}


def test_modal_year_wins_ties_to_earliest():
    # equal counts -> earliest year is canonical
    m = cym([2002, 2003])
    assert set(m.values()) == {2002}


def test_chained_years_within_gap_merge():
    # 2000, 2001, 2002 chain (each gap < 3) into one cluster
    m = cym([2000, 2001, 2001, 2002])
    assert set(m.values()) == {2001}  # 2001 is modal


def test_exactly_gap_splits():
    # gap of exactly 3 (YEAR_SPLIT_GAP) splits into two films
    m = cym([2000, 2003])
    assert m == {2000: 2000, 2003: 2003}


def test_empty():
    assert cym([]) == {}
