import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))

from normalize import parse_catalog_title  # noqa: E402


def parse(raw):
    return parse_catalog_title(raw)


def test_simple_format_tag():
    p = parse("Alien (DVD)")
    assert p["title_display"] == "Alien"
    assert p["title_year"] is None
    assert p["film_key"] == "alien"


def test_inverted_article_with_year_and_format():
    p = parse("Fighter, The (1952) (DVD-R)")
    assert p["title_display"] == "The Fighter"
    assert p["title_year"] == 1952


def test_inverted_article_before_subtitle():
    p = parse("Conqueror, The: Hollywood Fallout (DVD)")
    assert p["title_display"] == "The Conqueror: Hollywood Fallout"


def test_multiple_commas_inverts_last():
    p = parse("Good, The Bad And The Ugly, The (Blu-Ray)")
    assert p["title_display"] == "The Good, The Bad And The Ugly"


def test_new_york_not_inverted():
    p = parse("New York, New York (VHS)")
    assert p["title_display"] == "New York, New York"


def test_unknown_qualifier_stripped_and_kept():
    p = parse("Totem (2023) (Latin American) (DVD)")
    assert p["title_display"] == "Totem"
    assert p["title_year"] == 2023
    assert p["qualifiers"] == ["Latin American"]


def test_criterion_and_oop():
    p = parse("Evil Dead 2 (Criterion) (VHS) (Out of Print)")
    assert p["title_display"] == "Evil Dead 2"
    assert p["qualifiers"] == []  # all known tags, not qualifiers


def test_leading_parenthetical_kept():
    p = parse("(500) Days Of Summer (DVD)")
    assert p["title_display"] == "(500) Days Of Summer"
    assert p["film_key"] == "500 days of summer"


def test_all_paren_title_not_emptied():
    p = parse("(Untitled)")
    assert p["title_display"] == "(Untitled)"


def test_season_split():
    p = parse("Sopranos, The: Season 3 (DVD)")
    assert p["title_display"] == "The Sopranos: Season 3"
    assert p["title_base"] == "The Sopranos"
    assert p["season"] == "Season 3"
    # seasons stay distinct films
    assert p["film_key"] != parse("Sopranos, The: Season 2 (DVD)")["film_key"]


def test_complete_series():
    p = parse("Freaks And Geeks: The Complete Series (DVD)")
    assert p["title_base"] == "Freaks And Geeks"
    assert p["season"] == "The Complete Series"


def test_foreign_article():
    p = parse("Dolce Vita, La (Blu-Ray)")
    assert p["title_display"] == "La Dolce Vita"


def test_compound_season_suffix():
    p = parse("2 Broke Girls: Season 6 - The Final Season (DVD)")
    assert p["title_base"] == "2 Broke Girls"
    assert "Season 6" in p["season"]


def test_set_and_disc_suffixes():
    assert parse("Above Suspicion: Set 2 (DVD)")["title_base"] == "Above Suspicion"
    p = parse("7th Heaven: Complete 1st Season Disc 4 (DVD)")
    assert p["title_base"] == "7th Heaven"


def test_mini_series_suffix():
    p = parse("A Gentleman In Moscow: The Mini Series (DVD)")
    assert p["title_base"] == "A Gentleman In Moscow"


def test_season_word_and_word():
    p = parse("A Haunting: Seasons 1 And 2 (DVD)")
    assert p["title_base"] == "A Haunting"


def test_part_suffix_split_but_title_intact():
    # "Part 2" splits for matching purposes, but the film identity (display
    # title / film_key) keeps it — and looks_tv must not treat it as TV.
    p = parse("Friday The 13th Part 2 (VHS)")
    assert p["title_display"] == "Friday The 13th Part 2"
    assert p["season"] == "Part 2"
    assert "part 2" in p["film_key"]


def test_plural_discs_and_comma_separators():
    p = parse("Andy Griffith Show – The Complete 3rd Season Discs 1 & 2 (DVD)")
    assert p["title_base"] == "Andy Griffith Show"
    p = parse("Andy Griffith Show – The Complete 4th Season, Disc 1&2 (DVD)")
    assert p["title_base"] == "Andy Griffith Show"
    assert "Disc" in p["season"] and not p["season"].startswith(",")


def test_truncated_format_tag_stripped():
    p = parse("Andy Griffith Show – The Complete 3rd Season Discs 5 (Dv")
    assert p["title_base"] == "Andy Griffith Show"
    p = parse("Some Very Long Movie Title That Got Cut (Blu")
    assert p["title_display"] == "Some Very Long Movie Title That Got Cut"


def test_movie_titled_the_final_season_survives():
    p = parse("The Final Season (DVD)")
    assert p["title_display"] == "The Final Season"
    assert p["season"] is None


def test_format_casing_groups_together():
    a = parse("ALIEN (DVD)")
    b = parse("Alien (Blu-Ray)")
    assert a["film_key"] == b["film_key"]


def test_year_not_at_end_still_found():
    p = parse("Crash (1996) (DVD)")
    assert p["title_display"] == "Crash"
    assert p["title_year"] == 1996
