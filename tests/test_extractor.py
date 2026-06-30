"""Unit tests for the ingredient-parsing helpers in extractor.py.

These are the pure functions that turn messy LLM output into a clean,
de-duplicated ingredient list. No network or browser needed.
"""

import extractor


def test_clean_ingredient_strips_dutch_label():
    assert extractor.clean_ingredient("Ingrediënten: Aqua") == "Aqua"
    assert extractor.clean_ingredient("Samenstelling - Glycerin") == "Glycerin"


def test_clean_ingredient_trims_and_collapses():
    assert extractor.clean_ingredient("  Cetearyl   Alcohol .") == "Cetearyl Alcohol"
    assert extractor.clean_ingredient("") == ""
    assert extractor.clean_ingredient(None) == ""


def test_split_ingredient_string_separators():
    assert extractor.split_ingredient_string("Aqua, Cetearyl Alcohol; Linalool") == [
        "Aqua",
        "Cetearyl Alcohol",
        "Linalool",
    ]


def test_split_ingredient_string_bullets_and_label():
    assert extractor.split_ingredient_string(
        "Ingrediënten: Aqua · Glycerin • Parfum"
    ) == ["Aqua", "Glycerin", "Parfum"]


def test_split_ingredient_string_empty():
    assert extractor.split_ingredient_string("") == []


def test_parse_response_json_list_dedupes_preserving_order():
    raw, ings = extractor.parse_llm_ingredients_response(
        '{"found": true, "ingredients": ["Aqua", "Aqua", "Glycerin"]}'
    )
    assert ings == ["Aqua", "Glycerin"]
    assert "Aqua" in raw


def test_parse_response_json_string_value():
    _, ings = extractor.parse_llm_ingredients_response(
        '{"ingredients": "Aqua, Glycerin"}'
    )
    assert ings == ["Aqua", "Glycerin"]


def test_parse_response_json_embedded_in_text():
    _, ings = extractor.parse_llm_ingredients_response(
        'Sure: {"found": true, "ingredients": ["Aqua"]} done'
    )
    assert ings == ["Aqua"]


def test_parse_response_empty_input():
    assert extractor.parse_llm_ingredients_response("") == ("", [])


def test_parse_response_found_false():
    _, ings = extractor.parse_llm_ingredients_response(
        '{"found": false, "ingredients": []}'
    )
    assert ings == []
