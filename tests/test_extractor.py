"""Unit tests for the ingredient-parsing helpers in extractor.py.

These are the pure functions that turn messy LLM output into a clean,
de-duplicated ingredient list. No network or browser needed.
"""

from lib import extractor


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


def test_split_ingredient_string_english_label():
    # The OCC drugComposition field sometimes uses the English "Ingredients:" label.
    assert extractor.split_ingredient_string(
        "Ingredients: Aqua, Sodium Laureth Sulfate, Parfum"
    ) == ["Aqua", "Sodium Laureth Sulfate", "Parfum"]


def test_split_ingredient_string_empty():
    assert extractor.split_ingredient_string("") == []
