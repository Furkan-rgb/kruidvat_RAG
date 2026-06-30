"""INCI ingredient-string parsing and cleaning helpers.

The product API returns ingredients as a single label-prefixed string, e.g.
"Ingrediënten: Aqua, Sodium Laureth Sulfate, ...". These helpers strip the
label (Dutch or English) and split it into a clean list of ingredients.
"""

import re


def clean_ingredient(s):
    if not s:
        return ""
    s = s.strip()
    # Remove leading labels like 'Ingrediënten:', 'Ingredients:' or 'Samenstelling:'
    s = re.sub(r"(?i)^(?:ingredi[eë]nt(?:en|s)?|samenstelling)\s*[:\-\s]*", "", s)
    # Strip surrounding punctuation
    s = s.strip(" .;:-")
    # Collapse spaces
    s = re.sub(r"\s+", " ", s)
    return s


def split_ingredient_string(s):
    # Split on commas/•/·/; then clean each
    if not s:
        return []
    # Remove any leading free text before an Ingredients/Ingrediënten/Samenstelling label
    s = re.sub(r"(?i).*?(?:ingredi[eë]nt(?:en|s)?|samenstelling)[:\-\s]*", "", s)
    s = s.replace("·", ", ").replace("•", ", ")
    parts = [p.strip() for p in re.split(r",|;", s) if p.strip()]
    return [clean_ingredient(p) for p in parts if clean_ingredient(p)]
