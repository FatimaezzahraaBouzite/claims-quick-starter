"""Quick checks that the pipeline steps behave. Run: python -m pytest -q"""
import pandas as pd
import pytest

from src.pipeline import steps

CFG = {"data": {"text_column": "description", "target_column": "crash",
                "min_words": 3, "drop_duplicate_text": True}}


def test_clean_drops_short_and_duplicate_texts():
    df = pd.DataFrame({"description": ["The brakes failed and I crashed", "too short",
                                       "the brakes failed and i crashed", None],
                       "crash": [1, 0, 1, 0]})
    out = steps.validate_and_clean(df, CFG)
    assert len(out) == 1                      # short, duplicate (after lowercasing) and missing removed
    assert out.loc[0, "description"] == "the brakes failed and i crashed"


def test_clean_rejects_bad_target():
    df = pd.DataFrame({"description": ["one two three four"], "crash": ["yes"]})
    with pytest.raises(ValueError):
        steps.validate_and_clean(df, CFG)


def test_clean_rejects_missing_column():
    with pytest.raises(ValueError):
        steps.validate_and_clean(pd.DataFrame({"text": ["a b c"]}), CFG)
