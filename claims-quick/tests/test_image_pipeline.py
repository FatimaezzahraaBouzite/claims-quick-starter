"""Checks for the photo pipeline's label logic. Run: python -m pytest -q"""
import pandas as pd
import pytest
import yaml
from PIL import Image

from src.image_pipeline import steps


def test_parse_box_and_polygon():
    assert steps._parse_label_line("2 0.5 0.5 0.2 0.5") == (2, pytest.approx(0.1))
    cls, area = steps._parse_label_line("1 0.1 0.1 0.5 0.1 0.5 0.3 0.1 0.3")   # polygon
    assert cls == 1 and area == pytest.approx(0.08)


def test_photo_takes_worst_damage(tmp_path):
    root = tmp_path / "photos"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    yaml.safe_dump({"names": ["Minor-damage", "Moderate", "SEVERE"]}, open(root / "data.yaml", "w"))
    Image.new("RGB", (40, 40)).save(root / "train" / "images" / "a.jpg")
    (root / "train" / "labels" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n2 0.3 0.3 0.2 0.2\n")
    Image.new("RGB", (40, 40)).save(root / "train" / "images" / "b.jpg")   # no label file
    cfg = {"data": {"root": str(root), "class_order": ["minor", "moderate", "severe"]}}
    df = steps.load_labels(cfg).set_index(df_col := "path")
    a = df.loc[str(root / "train" / "images" / "a.jpg")]
    assert a["label"] == "severe" and a["n_damages"] == 2 and a["n_classes_in_photo"] == 2
    assert pd.isna(df.loc[str(root / "train" / "images" / "b.jpg"), "label"])
