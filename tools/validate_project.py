"""Static checks for the generated dataset and Colab notebook."""

from __future__ import annotations

import ast
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "user_story_test_cases.csv"
NOTEBOOK_PATH = ROOT / "FineTune_Test_Case_Generator_Qwen3.ipynb"
REQUIRED_CASE_FIELDS = {
    "id", "title", "type", "preconditions", "steps", "expected_result", "covers"
}


def validate_dataset() -> None:
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 80
    assert sum(row["split"] == "train" for row in rows) == 60
    assert sum(row["split"] == "validation" for row in rows) == 20

    train_families = {row["scenario_family"] for row in rows if row["split"] == "train"}
    val_families = {row["scenario_family"] for row in rows if row["split"] == "validation"}
    assert train_families.isdisjoint(val_families)

    for row in rows:
        cases = json.loads(row["expected_test_cases"])
        assert len(cases) == 3
        assert {case["type"] for case in cases} == {"positive", "negative", "boundary"}
        covered = set()
        for case in cases:
            assert set(case) == REQUIRED_CASE_FIELDS
            covered.update(case["covers"])
            assert case["steps"] and case["preconditions"]
        assert covered == {"AC1", "AC2", "AC3"}


def validate_notebook() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 21
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # IPython magics and shell commands are valid in Colab, not in Python's AST.
        if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            raise AssertionError(f"Python syntax error in notebook cell {index}: {exc}") from exc


if __name__ == "__main__":
    validate_dataset()
    validate_notebook()
    print("Project validation passed: dataset schema, split integrity, JSON targets, and notebook syntax.")
