import pytest
from src.data import parse_data_type, assign_leakage_groups


def test_parse_data_type():
    assert parse_data_type("adversarial_harmful") == ("adversarial", "harmful", "adversarial_harmful")
    assert parse_data_type("vanilla_benign") == ("vanilla", "benign", "vanilla_benign")
    with pytest.raises(ValueError):
        parse_data_type("unknown")


def test_exact_prompt_duplicates_share_group():
    rows = [
        {"source_id": "a", "prompt_hash": "x"},
        {"source_id": "b", "prompt_hash": "x"},
        {"source_id": "c", "prompt_hash": "z"},
    ]
    grouped = assign_leakage_groups(rows)
    assert grouped[0]["leakage_group_id"] == grouped[1]["leakage_group_id"]
    assert grouped[0]["leakage_group_id"] != grouped[2]["leakage_group_id"]
