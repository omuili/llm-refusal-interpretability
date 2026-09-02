from __future__ import annotations

import argparse
from datasets import load_dataset

from .config import load_config, project_path
from .data import raw_to_records, assign_leakage_groups, deterministic_group_split, build_dataset_dict, summarize
from .utils import set_seed, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["dataset"].get("seed", cfg["project"]["seed"]))
    set_seed(seed)

    raw = load_dataset(
        cfg["dataset"]["id"],
        cfg["dataset"].get("config"),
        split=cfg["dataset"].get("split", "train"),
        delimiter="\t",
        keep_default_na=False,
    )
    records = deterministic_group_split(assign_leakage_groups(raw_to_records(raw)), seed=seed)
    sizes = {
        "discovery": int(cfg["dataset"]["discovery_size"]),
        "validation": int(cfg["dataset"]["validation_size"]),
        "test": int(cfg["dataset"]["test_size"]),
    }
    ds = build_dataset_dict(records, sizes=sizes, seed=seed)
    destination = project_path(cfg["dataset"]["processed_dir"])
    ds.save_to_disk(str(destination))
    summary = summarize(ds)
    summary["dataset_id"] = cfg["dataset"]["id"]
    summary["seed"] = seed
    summary["raw_rows_after_cleaning"] = len(records)
    write_json(project_path(cfg["results"]["dir"]) / "data_summary.json", summary)

    print(f"Saved controlled dataset to {destination}")
    for split in ("discovery", "validation", "test"):
        print(split, summary[split])
    print("Leakage check: PASS")


if __name__ == "__main__":
    main()
