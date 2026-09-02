from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib

from .utils import normalize_text, sha256_text

ALLOWED_4WAY = {
    "adversarial_benign",
    "adversarial_harmful",
    "vanilla_benign",
    "vanilla_harmful",
}


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def parse_data_type(data_type: str) -> tuple[str, str, str]:
    value = str(data_type).strip().lower()
    if value not in ALLOWED_4WAY:
        raise ValueError(f"Unexpected WildJailbreak data_type: {data_type!r}")
    style, harmfulness = value.split("_", 1)
    return style, harmfulness, value


def raw_to_records(raw) -> list[dict]:
    records: list[dict] = []
    for row in raw:
        try:
            style, harmfulness, label = parse_data_type(row["data_type"])
        except (KeyError, ValueError):
            continue

        vanilla = str(row.get("vanilla") or "").strip()
        adversarial = str(row.get("adversarial") or "").strip()
        prompt = adversarial if style == "adversarial" else vanilla
        if not prompt or not vanilla:
            continue

        source_id = sha256_text(normalize_text(vanilla))
        prompt_hash = sha256_text(normalize_text(prompt))
        records.append({
            "instruction": prompt,
            "source_id": source_id,
            "prompt_hash": prompt_hash,
            "style": style,
            "harmfulness": harmfulness,
            "label_4way": label,
        })
    return records


def assign_leakage_groups(records: list[dict]) -> list[dict]:
    uf = UnionFind(len(records))
    seen_source: dict[str, int] = {}
    seen_prompt: dict[str, int] = {}

    for i, row in enumerate(records):
        for key, table in ((row["source_id"], seen_source), (row["prompt_hash"], seen_prompt)):
            if key in table:
                uf.union(i, table[key])
            else:
                table[key] = i

    roots = defaultdict(list)
    for i in range(len(records)):
        roots[uf.find(i)].append(i)

    root_to_id = {}
    for root, indices in roots.items():
        stable = "|".join(sorted(records[i]["source_id"] for i in indices))
        root_to_id[root] = hashlib.sha256(stable.encode("utf-8")).hexdigest()

    out = []
    for i, row in enumerate(records):
        copy = dict(row)
        copy["leakage_group_id"] = root_to_id[uf.find(i)]
        out.append(copy)
    return out


def deterministic_group_split(records: list[dict], seed: int = 42) -> list[dict]:
    out = []
    for row in records:
        token = f"{seed}|{row['leakage_group_id']}"
        bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)
        split = "discovery" if bucket < 0.70 else "validation" if bucket < 0.85 else "test"
        copy = dict(row)
        copy["split"] = split
        out.append(copy)
    return out


def _rank(seed: int, split: str, label: str, group_id: str, index: int) -> str:
    return hashlib.sha256(f"{seed}|{split}|{label}|{group_id}|{index}".encode()).hexdigest()


def balanced_unique_group_sample(records: list[dict], split: str, total_size: int, seed: int) -> list[dict]:
    labels = sorted(ALLOWED_4WAY)
    if total_size % len(labels):
        raise ValueError("Requested split size must be divisible by four.")
    per_class = total_size // len(labels)
    chosen: list[dict] = []

    for label in labels:
        candidates = []
        seen_groups = set()
        for i, row in enumerate(records):
            if row["split"] != split or row["label_4way"] != label:
                continue
            gid = row["leakage_group_id"]
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
            candidates.append((_rank(seed, split, label, gid, i), row))
        candidates.sort(key=lambda x: x[0])
        if len(candidates) < per_class:
            raise RuntimeError(f"Not enough unique groups for {split}/{label}: need {per_class}, found {len(candidates)}")
        chosen.extend(row for _, row in candidates[:per_class])

    chosen.sort(key=lambda row: hashlib.sha256(f"{seed}|mix|{split}|{row['leakage_group_id']}|{row['prompt_hash']}".encode()).hexdigest())
    return chosen


def build_dataset_dict(records: list[dict], sizes: dict[str, int], seed: int):
    from datasets import Dataset, DatasetDict
    split_rows = {
        split: balanced_unique_group_sample(records, split, size, seed)
        for split, size in sizes.items()
    }
    result = DatasetDict({split: Dataset.from_list(rows) for split, rows in split_rows.items()})
    assert_no_group_leakage(result)
    return result


def assert_no_group_leakage(dataset) -> None:
    seen = {}
    for split, ds in dataset.items():
        for gid in ds["leakage_group_id"]:
            if gid in seen and seen[gid] != split:
                raise AssertionError(f"Leakage group {gid} appears in both {seen[gid]} and {split}")
            seen[gid] = split


def summarize(dataset) -> dict:
    out = {}
    for split, ds in dataset.items():
        out[split] = {
            "rows": len(ds),
            "unique_source_ids": len(set(ds["source_id"])),
            "unique_leakage_groups": len(set(ds["leakage_group_id"])),
            "label_4way": dict(sorted(Counter(ds["label_4way"]).items())),
            "harmfulness": dict(sorted(Counter(ds["harmfulness"]).items())),
            "style": dict(sorted(Counter(ds["style"]).items())),
        }
    return out
