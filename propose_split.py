"""Propose a group split from saved audit evidence; never replace the active manifest."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from inspect_images import ROOT
from output_utils import new_output_directory
from prepare_dataset import assign_groups, save_csv


def distribution(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    if any(len({r["split"] for r in group}) != 1 for group in groups.values()):
        raise ValueError("A related-image group crosses splits.")
    return {
        split: {
            "images": sum(r["split"] == split for r in rows),
            "group_sizes": sorted([len(g) for g in groups.values() if g[0]["split"] == split], reverse=True),
        }
        for split in ("train", "validation", "test")
    }


def build_proposal(manifest, candidates, seed=42):
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    if len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Manifest contains repeated filenames.")
    original = distribution(rows)
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["group_id"]].append(i)
    assignments = assign_groups(list(groups.values()), *(original[s]["images"] for s in ("train", "validation", "test")), seed)
    if assignments is None:
        raise ValueError("No feasible whole-group allocation with related views in both evaluation sets.")
    proposed = [dict(row, split=assignments[i][0], seed=str(seed)) for i, row in enumerate(rows)]
    by_name = {r["filename"]: r for r in proposed}
    pairs = list(csv.DictReader(candidates.open(encoding="utf-8", newline="")))
    for pair in pairs:
        if by_name[pair["filename_a"]]["split"] != by_name[pair["filename_b"]]["split"]:
            raise ValueError("Proposed split separates a saved candidate pair.")
    changes = [{"filename": old["filename"], "previous_split": old["split"], "proposed_split": new["split"]}
               for old, new in zip(rows, proposed) if old["split"] != new["split"]]
    summary = {
        "status": "proposal_only_not_applied", "seed": seed,
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source_candidates_sha256": hashlib.sha256(candidates.read_bytes()).hexdigest(),
        "current": original, "proposed": distribution(proposed),
        "changed_assignments": len(changes), "flagged_cross_split_pairs": 0,
        "candidate_pairs_checked": len(pairs), "changes": changes,
        "selection_basis": "Group membership and count constraints only; no model scores or test outcomes.",
    }
    return proposed, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset/split_manifest.csv")
    parser.add_argument("--candidates", type=Path, default=ROOT / "dataset/duplicate_candidates.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True, help="New proposal directory, not the active dataset")
    args = parser.parse_args()
    try:
        with new_output_directory(args.output, [args.manifest.parent]) as staging:
            rows, report = build_proposal(args.manifest, args.candidates, args.seed)
            save_csv(staging / "split_manifest.proposed.csv", rows, list(rows[0]))
            (staging / "proposal.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "changes"}, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Split proposal failed: {exc}\n")


if __name__ == "__main__":
    main()
