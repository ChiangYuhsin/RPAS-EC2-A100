"""Materialize and hash the EC-1 HumanEval provenance bundle.

The files are deliberately downloaded outside the repository's tracked tree.
The script records hashes and source revisions but never adds benchmark data to
Git, because the upstream licenses and redistribution terms differ.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path


OFFICIAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
AFLOW_ARCHIVE_URL = "https://drive.google.com/uc?export=download&id=1DNoegtZiUhWtvkd2xoIuElmIi4ah7k8e"
AFLOW_MIRROR_BASE = "https://raw.githubusercontent.com/CitrusYL/AgentSlimming/0bb1afc677e3751e09dc535e373f0316b0a8369f/data/datasets/"
OFFICIAL_GZ_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
AFLOW_COMMIT = "0bb1afc677e3751e09dc535e373f0316b0a8369f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, path.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def fetch(output_dir: Path, *, skip_archive: bool = False) -> dict:
    official = output_dir / "official"
    aflow = output_dir / "aflow"
    official.mkdir(parents=True, exist_ok=True)
    aflow.mkdir(parents=True, exist_ok=True)
    gz_path = official / "HumanEval.jsonl.gz"
    if not gz_path.exists():
        download(OFFICIAL_URL, gz_path)
    actual_gz_sha = sha256(gz_path)
    if actual_gz_sha != OFFICIAL_GZ_SHA256:
        raise RuntimeError(f"official HumanEval gzip hash mismatch: {actual_gz_sha}")
    jsonl_path = official / "humaneval.jsonl"
    if not jsonl_path.exists():
        with gzip.open(gz_path, "rb") as source, jsonl_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    archive = aflow / "aflow_data.tar.gz"
    if not skip_archive and not archive.exists():
        download(AFLOW_ARCHIVE_URL, archive)
    extracted: dict[str, str] = {}
    if archive.exists():
        with tarfile.open(archive, "r:gz") as tar:
            names = {member.name.rsplit("/", 1)[-1]: member for member in tar.getmembers()}
            for filename in ("humaneval_validate.jsonl", "humaneval_test.jsonl", "humaneval_public_test.jsonl"):
                member = names.get(filename)
                if member is None:
                    continue
                target = aflow / filename
                with tar.extractfile(member) as source, target.open("wb") as handle:
                    assert source is not None
                    shutil.copyfileobj(source, handle)
                extracted[filename] = str(target)
    # The mirror is a deterministic fallback for the public-test fixture.
    public_test = aflow / "humaneval_public_test.jsonl"
    if not public_test.exists():
        download(AFLOW_MIRROR_BASE + public_test.name, public_test)
        extracted[public_test.name] = str(public_test)
    manifest = {
        "benchmark": "HumanEval",
        "official": {"upstream": "openai/human-eval", "version": "1.0.0", "artifact": str(gz_path), "sha256": actual_gz_sha, "derived_jsonl_sha256": sha256(jsonl_path), "tasks": sum(1 for _ in jsonl_path.open(encoding="utf-8"))},
        "aflow": {"archive_url": AFLOW_ARCHIVE_URL, "mirror_commit": AFLOW_COMMIT, "files": {}},
    }
    for name in ("humaneval_validate.jsonl", "humaneval_test.jsonl", "humaneval_public_test.jsonl"):
        path = aflow / name
        if path.exists():
            manifest["aflow"]["files"][name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    (output_dir / "DATASET_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/ec1_humaneval")
    parser.add_argument("--skip-aflow-archive", action="store_true")
    args = parser.parse_args()
    print(json.dumps(fetch(Path(args.output_dir), skip_archive=args.skip_aflow_archive), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
