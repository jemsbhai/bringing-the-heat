"""Read-only public Hub inventory. Standard library only; never loads model weights.

    python showcase/hub_inventory.py --offline
    python showcase/hub_inventory.py --refresh

The refresh command writes local metadata only. It uses no token, changes no remote
repository, and downloads no model weights or Parquet shards.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

AUTHOR = "Jemsbhai"  # Canonical casing matters to the author filter.
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "hub_manifest.json"


def request(url: str) -> tuple[bytes, dict]:
    req = Request(url, headers={"User-Agent": "hf-talk-public-inventory/1.0"})
    with urlopen(req, timeout=30) as response:
        return response.read(), dict(response.headers)


def read_json(url: str) -> dict | list:
    return json.loads(request(url)[0])


def list_public(kind: str) -> list[dict]:
    url = f"https://huggingface.co/api/{kind}?" + urlencode(
        {"author": AUTHOR, "limit": 100, "full": "true"}
    )
    result = []
    while url:
        payload, headers = request(url)
        result.extend(json.loads(payload))
        links = headers.get("Link", headers.get("link", ""))
        url = next((part.split("<", 1)[1].split(">", 1)[0]
                    for part in links.split(",") if 'rel="next"' in part), "")
    return [item for item in result if not item.get("private", False)]


def repo_record(kind: str, repo_id: str) -> dict:
    api_url = f"https://huggingface.co/api/{kind}/{repo_id}"
    info = read_json(api_url)
    prefix = "datasets/" if kind == "datasets" else ""
    base = f"https://huggingface.co/{prefix}{repo_id}"
    sha = info["sha"]
    record = {
        "id": repo_id, "kind": kind, "sha": sha,
        "last_modified": info.get("lastModified"),
        "library_name": info.get("library_name"),
        "card_metadata": info.get("cardData"),
        "files": [entry["rfilename"] for entry in info.get("siblings", [])],
        "repo_url": base, "api_url": api_url,
        "pinned_tree_url": f"{base}/tree/{sha}",
        "commits_url": f"{base}/commits/main",
    }
    if "README.md" in record["files"]:
        card_url = f"{base}/resolve/{sha}/README.md"
        raw = request(card_url)[0]
        record["card"] = {"url": card_url, "sha256": hashlib.sha256(raw).hexdigest(),
                          "text": raw.decode("utf-8")}
    else:
        record["card"] = None
    if kind == "datasets":
        size_url = "https://datasets-server.huggingface.co/size?" + urlencode({"dataset": repo_id})
        record["viewer_size_url"] = size_url
        try:
            record["viewer_size"] = read_json(size_url)
        except (HTTPError, TimeoutError, OSError) as exc:
            record["viewer_size_error"] = str(exc)
    return record


def refresh() -> dict:
    records = []
    for kind in ("models", "datasets"):
        for item in list_public(kind):
            records.append(repo_record(kind, item["id"]))
    commit = read_json("https://api.github.com/repos/jemsbhai/multispecqr/commits/main")
    sha = commit["sha"]
    source = {"repo": "https://github.com/jemsbhai/multispecqr", "sha": sha,
              "commit_url": commit["html_url"], "files": {}}
    for path in ("README.md", "src/multispecqr/ml_decoder.py", "LICENSE.txt"):
        url = f"https://raw.githubusercontent.com/jemsbhai/multispecqr/{sha}/{path}"
        raw = request(url)[0]
        source["files"][path] = {
            "raw_url": url,
            "view_url": f"https://github.com/jemsbhai/multispecqr/blob/{sha}/{path}",
            "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
        }
    return {
        "schema_version": 1,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "author": AUTHOR, "profile_url": f"https://huggingface.co/{AUTHOR}",
        "scope": "Unauthenticated public metadata; no model weights loaded or downloaded.",
        "viewer_note": "Dataset viewer counts describe its current cached view, not a pinned revision request.",
        "repositories": records, "source": source,
    }


def summarize(manifest: dict, mode: str) -> None:
    print(f"{manifest['author']} | {mode} | captured {manifest['captured_at_utc']}")
    for repo in manifest["repositories"]:
        detail = repo.get("library_name") or "dataset"
        splits = repo.get("viewer_size", {}).get("size", {}).get("splits", [])
        if splits:
            detail = ", ".join(f"{s['split']}={s['num_rows']:,}" for s in splits)
        print(f"  {repo['id']}\n    {repo['sha']} | {detail}")
    print(f"\nSource commit: {manifest['source']['sha']}")
    print("Model weights loaded: 0 | remote writes: 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--refresh", action="store_true", help="Fetch public metadata and save local manifest.")
    choice.add_argument("--offline", action="store_true", help="Read the saved manifest; no network. (Default)")
    args = parser.parse_args()
    if args.refresh:
        manifest = refresh()
        HERE.mkdir(parents=True, exist_ok=True)
        temporary = MANIFEST.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(MANIFEST)
        summarize(manifest, "live metadata")
    else:
        if not MANIFEST.exists():
            parser.error("No saved manifest; run once with --refresh while online.")
        summarize(json.loads(MANIFEST.read_text(encoding="utf-8")), "OFFLINE SNAPSHOT")


if __name__ == "__main__":
    main()
