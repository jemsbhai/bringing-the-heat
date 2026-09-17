"""Render exact public dataset pixel values as a PNG, with provenance.

No AI image generation, model execution, dependencies, or full dataset download.
The viewer may lag the Hub repository; the provenance records its HTTP revision
headers when supplied and the repository SHA observed immediately before/after.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
from pathlib import Path
import struct
from urllib.parse import urlencode
import zlib

from hub_inventory import read_json, request

HERE = Path(__file__).resolve().parent
REPO = "Jemsbhai/multispecqr-rgb-dataset"


def chunk(name: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))


def main() -> None:
    api = f"https://huggingface.co/api/datasets/{REPO}"
    before = read_json(api)["sha"]
    params = {"dataset": REPO, "config": "default", "split": "test", "offset": 0, "length": 1}
    url = "https://datasets-server.huggingface.co/rows?" + urlencode(params)
    response, headers = request(url)
    document = json.loads(response)
    row = document["rows"][0]["row"]
    after = read_json(api)["sha"]
    if before != after:
        raise RuntimeError("Repository changed during capture; retry for coherent provenance.")
    pixels = row["image"]
    height, width = len(pixels), len(pixels[0])
    assert height > 0 and width > 0
    assert all(len(line) == width for line in pixels)
    assert all(len(pixel) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in pixel)
               for line in pixels for pixel in line)
    # PNG color type 2 / RGB / 8 bits. Store source values without resampling.
    raw_pixels = b"".join(bytes(c for pixel in line for c in pixel) for line in pixels)
    scanlines = b"".join(b"\x00" + bytes(c for pixel in line for c in pixel) for line in pixels)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))
    assets = HERE / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    path = assets / "multispecqr-rgb-test-row0.png"
    path.write_bytes(png)
    (assets / "multispecqr-rgb-test-row0.viewer.json.gz").write_bytes(gzip.compress(response, mtime=0))
    provenance = {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": REPO, "observed_repo_sha_before_and_after": before,
        "pinned_repository_url": f"https://huggingface.co/datasets/{REPO}/tree/{before}",
        "viewer_url": f"https://huggingface.co/datasets/{REPO}/viewer/default/test?row=0",
        "viewer_api_url": url, "viewer_request": params,
        "viewer_http_revision_headers": {key: value for key, value in headers.items()
                                          if "revision" in key.lower() or "commit" in key.lower()},
        "viewer_caveat": "Viewer is a cached current-branch service; observed repo SHA is not proof of a pinned-row response.",
        "row_index": document["rows"][0]["row_idx"],
        "sample_metadata": {key: row[key] for key in ("sample_id", "mode", "num_layers", "qr_version", "error_correction", "augmentation", "augmentation_params")},
        "source_dimensions": [width, height], "channels": "RGB", "dtype": "uint8",
        "pixel_transformation": "None. PNG encodes the exact image column values with no resizing or editing.",
        "pixel_sha256": hashlib.sha256(raw_pixels).hexdigest(),
        "png_sha256": hashlib.sha256(png).hexdigest(),
        "viewer_response_sha256": hashlib.sha256(response).hexdigest(),
        "license_note": "No dataset license metadata was present at capture; included for the owner's talk preparation.",
    }
    (assets / "multispecqr-rgb-test-row0.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Saved exact {width} x {height} RGB pixels: {path}")
    print(f"Sample {row['sample_id']} | {row['augmentation']} | observed repo SHA {before}")


if __name__ == "__main__":
    main()
