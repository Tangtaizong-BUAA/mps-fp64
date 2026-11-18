"""Fetch and verify the public historical FP64 wheels without installing them."""

import argparse
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import sys
import tempfile
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "baselines.json"


class HttpsRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(newurl).scheme != "https":
            raise URLError("the release asset redirected outside HTTPS")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_wheel(path, baseline):
    asset = baseline["public_wheel"]
    if path.stat().st_size != asset["size_bytes"]:
        raise ValueError(f"{asset['filename']}: unexpected wheel size")
    # The pinned historical wheels are both below 1 MiB.
    if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
        raise ValueError(f"{asset['filename']}: archive SHA-256 mismatch")
    runtime = hashlib.sha256()
    package = hashlib.sha256()
    with ZipFile(path) as wheel:
        for name in sorted(wheel.namelist()):
            if not name.startswith("mps_fp64/") or name.endswith("/"):
                continue
            content = name.encode() + b"\0" + wheel.read(name)
            package.update(content)
            if name.endswith((".py", ".metal")):
                runtime.update(content)
    if (runtime.hexdigest(), package.hexdigest()) != (
        baseline["runtime_sha256"],
        baseline["package_tree_sha256"],
    ):
        raise ValueError(
            f"{asset['filename']}: package content disagrees with manifest"
        )


def copy_asset(baseline, output, local_assets):
    asset = baseline["public_wheel"]
    if local_assets is None:
        url = asset["url"]
        if urlsplit(url).scheme != "https":
            raise ValueError("release asset URL must use HTTPS")
        request = Request(url, headers={"User-Agent": "mps-fp64-benchmark-baselines"})
        source = build_opener(HttpsRedirect()).open(request, timeout=45)
    else:
        source = (local_assets / asset["filename"]).open("rb")
    count = 0
    with source:
        while block := source.read(65536):
            count += len(block)
            if count > asset["size_bytes"]:
                raise ValueError(
                    f"{asset['filename']}: download exceeded manifest size"
                )
            output.write(block)


def fetch(baseline, dest, local_assets=None, verify_only=False):
    asset = baseline["public_wheel"]
    name = asset["filename"]
    if Path(name).name != name:
        raise ValueError("release asset must be a filename")
    target = dest / name
    if verify_only:
        check_wheel(target, baseline)
        return "verified"
    if target.is_file():
        try:
            check_wheel(target, baseline)
            return "verified cached"
        except (ValueError, BadZipFile):
            pass  # Replace a corrupt cache entry only after its replacement verifies.
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=dest, prefix=f".{name}.", delete=False
    ) as temp:
        pending = Path(temp.name)
    try:
        with pending.open("wb") as output:
            copy_asset(baseline, output, local_assets)
        check_wheel(pending, baseline)
        os.replace(pending, target)
    finally:
        pending.unlink(missing_ok=True)
    return "verified and saved"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=ROOT / ".benchmarks",
        help="destination directory (default: checkout/.benchmarks)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--verify-only",
        action="store_true",
        help="check the cached wheels without network access",
    )
    group.add_argument(
        "--from-dir",
        type=Path,
        metavar="DIR",
        help="copy previously downloaded release assets from DIR instead of HTTPS",
    )
    manifest = json.loads(MANIFEST.read_text())
    available = {row["version"]: row for row in manifest["baselines"]}
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=sorted(available),
        default=sorted(available),
        help="which versions to fetch (default: all)",
    )
    args = parser.parse_args()
    failed = False
    for version in dict.fromkeys(args.versions):
        baseline = available[version]
        name = baseline["public_wheel"]["filename"]
        try:
            status = fetch(baseline, args.dest, args.from_dir, args.verify_only)
            print(f"{name}: {status}, sha256={baseline['public_wheel']['sha256']}")
        except (OSError, URLError, HTTPException, ValueError, BadZipFile) as exc:
            failed = True
            print(f"{name}: {exc}", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
