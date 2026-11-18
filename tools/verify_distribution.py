"""Check a source distribution and wheel against this checkout, without a GPU."""

import argparse
import base64
import csv
from email.parser import BytesParser
import hashlib
import io
from pathlib import Path
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_files(folder, suffixes):
    return sorted(
        path
        for path in (ROOT / folder).rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and "__pycache__" not in path.parts
    )


def verify(folder):
    (wheel,) = folder.glob("*.whl")
    (sdist,) = folder.glob("*.tar.gz")
    runtime = source_files("mps_fp64", {".py", ".metal"})
    notices = [
        ROOT / "LICENSE",
        ROOT / "THIRD_PARTY.md",
        *sorted((ROOT / "LICENSES").glob("*.txt")),
        *sorted((ROOT / "mps_fp64/vendor").glob("LICENSE-*")),
    ]

    with zipfile.ZipFile(wheel) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(set(archive.namelist()))
        (metadata,) = (
            n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
        )
        distinfo = metadata.rsplit("/", 1)[0]
        information = BytesParser().parsebytes(archive.read(metadata))
        declared = set(information.get_all("License-File"))
        expected = {p.relative_to(ROOT).as_posix() for p in notices}
        assert declared == expected, declared ^ expected
        for path in notices:
            name = path.relative_to(ROOT).as_posix()
            assert archive.read(f"{distinfo}/licenses/{name}") == path.read_bytes(), (
                name
            )
        for path in runtime:
            name = path.relative_to(ROOT).as_posix()
            assert archive.read(name) == path.read_bytes(), name
        records = list(
            csv.reader(io.StringIO(archive.read(f"{distinfo}/RECORD").decode()))
        )
        assert {row[0] for row in records} == set(archive.namelist())
        for name, encoded, length in records:
            if name == f"{distinfo}/RECORD":
                assert encoded == length == ""
                continue
            data = archive.read(name)
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode()
            )
            assert encoded == "sha256=" + digest and int(length) == len(data), name

    public_sources = [
        *runtime,
        *notices,
        *ROOT.glob("*.md"),
        ROOT / "pyproject.toml",
        ROOT / "MANIFEST.in",
        ROOT / "index.html",
        ROOT / ".nojekyll",
        *source_files("examples", {".py"}),
        *source_files("tools", {".py", ".c"}),
        *source_files("tests", {".py"}),
        *source_files("results", {".json", ".jsonl", ".csv", ".txt"}),
        *source_files("benchmarks", {".svg", ".json", ".csv", ".md", ".css", ".js"}),
    ]
    with tarfile.open(sdist, "r:gz") as archive:
        (prefix,) = {name.split("/", 1)[0] for name in archive.getnames()}
        for path in set(public_sources):
            name = path.relative_to(ROOT).as_posix()
            member = archive.extractfile(f"{prefix}/{name}")
            assert member is not None and member.read() == path.read_bytes(), name
    print(
        f"Verified {wheel.name} and {sdist.name}, {len(runtime)} Python/Metal sources and {len(notices)} license files."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        help="directory containing one source distribution and its wheel",
    )
    verify(parser.parse_args().folder)
