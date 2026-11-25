"""Compare a reproduced proposal run to the archived numerical reference."""
import argparse
import csv
import json
import math
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "reference"
RTOL = 1e-8
ATOL = 1e-12


def read_table(path, keys):
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(set(columns)) != len(columns) or not set(keys).issubset(columns):
            raise ValueError(f"{path.name}: invalid header")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}: malformed row {reader.line_num}")
            key = tuple(row[name] for name in keys)
            if key in rows:
                raise ValueError(f"{path.name}: duplicate key {key}")
            rows[key] = row
    return columns, rows


def compare_table(directory, name, keys, numeric):
    expected_columns, expected = read_table(REFERENCE / name, keys)
    columns, actual = read_table(directory / name, keys)
    if columns != expected_columns or expected.keys() != actual.keys():
        raise ValueError(f"{name}: columns or row keys differ from reference")
    for key, record in actual.items():
        for column in columns:
            observed = record[column]
            target = expected[key][column]
            if column in numeric:
                value = float(observed)
                matches = math.isfinite(value) and math.isclose(
                    value, float(target), rel_tol=RTOL, abs_tol=ATOL
                )
            else:
                matches = observed == target
            if not matches:
                raise ValueError(f"{name}, {key}, {column}: expected {target}, found {observed}")
    return actual


def verify(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8-sig"))
    if config != json.loads((REFERENCE / "config.json").read_text(encoding="utf-8")):
        raise ValueError("The run configuration differs from the proposal reference.")
    manifest = compare_table(directory, "asset_manifest.csv", ("asset",), set())
    measures = compare_table(directory, "measures.csv", ("asset", "policy", "step"), {
        "amplitude", "detail_length_change_nrmse", "protected_max", "shared_edge_gap_rms",
        "protected_boundary_gap_max", "absolute_displaced_error_from_rest_rms",
    })
    contours = compare_table(directory, "contours.csv", ("asset", "fit"), {
        "amplitude", "target_x", "displaced_extent_absolute_error",
    })
    for asset in config["assets"]:
        name = asset + ".npz"
        with np.load(REFERENCE / name, allow_pickle=False) as expected, np.load(
            directory / name, allow_pickle=False
        ) as actual:
            if set(actual.files) != set(expected.files):
                raise ValueError(f"{name}: fields differ from reference")
            for field in expected.files:
                a, b = actual[field], expected[field]
                if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
                    raise ValueError(f"{name}:{field} shape or type differs from reference")
                if np.issubdtype(b.dtype, np.floating):
                    matches = np.isfinite(a).all() and np.allclose(a, b, rtol=RTOL, atol=ATOL)
                else:
                    matches = np.array_equal(a, b)
                if not matches:
                    raise ValueError(f"{name}:{field} differs from reference")
    return len(manifest), len(measures), len(contours)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=ROOT / "runs" / "pilot")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if directory == REFERENCE:
        parser.error("Choose a reproduced run to compare against reference/.")
    try:
        assets, measurements, fits = verify(directory)
    except (OSError, KeyError, ValueError, EOFError, BadZipFile) as error:
        parser.exit(1, f"Verification failed: {error}\n")
    print(f"PASS: {assets} assets and manifests, {measurements} measurements, {fits} contour fits.")
    print(f"Numeric tolerances: relative {RTOL:g}, absolute {ATOL:g}.")


if __name__ == "__main__":
    main()
