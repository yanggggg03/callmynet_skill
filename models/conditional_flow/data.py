"""Measured reflection data ingestion. All artifacts use explicit units and splits."""

import csv
import hashlib
import json
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCHEMA = "measured-double-stub-v1"
REQUIRED = ("Mag0", "Deg0", "L1", "L2", "matched_mag", "matched_deg")
ALIASES = {
    "mag0": "Mag0",
    "mag": "Mag0",
    "deg0": "Deg0",
    "deg": "Deg0",
    "l1": "L1",
    "l2": "L2",
    "matched_mag": "matched_mag",
    "match_mag": "matched_mag",
    "matched_deg": "matched_deg",
    "match_deg": "matched_deg",
    "group_id": "group_id",
}
DEFAULT_CONFIG = {
    "magnitude_unit": "linear",
    "phase_unit": "deg",
    "length_unit": "deg",
    "length_bounds": None,
    "seed": 42,
    "input_group_decimals": 6,
    "quality_power_scale": 0.05,
    "quality_weight_floor": 0.05,
    "inverse_max_matched_mag": None,
}


def local_path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else HERE / p


def load_config(path=None):
    cfg = dict(DEFAULT_CONFIG)
    path = local_path(path or "config.json")
    if path.exists():
        supplied = json.loads(path.read_text(encoding="utf-8-sig"))
        unknown = set(supplied) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        cfg.update(supplied)
    if cfg["magnitude_unit"] not in ("linear", "db") or cfg["phase_unit"] != "deg":
        raise ValueError("magnitude_unit must be linear/db; phase_unit must be deg")
    if cfg["length_unit"] not in ("deg", "rad", "lambda"):
        raise ValueError("length_unit must be deg/rad/lambda")
    if not 0 < cfg["quality_weight_floor"] <= 1 or cfg["quality_power_scale"] <= 0:
        raise ValueError("Invalid quality weight parameters")
    if (
        not isinstance(cfg["input_group_decimals"], int)
        or not 0 <= cfg["input_group_decimals"] <= 12
    ):
        raise ValueError("input_group_decimals must be an integer from 0 to 12")
    threshold = cfg["inverse_max_matched_mag"]
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("inverse_max_matched_mag is a LINEAR threshold in [0,1]")
    return cfg


def to_complex(mag, phase, magnitude_unit="linear"):
    mag = np.asarray(mag, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    amplitude = 10 ** (mag / 20) if magnitude_unit == "db" else mag
    if not np.isfinite(amplitude).all() or not np.isfinite(phase).all():
        raise ValueError("Nonfinite reflection magnitude or phase")
    if (amplitude < 0).any() or (amplitude > 1 + 1e-6).any():
        raise ValueError(
            "Expected passive reflection magnitudes in [0,1]; verify linear/db setting"
        )
    amplitude = np.clip(amplitude, 0, 1)
    angle = np.deg2rad(phase)
    return np.stack(
        (amplitude * np.cos(angle), amplitude * np.sin(angle)), axis=-1
    ).astype(np.float32)


def _xlsx_rows(path, sheet=None):
    """Read ordinary XLSX using standard OOXML; formulas require saved cached values."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            strings = ["".join(item.itertext()) for item in root]
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = workbook.find("m:sheets", ns)
        selected = next(
            (s for s in sheets if sheet is None or s.attrib["name"] == sheet), None
        )
        if selected is None:
            raise ValueError(f"Worksheet not found: {sheet}")
        relations = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rid = selected.attrib[f"{{{rns}}}id"]
        target = next(r.attrib["Target"] for r in relations if r.attrib["Id"] == rid)
        target = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath("xl/" + target)
        )
        root = ET.fromstring(z.read(target))
        for row in root.findall("m:sheetData/m:row", ns):
            cells = {}
            for c in row:
                reference = c.attrib.get("r", "")
                column = 0
                for letter in reference:
                    if not letter.isalpha():
                        break
                    column = column * 26 + ord(letter.upper()) - ord("A") + 1
                val = c.find("m:v", ns)
                if c.attrib.get("t") == "inlineStr":
                    value = "".join(t.text or "" for t in c.findall(".//m:t", ns))
                elif c.attrib.get("t") == "s" and val is not None:
                    value = strings[int(val.text)]
                else:
                    value = val.text if val is not None else ""
                if column:
                    cells[column - 1] = value
            if cells:
                yield [cells.get(i, "") for i in range(max(cells) + 1)]


def read_records(path, sheet=None, prediction=False):
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        rows = list(_xlsx_rows(path, sheet))
    elif path.suffix.lower() in (".csv", ".tsv"):
        raw = path.read_bytes()
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw.decode("gb18030")
        try:
            dialect = csv.Sniffer().sniff(content[:8192], delimiters=",;\t")
        except csv.Error:
            raise ValueError(
                "Cannot identify table separator; use comma/tab separated columns"
            ) from None
        rows = list(csv.reader(content.splitlines(), dialect))
    else:
        raise ValueError("Use .xlsx or .csv/.tsv; save old .xls as .xlsx first")
    if not rows:
        raise ValueError("Empty workbook/table")
    headers = [ALIASES.get(str(v).strip().lower(), str(v).strip()) for v in rows[0]]
    needed = REQUIRED[:2] if prediction else REQUIRED
    if any(headers.count(k) != 1 for k in needed):
        raise ValueError(
            f"Header row must contain each column once: {needed}; got {headers}"
        )
    records, row_ids = [], []
    for line, row in enumerate(rows[1:], 2):
        if all(not str(c).strip() for c in row):
            continue
        record = dict(zip(headers, row))
        for key in needed:
            try:
                value = float(record[key])
            except (ValueError, KeyError, TypeError):
                raise ValueError(
                    f"Invalid/missing {key} at source row {line}; recalculate Excel formulas before saving"
                ) from None
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite {key} at source row {line}")
            record[key] = value
        records.append(record)
        row_ids.append(line)
    if not records:
        raise ValueError("No data rows")
    return records, np.array(row_ids, dtype=np.int64)


def group_partition(x, records, cfg):
    """Union identical rounded inputs AND user group IDs before group-level splitting."""
    parents = list(range(len(x)))

    def root(a):
        while parents[a] != a:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a

    seen = {}
    for i, (value, record) in enumerate(
        zip(np.round(x, cfg["input_group_decimals"]), records)
    ):
        # Collapse signed zero; tuples have numeric equality.
        keys = [("input", *value.tolist())]
        if str(record.get("group_id", "")).strip():
            keys.append(("experiment", str(record["group_id"]).strip()))
        for key in keys:
            if key in seen:
                parents[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups = np.array([root(i) for i in range(len(x))])
    unique = np.unique(groups)
    if len(unique) < 10:
        raise ValueError(
            "Need at least 10 independent input/experiment groups for train/validation/test"
        )
    rng = np.random.default_rng(cfg["seed"])
    unique = rng.permutation(unique)
    ntrain = max(1, int(0.7 * len(unique)))
    nval = max(1, int(0.15 * len(unique)))
    split_groups = (
        unique[:ntrain],
        unique[ntrain : ntrain + nval],
        unique[ntrain + nval :],
    )
    return groups, [np.flatnonzero(np.isin(groups, g)) for g in split_groups]


def prepare(source, output, config=None, sheet=None):
    cfg = load_config(config)
    records, row_ids = read_records(source, sheet)
    x = to_complex(
        [r["Mag0"] for r in records],
        [r["Deg0"] for r in records],
        cfg["magnitude_unit"],
    )
    after = to_complex(
        [r["matched_mag"] for r in records],
        [r["matched_deg"] for r in records],
        cfg["magnitude_unit"],
    )
    y = np.array([[r["L1"], r["L2"]] for r in records], dtype=np.float32)
    groups, splits = group_partition(x, records, cfg)
    supplied = cfg["length_bounds"]
    if supplied is not None:
        bounds = np.asarray(supplied, dtype=np.float32)
        if (
            bounds.shape != (2, 2)
            or not np.isfinite(bounds).all()
            or (bounds[:, 1] <= bounds[:, 0]).any()
        ):
            raise ValueError("length_bounds must be [[L1_min,L1_max],[L2_min,L2_max]]")
        if ((y < bounds[:, 0]) | (y > bounds[:, 1])).any():
            raise ValueError("Recorded lengths exceed configured actuator bounds")
        bounds_source = "user_device_limits"
    else:
        # Never use validation/test targets to fit a transformation or range.
        lower, upper = y[splits[0]].min(0), y[splits[0]].max(0)
        if (upper - lower <= 1e-8).any():
            raise ValueError(
                "Training data must vary both lengths; set meaningful device bounds and collect varied data"
            )
        pad = (upper - lower) * 0.02
        bounds = np.stack([lower - pad, upper + pad], -1)
        bounds_source = "training_observed_range_plus_2pct_not_hardware_limits"
    metadata = {
        "schema": SCHEMA,
        "config": cfg,
        "source": str(Path(source).resolve()),
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "sheet": sheet,
        "length_bounds": bounds.tolist(),
        "bounds_source": bounds_source,
        "independent_groups": len(np.unique(groups)),
        "rows": len(x),
        "assumption": "fixed apparatus/frequency and initial state, or common load reference plane",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        y=y,
        after=after,
        groups=groups,
        row_ids=row_ids,
        train_idx=splits[0],
        val_idx=splits[1],
        test_idx=splits[2],
        metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
    )
    print(
        json.dumps(
            {
                "saved": str(output.resolve()),
                "rows": len(x),
                "split_rows": [len(s) for s in splits],
                "length_unit": cfg["length_unit"],
                "bounds_source": bounds_source,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return output


def load(path):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    required = {
        "x",
        "y",
        "after",
        "groups",
        "row_ids",
        "train_idx",
        "val_idx",
        "test_idx",
        "metadata",
    }
    if not required.issubset(d):
        raise ValueError(
            "Not a prepared measured dataset. Run common.py on the six-column measured table"
        )
    meta = json.loads(d["metadata"].item())
    if meta.get("schema") != SCHEMA:
        raise ValueError(
            "This is an old nine-input analytic dataset. Run common.py on the six-column measured table"
        )
    n = len(d["x"])
    for k in ("x", "y", "after"):
        if d[k].shape != (n, 2) or not np.isfinite(d[k]).all():
            raise ValueError(f"Invalid {k} array")
    ids = [d[k + "_idx"] for k in ("train", "val", "test")]
    if any(not len(i) or i.dtype.kind not in "iu" for i in ids) or not np.array_equal(
        np.sort(np.concatenate(ids)), np.arange(n)
    ):
        raise ValueError("Invalid partition")
    gs = [set(d["groups"][i].tolist()) for i in ids]
    if gs[0] & gs[1] or gs[0] & gs[2] or gs[1] & gs[2]:
        raise ValueError("Group leakage across splits")
    return d, meta


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
