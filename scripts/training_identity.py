"""Compare actual prepared arrays/splits, ignoring descriptive source paths."""
import hashlib
import json
from pathlib import Path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_digest(path):
    import numpy as np

    digest = hashlib.sha256(b"matching-prepared-content-v1\0")
    with np.load(path, allow_pickle=False) as prepared:
        required = {"x", "y", "after", "groups", "row_ids", "train_idx", "val_idx", "test_idx", "metadata"}
        if not required.issubset(prepared.files):
            raise ValueError("准备数据缺少训练数组或固定划分")
        metadata = json.loads(prepared["metadata"].item())
        # Keep training-relevant configuration, while excluding source path, source
        # filename, workbook path and other descriptive metadata from the identity.
        configuration = {"schema": metadata["schema"], "config": metadata["config"],
                         "length_bounds": metadata["length_bounds"],
                         "bounds_source": metadata["bounds_source"]}
        digest.update(json.dumps(configuration, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":")).encode("utf-8"))
        for name in sorted(set(prepared.files) - {"metadata"}):
            array = prepared[name]
            if array.dtype.kind not in "biufc":
                raise ValueError(f"准备数据数组类型不支持：{name}")
            if not np.isfinite(array).all():
                raise ValueError(f"准备数据含非有限值：{name}")
            array = np.ascontiguousarray(array.astype(array.dtype.newbyteorder("<"), copy=False))
            description = json.dumps([name, array.dtype.str, list(array.shape)], separators=(",", ":"))
            digest.update(description.encode("ascii") + b"\0")
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def checkpoint_identity(checkpoint, checkpoint_path, prepared_path=None):
    """Only trust a prepared file if its entire archive matches the saved checkpoint."""
    checkpoint_path = Path(checkpoint_path).resolve()
    raw = prepared_path or checkpoint.get("training_args", {}).get("data")
    candidates = []
    if raw:
        data_path = Path(raw).expanduser()
        if data_path.is_absolute():
            candidates.append(data_path)
        else:
            source = checkpoint.get("metadata", {}).get("source")
            if source:
                # Original project convention: project/DATA/input.xlsx.
                candidates.append(Path(source).parent.parent / data_path)
            candidates.append(checkpoint_path.parent.parent.parent / data_path)
    found = []
    for candidate in dict.fromkeys(p.resolve() for p in candidates):
        if not candidate.is_file():
            continue
        found.append(candidate)
        if file_digest(candidate) != checkpoint["data_sha256"]:
            continue
        return {"method": "checkpoint_bound_prepared_content_v1", "verified": True,
                "sha256": content_digest(candidate)}
    return {"method": "checkpoint_bound_prepared_content_v1", "verified": False,
            "reason": "prepared_file_does_not_match_checkpoint" if found else "prepared_file_unavailable"}


def validate_comparison(metadata):
    """Identical saved archives suffice; differing archives require verified contents."""
    if len({m["data_sha256"] for m in metadata}) == 1:
        return "identical_prepared_archive"
    identities = [m.get("training_identity", {}) for m in metadata]
    if not all(i.get("verified") and i.get("sha256") for i in identities):
        raise ValueError("训练文件哈希不同，且无法核验其原始准备数据。请保留各训练使用的measured_matching.npz，或在模型配置填写prepared_data；无需重新训练。")
    if len({i["sha256"] for i in identities}) != 1:
        raise ValueError("所选权重的训练数组、固定划分或数据配置实际不一致，不能公平比较")
    return "identical_verified_prepared_content"
