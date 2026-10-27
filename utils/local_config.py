"""Machine-local paths for `--data_source vr_pose_pred` (RPM on VR_Pose_Pred's
AMASS trees, the majority protocol; docs: VR_Pose_Pred
docs/related_work_code/rpm.md §11).

`config/local.yaml` (gitignored; template `config/local.example.yaml`) either
names VR_Pose_Pred's own local config (`VR_POSE_PRED_LOCAL`), whose
`DATA_ROOT` / `SUPPORT_DIR` are then used, so both codebases read the same
`.pt` files and the same SMPL-H model, or sets `DATA_ROOT` / `SUPPORT_DIR`
itself (these win). Relative paths in VR_Pose_Pred's file are resolved
against that repo's root, as its own loader does.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG = REPO_ROOT / "config" / "local.yaml"


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def vr_pose_pred_paths() -> dict:
    """{"DATA_ROOT": Path, "SUPPORT_DIR": Path or None, "source": str}."""
    assert LOCAL_CONFIG.exists(), (
        f"--data_source vr_pose_pred needs {LOCAL_CONFIG} (copy config/local.example.yaml)"
    )
    cfg = _read(LOCAL_CONFIG)
    other, other_root = {}, None
    if cfg.get("VR_POSE_PRED_LOCAL"):
        other_path = Path(cfg["VR_POSE_PRED_LOCAL"])
        assert other_path.exists(), f"VR_POSE_PRED_LOCAL {other_path} does not exist"
        other = _read(other_path)
        other_root = other_path.resolve().parent.parent  # <repo>/config/local.yaml

    def resolve(key):
        if cfg.get(key):
            return Path(cfg[key]), str(LOCAL_CONFIG)
        if other.get(key):
            p = Path(other[key])
            return (p if p.is_absolute() else other_root / p), cfg["VR_POSE_PRED_LOCAL"]
        return None, None

    data_root, data_src = resolve("DATA_ROOT")
    assert data_root is not None, (
        f"DATA_ROOT is set neither in {LOCAL_CONFIG} nor in its VR_POSE_PRED_LOCAL file"
    )
    support_dir, _ = resolve("SUPPORT_DIR")
    return {"DATA_ROOT": data_root, "SUPPORT_DIR": support_dir, "source": data_src}
