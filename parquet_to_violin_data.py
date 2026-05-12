"""
Extract embedded images from VIOLIN_v2/violin-test.parquet into a local data/ tree.

Place this script next to the VIOLIN_v2 folder (same parent directory).

Usage:
  python parquet_to_violin_data.py
  python parquet_to_violin_data.py --data-test          # VIOLIN_v2/data_test, max 10 files per subfolder
  python parquet_to_violin_data.py --skip-jsonl         # do not regenerate data/test.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _get_task(row: dict) -> int:
    if row.get("task") is not None:
        return int(row["task"])
    if row.get("variation") is not None:
        return int(row["variation"])
    raise KeyError("Row has neither 'task' nor 'variation'")


def _id_numeric_suffix(sample_id: Any) -> str:
    s = str(sample_id).strip()
    if s.startswith("id_"):
        return s[3:]
    return s


def _image_to_bytes(val: Any) -> Optional[bytes]:
    if val is None:
        return None
    if isinstance(val, dict):
        b = val.get("bytes")
        if b:
            return bytes(b) if not isinstance(b, bytes) else b
        return None
    try:
        from PIL import Image

        if isinstance(val, Image.Image):
            fmt = (val.format or "PNG").upper()
            buf = io.BytesIO()
            save_fmt = "JPEG" if fmt == "JPEG" else ("PNG" if fmt not in ("JPEG", "JPG") else "JPEG")
            if save_fmt == "JPEG" and val.mode in ("RGBA", "P"):
                val = val.convert("RGB")
            val.save(buf, format=save_fmt)
            return buf.getvalue()
    except ImportError:
        pass
    return None


def _image_extension_for_bytes_pil(val: Any) -> str:
    """Prefer .jpg for JPEG, else .png (for heuristic paths)."""
    if isinstance(val, dict):
        return ".png"
    try:
        from PIL import Image

        if isinstance(val, Image.Image):
            fmt = (val.format or "").upper()
            if fmt in ("JPEG", "JPG"):
                return ".jpg"
    except ImportError:
        pass
    return ".png"


def _ground_truth_rel_path(task: int, id_suffix: str, mask_type: str) -> str:
    if task == 1:
        return f"Task_Color_Var1/id_{id_suffix}.png"
    if task == 2:
        return f"Task_Color_Var2/id_{id_suffix}.png"
    if task == 3:
        return f"Task_Geometric/id_{id_suffix}.png"
    if task == 4:
        mt = (mask_type or "").strip() or "inpainting"
        return f"Task_Image_Mask/{mt}/id_{id_suffix}.png"
    raise ValueError(f"Unknown task {task}")


def _mask_raw_image1_rel(image_id: str, ext: str) -> str:
    return f"Task_Image_Mask_raw_image/images/{image_id}{ext}"


def _mask_raw_image2_rel(mask_type: str, image_id: str) -> str:
    mt = (mask_type or "").strip() or "inpainting"
    return f"Task_Image_Mask_raw_image/{mt}_mask/{image_id}.png"


def _folder_key_for_rel(rel_path: str) -> str:
    """Group by logical subfolder for --data-test caps."""
    parts = Path(rel_path).parts
    if not parts:
        return ""
    if len(parts) >= 2 and parts[0] == "Task_Image_Mask":
        return str(Path(parts[0]) / parts[1])
    if len(parts) >= 2 and parts[0] == "Task_Image_Mask_raw_image":
        return str(Path(parts[0]) / parts[1])
    return parts[0]


class TestModeLimiter:
    def __init__(self, enabled: bool, per_folder: int = 10):
        self.enabled = enabled
        self.per_folder = per_folder
        self._seen: dict[str, set[str]] = defaultdict(set)

    def allow(self, rel_path: str) -> bool:
        if not self.enabled:
            return True
        key = _folder_key_for_rel(rel_path)
        s = self._seen[key]
        if rel_path in s:
            return True
        if len(s) >= self.per_folder:
            return False
        s.add(rel_path)
        return True


def _write_bytes(out_root: Path, rel_path: str, data: bytes, limiter: TestModeLimiter) -> bool:
    rel_path = rel_path.replace("\\", "/")
    if not limiter.allow(rel_path):
        return False
    full = out_root / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(data)
    return True


def extract_parquet(
    parquet_path: Path,
    out_root: Path,
    *,
    data_test: bool,
    per_folder_limit: int = 10,
) -> None:
    ds = Dataset.from_parquet(str(parquet_path))
    limiter = TestModeLimiter(enabled=data_test, per_folder=per_folder_limit)

    n_gt = 0
    n_img1 = 0
    n_img2 = 0
    skipped = 0

    for i in range(len(ds)):
        row = ds[i]
        try:
            task = _get_task(row)
        except KeyError:
            skipped += 1
            continue

        id_suffix = _id_numeric_suffix(row.get("id", ""))
        mask_type = str(row.get("mask_type") or "")
        image_id = str(row.get("image_id") or "").strip()

        gt_rel = _ground_truth_rel_path(task, id_suffix, mask_type)
        gt_bytes = _image_to_bytes(row.get("ground_truth"))
        if gt_bytes:
            if _write_bytes(out_root, gt_rel, gt_bytes, limiter):
                n_gt += 1

        if task == 4 and image_id:
            img1_val = row.get("image1_path")
            img2_val = row.get("image2_path")
            ext1 = _image_extension_for_bytes_pil(img1_val)
            rel1 = _mask_raw_image1_rel(image_id, ext1)
            rel2 = _mask_raw_image2_rel(mask_type, image_id)

            b1 = _image_to_bytes(img1_val)
            b2 = _image_to_bytes(img2_val)
            if b1:
                if _write_bytes(out_root, rel1, b1, limiter):
                    n_img1 += 1
            if b2:
                if _write_bytes(out_root, rel2, b2, limiter):
                    n_img2 += 1

    print(
        f"Done. out_root={out_root}  ground_truth writes: {n_gt}, "
        f"image1: {n_img1}, image2: {n_img2}, rows skipped (no task): {skipped}"
    )
    if data_test:
        print(f"Test mode: at most {per_folder_limit} distinct files per subfolder under {out_root.name}/.")


def run_make_jsonl(violin_root: Path) -> None:
    """Call VIOLIN_v2/scripts/make_jsonl.py build_metadata() with cwd = violin_root (uses metadata/*.csv)."""
    script_path = violin_root / "scripts" / "make_jsonl.py"
    if not script_path.is_file():
        print(f"Skipping make_jsonl: not found {script_path}")
        return

    spec = importlib.util.spec_from_file_location("violin_make_jsonl", script_path)
    if spec is None or spec.loader is None:
        print("Skipping make_jsonl: could not load module spec")
        return
    mod = importlib.util.module_from_spec(spec)
    prev_cwd = os.getcwd()
    try:
        os.chdir(violin_root)
        sys.path.insert(0, str(violin_root / "scripts"))
        spec.loader.exec_module(mod)
        mod.build_metadata()
    finally:
        os.chdir(prev_cwd)
        try:
            sys.path.remove(str(violin_root / "scripts"))
        except ValueError:
            pass


def main() -> None:
    root = _repo_root()
    violin = root / "Violin"
    parser = argparse.ArgumentParser(description="Restore VIOLIN parquet images to a local data/ tree.")
    parser.add_argument(
        "--violin-root",
        type=Path,
        default=violin,
        help="Path to VIOLIN_v2 folder (default: sibling of this script).",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Path to violin-test.parquet (default: <violin-root>/violin-test.parquet).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory (default: <violin-root>/data, or data_test if --data-test).",
    )
    parser.add_argument(
        "--data-test",
        action="store_true",
        help="Write to <violin-root>/data_test and cap each subfolder to --per-folder files (default 10).",
    )
    parser.add_argument(
        "--per-folder",
        type=int,
        default=10,
        help="With --data-test: max distinct image paths per subfolder (default: 10).",
    )
    parser.add_argument(
        "--skip-jsonl",
        action="store_true",
        help="Do not run make_jsonl.build_metadata() after extract (default: run when out-root is .../data).",
    )
    args = parser.parse_args()

    violin_root = args.violin_root.resolve()
    parquet_path = args.parquet.resolve() if args.parquet else (violin_root / "violin-test.parquet")
    if not parquet_path.is_file():
        raise SystemExit(f"Parquet not found: {parquet_path}")

    if args.out_root is not None:
        out_root = args.out_root.resolve()
    elif args.data_test:
        out_root = violin_root / "data_test"
    else:
        out_root = violin_root / "data"

    extract_parquet(
        parquet_path,
        out_root,
        data_test=args.data_test,
        per_folder_limit=max(1, args.per_folder),
    )

    data_dir = (violin_root / "data").resolve()
    if args.skip_jsonl:
        print("Skipped make_jsonl (--skip-jsonl).")
    elif out_root.resolve() != data_dir:
        print(
            "Skipped make_jsonl: output is not <violin-root>/data; "
            "make_jsonl paths assume images under data/. Re-run without --data-test or copy files to data/ first."
        )
    else:
        print("Running make_jsonl.build_metadata() ...")
        run_make_jsonl(violin_root)


if __name__ == "__main__":
    main()
