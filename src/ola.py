"""R2: per-disease out-of-lung-localization (OLL) screen.

OLL(image, label) = attribution_outside_mask(gradcam(image,label), lung_mask(image)),
i.e. the fraction of positive Grad-CAM attribution falling outside the lung field.
This is a LOCALIZATION statistic (where the model looks), not a reliance/causal
claim. Reported stratified by ground truth and with image-row bootstrap CIs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.interpretability import attribution_outside_mask
from src.utils import set_seed

MIN_N = 10  # ranking gate: need >=10 pos AND >=10 neg to rank a disease


def stratify_oll(rows, threshold: float = 0.5) -> dict:
    """Split per-row OLL values into truth/error strata."""
    y1, y0, fp, fn, allv = [], [], [], [], []
    for r in rows:
        v = float(r["oll"])
        yt = int(r["y_true"])
        yp = float(r["y_pred"])
        allv.append(v)
        if yt == 1:
            y1.append(v)
            if yp < threshold:
                fn.append(v)
        elif yt == 0:
            y0.append(v)
            if yp >= threshold:
                fp.append(v)
    return {"all": allv, "y1": y1, "y0": y0, "fp": fp, "fn": fn}


def bootstrap_mean_ci(values, n_boot: int = 2000, seed: int = 229, alpha: float = 0.05):
    """Mean + percentile CI, resampling rows (the image is the unit)."""
    arr = np.asarray(values, dtype="float64")
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype="float64")
    for b in range(n_boot):
        idx = rng.integers(0, arr.size, arr.size)
        boots[b] = arr[idx].mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (float(arr.mean()), lo, hi)


def is_degenerate_mask(mask) -> bool:
    """True if a lung mask is empty (all-False / all-zero).

    ``attribution_outside_mask`` returns 1.0 for ANY CAM when the mask is empty
    (all attribution is "outside" a mask that covers nothing), so a segmentation
    failure on a frontal view would silently contribute a spurious OLL of 1.0 and
    inflate the per-disease mean. Such images are skipped and counted, not scored.

    Only the empty (all-False / all-zero) case is treated as degenerate. An
    all-True mask is the opposite failure (it *deflates* OLL toward 0 rather than
    inflating it) and is left in; ``lung_mask`` always returns a boolean array, so
    NaN inputs do not arise here.
    """
    return not np.asarray(mask).astype(bool).any()


def is_frontal_path(path: str) -> bool:
    """Heuristic: a CheXpert path is a frontal view unless it is marked lateral.

    The xrv PSPNet lung segmenter is trained on frontal CXRs and produces empty
    (0%-coverage) masks on lateral views, which would make every lateral score a
    spurious OLL of 1.0. We restrict the OLL screen to frontal views and report
    the lateral-segmenter failure as a limitation. CheXpert names lateral files
    ``view2_lateral.png`` (frontals are ``view1_frontal.png``).
    """
    return "lateral" not in str(path).lower()


def compute_oll_rows(predictions, label, image_root, seg_model, model, device="cpu",
                     frontal_only=True, method="gradcam"):
    """Per-image OLL for one label over scorable rows (y_true in {0,1}).

    frontal_only=True drops lateral views, whose lung masks are invalid (see
    is_frontal_path). Set False only to reproduce the confounded all-views run.

    method selects the attribution map: "gradcam" (default) or "ig" (Integrated
    Gradients). Both return a (224,224) map consumed identically by
    attribution_outside_mask, so OLL can be cross-checked across methods.
    """
    import torch

    from src.data import default_xrv_transform, load_xray_image
    from src.saliency import (
        compute_gradcam, compute_ig, find_path_column, pred_column, true_column,
        resolve_image_path,
    )
    from src.segmentation import lung_mask

    if method not in ("gradcam", "ig"):
        raise ValueError(f"Unknown attribution method: {method!r} (expected 'gradcam' or 'ig').")
    attribution_fn = compute_ig if method == "ig" else compute_gradcam

    path_col = find_path_column(predictions.columns)
    frame = predictions[[path_col, true_column(label), pred_column(label)]].copy()
    frame.columns = ["path", "y_true", "y_pred"]
    frame = frame[frame["y_true"].isin((0.0, 1.0))]
    if frontal_only:
        frame = frame[frame["path"].map(is_frontal_path)]

    rows = []
    n_empty_mask = 0
    for rec in frame.itertuples(index=False):
        p = resolve_image_path(Path(image_root), str(rec.path))
        img = default_xrv_transform()(load_xray_image(p))           # (1,224,224)
        mask = lung_mask(seg_model, img, device=device)             # (224,224) bool
        if is_degenerate_mask(mask):
            # Empty lung mask → attribution_outside_mask would return a spurious
            # 1.0 for any CAM. Skip and count rather than inflate the OLL mean.
            n_empty_mask += 1
            continue
        tensor = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
        cam = attribution_fn(model, tensor, label)                   # (224,224)
        oll = attribution_outside_mask(cam, mask)
        rows.append({"y_true": int(rec.y_true), "y_pred": float(rec.y_pred), "oll": oll})
    return rows, n_empty_mask


def summarize_label(label, rows, n_boot=2000, seed=229, n_empty_mask=0):
    strata = stratify_oll(rows)
    n_pos, n_neg = len(strata["y1"]), len(strata["y0"])
    mean, lo, hi = bootstrap_mean_ci(strata["y1"], n_boot=n_boot, seed=seed)  # primary = y_true==1
    return {
        "label": label, "oll_pos_mean": mean, "oll_pos_lo": lo, "oll_pos_hi": hi,
        "oll_neg_mean": float(np.mean(strata["y0"])) if n_neg else float("nan"),
        "oll_fp_mean": float(np.mean(strata["fp"])) if strata["fp"] else float("nan"),
        "oll_fn_mean": float(np.mean(strata["fn"])) if strata["fn"] else float("nan"),
        "n_pos": n_pos, "n_neg": n_neg, "n_empty_mask": n_empty_mask, "n_boot": n_boot,
        "ranked": bool(n_pos >= MIN_N and n_neg >= MIN_N),
    }


def main():
    ap = argparse.ArgumentParser(description="R2: per-disease out-of-lung localization screen.")
    ap.add_argument("--predictions-csv", type=Path, default=Path("results/predictions_val.csv"))
    ap.add_argument("--image-root", type=Path, default=Path("data/chexpert"))
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    ap.add_argument("--num-labels", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument(
        "--include-laterals", action="store_true",
        help="Include lateral views (default: frontal-only; lateral lung masks are invalid).",
    )
    ap.add_argument(
        "--method", default="gradcam", choices=["gradcam", "ig"],
        help="Attribution method for OLL: gradcam (default) or ig (Integrated Gradients).",
    )
    args = ap.parse_args()

    from src.models import build_model
    from src.saliency import compute_label_aurocs, lowest_auroc_labels
    from src.segmentation import load_lung_segmenter

    set_seed(229)
    preds = pd.read_csv(args.predictions_csv)
    labels = lowest_auroc_labels(compute_label_aurocs(preds), args.num_labels)
    model = build_model("densenet121")
    model = model.to(args.device)
    model.eval()
    seg = load_lung_segmenter(device=args.device)

    summary = []
    for label in labels:
        rows, n_empty_mask = compute_oll_rows(
            preds, label, args.image_root, seg, model, device=args.device,
            frontal_only=not args.include_laterals, method=args.method,
        )
        if n_empty_mask:
            print(f"[{label}] skipped {n_empty_mask} image(s) with empty lung masks")
        summary.append(
            summarize_label(label, rows, n_boot=args.n_boot, n_empty_mask=n_empty_mask)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary).sort_values("oll_pos_mean", ascending=False)
    suffix = "" if args.method == "gradcam" else f"_{args.method}"
    out = args.output_dir / f"oll_by_disease{suffix}.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ranked = df[df["ranked"]]
    if not ranked.empty:
        yerr = [ranked["oll_pos_mean"] - ranked["oll_pos_lo"],
                ranked["oll_pos_hi"] - ranked["oll_pos_mean"]]
        plt.figure()
        plt.bar(ranked["label"], ranked["oll_pos_mean"], yerr=yerr, capsize=4)
        plt.ylabel("Out-of-lung attribution fraction (y_true=1)")
        plt.title("Per-disease out-of-lung localization screen")
        plt.tight_layout()
        fig = args.output_dir / "figures" / "oll_by_disease.png"
        fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig)
        plt.close()
        print(f"Wrote {fig}")


if __name__ == "__main__":
    main()
