import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt

from pose_enc_utils import pose_encoding_to_extri_intri
from test_co3d import se3_to_relative_pose_error


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def compute_baseline(tvec_1: torch.Tensor, tvec_2: torch.Tensor) -> float:
    """
    Baseline between the two camera translations (OpenCV convention):
        baseline = ||t2 - t1||_2

    Handles tvec shapes like (3,1), (1,3), (3,), etc.
    """
    t1 = tvec_1.reshape(-1).double()
    t2 = tvec_2.reshape(-1).double()
    return float(torch.linalg.norm(t2 - t1, ord=2).item())


def compute_errors_vs_samples_for_sequence(
        sequence: str,
        pred_root: str,
        gt_root: str,
        num_total_passes: int = 20,
        sample_counts_to_track: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Loads GT (two frames) and predictions (multi-pass), then computes
    relative rotation/translation errors as you average over n=1..num_total_passes.

    Returns a dict with:
      - samples: [1..num_total_passes]
      - rel_r_errs_deg: list[float]
      - rel_t_errs_deg: list[float]
      - baseline: float
      - tracked_points: { "1": {...}, "10": {...}, "20": {...} } (if requested)
    """
    if sample_counts_to_track is None:
        sample_counts_to_track = [1, 10, 20]

    # --- Load GT camera poses ---
    cam_frame_1 = torch.load(
        os.path.join(gt_root, sequence, "cam_poses_gt", "frame_0000.pt"),
        map_location="cpu",
    )
    cam_frame_2 = torch.load(
        os.path.join(gt_root, sequence, "cam_poses_gt", "frame_0001.pt"),
        map_location="cpu",
    )

    R_1 = cam_frame_1["R"]
    tvec_1 = cam_frame_1["tvec"]
    R_2 = cam_frame_2["R"]
    tvec_2 = cam_frame_2["tvec"]

    baseline = compute_baseline(tvec_1, tvec_2)

    # Build GT extrinsics stack (2, 3, 4)
    extri_opencv_frame_1 = torch.hstack((R_1.squeeze(), tvec_1.T))
    extri_opencv_frame_2 = torch.hstack((R_2.squeeze(), tvec_2.T))
    gt_extri = torch.stack([extri_opencv_frame_1, extri_opencv_frame_2], dim=0)

    # Add homogeneous row for se3 (2, 4, 4)
    add_row = torch.tensor([0, 0, 0, 1], dtype=gt_extri.dtype).expand(gt_extri.size(0), 1, 4)
    gt_se3 = torch.cat((gt_extri, add_row), dim=1)

    # --- Accumulate predictions across passes ---
    predictions_mult_passes = []
    samples: List[int] = []
    rel_r_errs: List[float] = []
    rel_t_errs: List[float] = []

    tracked_points: Dict[str, Dict[str, float]] = {}

    for i in range(num_total_passes):
        pred_path = os.path.join(pred_root, sequence, f"vggt_predictions_{i}.pt")
        pred_pack = torch.load(pred_path, map_location="cpu")
        pose_enc = pred_pack["pose_enc"]  # shape like (K, D) or (1, D) etc.
        predictions_mult_passes.append(pose_enc)

        n = i + 1
        predictions = torch.cat(predictions_mult_passes, dim=0).mean(dim=0, keepdim=True)

        extrinsic, _ = pose_encoding_to_extri_intri(predictions, build_intrinsics=False)
        pred_extrinsic = extrinsic[0]  # (2, 3, 4) expected if two views

        add_row_pred = torch.tensor([0, 0, 0, 1], dtype=pred_extrinsic.dtype).expand(
            pred_extrinsic.size(0), 1, 4
        )
        pred_se3 = torch.cat((pred_extrinsic, add_row_pred), dim=1)

        rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, 2)

        samples.append(n)
        rel_r_errs.append(_to_float(rel_rangle_deg))
        rel_t_errs.append(_to_float(rel_tangle_deg))

        if n in sample_counts_to_track:
            tracked_points[str(n)] = {
                "rel_rangle_deg": rel_r_errs[-1],
                "rel_tangle_deg": rel_t_errs[-1],
            }

        print(
            f"[{sequence}] Pass {n}: rel_rangle_deg = {rel_r_errs[-1]:.4f}, "
            f"rel_tangle_deg = {rel_t_errs[-1]:.4f}, baseline = {baseline:.6f}"
        )

    return {
        "sequence": sequence,
        "baseline": baseline,
        "samples": samples,
        "rel_r_errs_deg": rel_r_errs,
        "rel_t_errs_deg": rel_t_errs,
        "tracked_points": tracked_points,
    }


def plot_per_sequence_errors(metrics_seq: Dict[str, Any], out_png: Path) -> None:
    samples = metrics_seq["samples"]
    r = metrics_seq["rel_r_errs_deg"]
    t = metrics_seq["rel_t_errs_deg"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(samples, r, marker="o", label="Rotation")
    ax.plot(samples, t, marker="o", label="Translation")
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Error (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title(f"{metrics_seq['sequence']} — Effect of #Samples on Camera Pose Error")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def aggregate_across_sequences(all_seq_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    For each n, compute mean/std across sequences for rotation and translation errors.
    Assumes all sequences have the same sample list length.
    """
    samples = all_seq_metrics[0]["samples"]
    r_mat = np.array([m["rel_r_errs_deg"] for m in all_seq_metrics], dtype=np.float64)  # (S, N)
    t_mat = np.array([m["rel_t_errs_deg"] for m in all_seq_metrics], dtype=np.float64)

    r_mean = r_mat.mean(axis=0).tolist()
    r_std = r_mat.std(axis=0, ddof=0).tolist()
    t_mean = t_mat.mean(axis=0).tolist()
    t_std = t_mat.std(axis=0, ddof=0).tolist()

    return {
        "samples": samples,
        "rotation_mean_deg": r_mean,
        "rotation_std_deg": r_std,
        "translation_mean_deg": t_mean,
        "translation_std_deg": t_std,
        "num_sequences": len(all_seq_metrics),
    }


def plot_average_vs_samples(agg: Dict[str, Any], out_png: Path) -> None:
    samples = np.array(agg["samples"], dtype=np.int32)

    r_mean = np.array(agg["rotation_mean_deg"], dtype=np.float64)
    r_std = np.array(agg["rotation_std_deg"], dtype=np.float64)

    t_mean = np.array(agg["translation_mean_deg"], dtype=np.float64)
    t_std = np.array(agg["translation_std_deg"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(samples, r_mean, marker="o", label="Rotation mean")
    ax.fill_between(samples, r_mean - r_std, r_mean + r_std, alpha=0.2)

    ax.plot(samples, t_mean, marker="o", label="Translation mean")
    ax.fill_between(samples, t_mean - t_std, t_mean + t_std, alpha=0.2)

    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Error (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_title("Average Effect of #Samples on Camera Pose Error (mean ± std across sequences)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_error_vs_baseline(
        all_seq_metrics: List[Dict[str, Any]],
        out_png: Path,
        sample_counts: List[int] = [1, 10, 20],
) -> Dict[str, Any]:
    """
    Creates baseline-sorted curves for errors at specific sample-counts.
    Returns the extracted arrays too (so they can be stored to JSON).
    """
    # Sort sequences by baseline
    seq_sorted = sorted(all_seq_metrics, key=lambda m: m["baseline"])
    baselines = np.array([m["baseline"] for m in seq_sorted], dtype=np.float64)
    sequences = [m["sequence"] for m in seq_sorted]

    # Map sample count -> index into samples list
    samples_list = seq_sorted[0]["samples"]
    sample_to_idx = {n: samples_list.index(n) for n in sample_counts if n in samples_list}

    rot_curves: Dict[str, List[float]] = {}
    trans_curves: Dict[str, List[float]] = {}

    for n, idx in sample_to_idx.items():
        rot_curves[str(n)] = [m["rel_r_errs_deg"][idx] for m in seq_sorted]
        trans_curves[str(n)] = [m["rel_t_errs_deg"][idx] for m in seq_sorted]

    # Plot: 2 rows, shared x
    fig, (ax_r, ax_t) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for n in sample_to_idx.keys():
        ax_r.plot(baselines, rot_curves[str(n)], marker="o", label=f"n={n}")
    ax_r.set_ylabel("Rotation error (deg)")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="upper right")
    ax_r.set_title("Pose Error vs Baseline (sorted by baseline)")

    for n in sample_to_idx.keys():
        ax_t.plot(baselines, trans_curves[str(n)], marker="o", label=f"n={n}")
    ax_t.set_xlabel("Baseline  ||t2 - t1||2")
    ax_t.set_ylabel("Translation error (deg)")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    return {
        "baselines_sorted": baselines.tolist(),
        "sequences_sorted": sequences,
        "rotation_error_deg_at_n": rot_curves,
        "translation_error_deg_at_n": trans_curves,
        "sample_counts": list(sample_to_idx.keys()),
    }


def save_metrics_json(metrics: Dict[str, Any], out_json: Path) -> None:
    _ensure_dir(out_json.parent)
    with out_json.open("w") as f:
        json.dump(metrics, f, indent=2)


def load_metrics_json(in_json: Path) -> Dict[str, Any]:
    with in_json.open("r") as f:
        return json.load(f)


def recreate_plots_from_json(metrics_json_path: str, out_dir: str) -> None:
    """
    Utility: regenerate all plots without recomputing errors.
    """
    metrics = load_metrics_json(Path(metrics_json_path))
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Per-sequence plots
    for seq_m in metrics["per_sequence"]:
        seq = seq_m["sequence"]
        seq_dir = out_dir / seq
        _ensure_dir(seq_dir)
        plot_per_sequence_errors(seq_m, seq_dir / "pose_error_and_variance_vs_samples.png")

    # Average plot
    plot_average_vs_samples(metrics["aggregate"], out_dir / "average_pose_error_and_variance_vs_samples.png")

    # Baseline plot
    # (We can reuse stored extracted curves if you prefer; here we replot from stored baseline_payload.)
    bp = metrics["baseline_payload"]
    baselines = np.array(bp["baselines_sorted"], dtype=np.float64)

    fig, (ax_r, ax_t) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for n in bp["sample_counts"]:
        ax_r.plot(baselines, bp["rotation_error_deg_at_n"][str(n)], marker="o", label=f"n={n}")
    ax_r.set_ylabel("Rotation error (deg)")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="upper right")
    ax_r.set_title("Pose Error vs Baseline (sorted by baseline)")

    for n in bp["sample_counts"]:
        ax_t.plot(baselines, bp["translation_error_deg_at_n"][str(n)], marker="o", label=f"n={n}")
    ax_t.set_xlabel("Baseline  ||t2 - t1||2")
    ax_t.set_ylabel("Translation error (deg)")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_dir / "pose_error_vs_baseline.png")
    plt.close(fig)


def run_uco3d_pose_sampling_analysis(
        sequences: List[str],
        pred_root: str,
        gt_root: str,
        out_root: str = "UCO3D_DEBUG_ANALYSIS",
        num_total_passes: int = 20,
        sample_counts_for_baseline_plot: List[int] = [1, 10, 20],
) -> Dict[str, Any]:
    """
    Main entry point that:
      - computes per-sequence errors vs samples,
      - saves per-seq plots under: out_root/{sequence}/pose_error_and_variance_vs_samples.png
      - saves average plot: out_root/average_pose_error_and_variance_vs_samples.png
      - saves baseline plot: out_root/pose_error_vs_baseline.png
      - stores everything into: out_root/metrics.json

    Returns the full metrics dict.
    """
    out_root = Path(out_root)
    _ensure_dir(out_root)

    per_sequence: List[Dict[str, Any]] = []

    for sequence in sequences:
        seq_metrics = compute_errors_vs_samples_for_sequence(
            sequence=sequence,
            pred_root=pred_root,
            gt_root=gt_root,
            num_total_passes=num_total_passes,
            sample_counts_to_track=sample_counts_for_baseline_plot,
        )
        per_sequence.append(seq_metrics)

        # per-seq plot
        seq_dir = out_root / sequence
        _ensure_dir(seq_dir)
        plot_per_sequence_errors(seq_metrics, seq_dir / "pose_error_and_variance_vs_samples.png")

    # aggregate mean/std across sequences
    agg = aggregate_across_sequences(per_sequence)
    plot_average_vs_samples(agg, out_root / "average_pose_error_and_variance_vs_samples.png")

    # baseline plot (for n in {1,10,20})
    baseline_payload = plot_error_vs_baseline(
        per_sequence,
        out_root / "pose_error_vs_baseline.png",
        sample_counts=sample_counts_for_baseline_plot,
    )

    # save JSON
    metrics = {
        "config": {
            "pred_root": pred_root,
            "gt_root": gt_root,
            "out_root": str(out_root),
            "num_total_passes": num_total_passes,
            "sample_counts_for_baseline_plot": sample_counts_for_baseline_plot,
        },
        "per_sequence": per_sequence,
        "aggregate": agg,
        "baseline_payload": baseline_payload,
    }
    save_metrics_json(metrics, out_root / "metrics.json")

    return metrics


if __name__ == "__main__":
    sequences_all = [
        '1-89973-52840', '1-90320-26441', '1-95979-13606', '10-81738-74119', '117-3686-24815',
        '12-10308-28681', '13-68075-18110', '156-6699-92799', '16-11679-97735', '20-38390-68010',
        '2017-26879-34905', '21-1781-33291', '26-14460-30571', '267-59948-42658', '27-79013-34290',
        '27-92237-65486', '28006-14019-9638', '3-9266-65507', '305-68822-66727', '33-11430-17759',
        '38-75852-49792', '4-91634-39388', '4-91751-93241', '401-36913-71936', '411-19840-28276',
        '42033-32228-40100', '47433-58842-22217', '47480-57073-46870', '475-99485-99578', '494-29795-17364',
        '533-16958-69462', '538-38847-906', '54207-48862-40002', '57-41030-21122', '60458-35176-64843',
        '605-31775-92635', '62387-12826-25112', '65154-22082-61695', '654-18524-38608', '69-40182-31015',
        '7-88671-19551', '70-1164-44120', '732-27981-57173', '767-14445-38072', '8-20919-19870'
    ]

    sequences_good = [
        '1-89973-52840', '1-90320-26441', '1-95979-13606', '10-81738-74119', '117-3686-24815', '156-6699-92799',
        '21-1781-33291', '26-14460-30571', '27-79013-34290', '28006-14019-9638', '3-9266-65507', '33-11430-17759',
        '38-75852-49792', '401-36913-71936', '411-19840-28276', '47480-57073-46870', '475-99485-99578',
        '54207-48862-40002', '57-41030-21122', '60458-35176-64843', '62387-12826-25112', '65154-22082-61695',
        '7-88671-19551', '732-27981-57173', '767-14445-38072',
    ]

    sequences_bad = []
    for sequence in sequences_all:
        if sequence not in sequences_good:
            sequences_bad.append(sequence)
    # Set your roots

    pred_root = "/Users/aleksandrekandelaki/git/dl4sai/vggt/small_baseline_std_1"
    gt_root = "/Users/aleksandrekandelaki/git/private/3d/uco3d/uco3d/random_2_view_samples"

    run_uco3d_pose_sampling_analysis(
        sequences=sequences_bad,
        pred_root=pred_root,
        gt_root=gt_root,
        out_root="UCO3D_DEBUG_ANALYSIS_BAD_IMAGES",
        num_total_passes=20,
        sample_counts_for_baseline_plot=[1, 10, 20],
    )

    # If later you want to recreate plots only:
    # recreate_plots_from_json("UCO3D_DEBUG_ANALYSIS/metrics.json", "UCO3D_DEBUG_ANALYSIS")
