import csv
import math
import matplotlib
matplotlib.use("Agg")           # headless
import matplotlib.pyplot as plt
import os

def _read_training_log_csv(path):
    """Read your training_log.csv into a dict of lists."""
    if not os.path.exists(path):
        return None
    metrics = {"epoch": [], "miou": [], "pix_acc": [], "elev_mae": [], "elev_rmse": [], "bf_score": []}
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # epoch,miou,pix_acc,elev_mae,elev_rmse,bf_score
        for row in reader:
            if not row or len(row) < 6: 
                continue
            e, miou, acc, mae, rmse, bf = row
            metrics["epoch"].append(int(e))
            metrics["miou"].append(float(miou))
            metrics["pix_acc"].append(float(acc))
            metrics["elev_mae"].append(float(mae))
            metrics["elev_rmse"].append(float(rmse))
            metrics["bf_score"].append(float(bf))
    return metrics

def _plot_metric(x, ys, labels, title, ylabel, out_path):
    plt.figure(figsize=(8,5))
    for y, lbl in zip(ys, labels):
        plt.plot(x, y, label=lbl)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if len(ys) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _save_training_plots(log_csv_path, out_dir, lr_hist=None):
    """Generate a few helpful plots from training_log.csv (+ optional LR curve)."""
    M = _read_training_log_csv(log_csv_path)
    if not M or not M["epoch"]:
        return

    x = M["epoch"]

    # 1) Semantics
    _plot_metric(x, [M["miou"]], ["mIoU"], 
                 "Semantic mIoU vs Epoch", "mIoU", os.path.join(out_dir, "plot_miou.png"))
    _plot_metric(x, [M["pix_acc"]], ["Pixel Acc"], 
                 "Pixel Accuracy vs Epoch", "Accuracy", os.path.join(out_dir, "plot_pixacc.png"))
    _plot_metric(x, [M["bf_score"]], ["BF Score"], 
                 "Boundary F1 vs Epoch", "BF Score", os.path.join(out_dir, "plot_bfscore.png"))

    # 2) Elevation
    _plot_metric(x, [M["elev_mae"]], ["MAE"], 
                 "Elevation MAE vs Epoch", "MAE (meters)", os.path.join(out_dir, "plot_elev_mae.png"))
    _plot_metric(x, [M["elev_rmse"]], ["RMSE"], 
                 "Elevation RMSE vs Epoch", "RMSE (meters)", os.path.join(out_dir, "plot_elev_rmse.png"))

    # 3) Combined elevation (nice overview)
    _plot_metric(x, [M["elev_mae"], M["elev_rmse"]], ["MAE", "RMSE"], 
                 "Elevation Error vs Epoch", "Error (meters)", os.path.join(out_dir, "plot_elev_mae_rmse.png"))

    # 4) Learning rate curve (if provided)
    if lr_hist is not None and len(lr_hist) == len(x):
        _plot_metric(x, [lr_hist], ["lr"], 
                     "Learning Rate vs Epoch", "LR", os.path.join(out_dir, "plot_lr.png"))