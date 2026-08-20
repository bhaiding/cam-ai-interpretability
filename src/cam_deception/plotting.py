from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def save_auroc_heatmap(metrics_df, title: str, output_path: Path | None = None):
    pivot = metrics_df.pivot(index="train_scope", columns="eval_scope", values="auroc")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0,
                cbar_kws={"label": "AUROC"}, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Evaluated On")
    ax.set_ylabel("Trained On")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig, ax
