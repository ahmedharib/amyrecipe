"""Generate architecture diagram (R6) and benchmark charts (R4) directly from benchmark_results.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def _load_benchmarks() -> dict:
    return json.loads(Path("docs/benchmark_results.json").read_text(encoding="utf-8"))


def draw_architecture_diagram(out_path: Path = Path("docs/architecture.png")) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bench = _load_benchmarks()
    v_rows = {r["model"]: r for r in bench["vision_results"]}
    r_rows = {r["model"]: r for r in bench["retrieval_results"]}

    vq = v_rows["ResNet18 Quantized (ONNX)"]
    vs = v_rows["ScratchCNN (4-block)"]
    vf = v_rows["ResNet18 Fine-Tuned"]

    rq = r_rows["Quantized MiniLM (ONNX)"]
    rt = r_rows["Regex + TF-IDF Cosine"]
    rn = r_rows["EmbeddingBag + MLP (Trained)"]
    rf = r_rows["Fine-Tuned MiniLM (BERT)"]

    fig, ax = plt.subplots(figsize=(15.5, 8.5), dpi=220)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.5)
    ax.axis("off")
    fig.patch.set_facecolor("#F9FAFB")

    def box(x, y, w, h, title, body, bg="#FFFFFF", border="#4200FF", title_color="#111114", lw=2.0):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            facecolor=bg, edgecolor=border, linewidth=lw,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.25, title, fontsize=10.2, fontweight="bold",
                color=title_color, ha="center", va="top")
        ax.text(x + w / 2, y + (h - 0.5) / 2, body, fontsize=8.4,
                color="#374151", ha="center", va="center", linespacing=1.35)

    def arrow(x1, y1, x2, y2, label="", dy=0.18):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.26,head_length=0.4", color="#4200FF", lw=1.8),
        )
        if label:
            ax.text(
                (x1 + x2) / 2, (y1 + y2) / 2 + dy, label,
                fontsize=7.8, fontweight="bold", color="#4200FF", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#F9FAFB", edgecolor="none"),
            )

    ax.text(
        7.75, 8.1,
        "Food Photo to Recipe Finder — System Architecture (R6 · Local CPU Runtime)",
        fontsize=14, fontweight="bold", color="#111114", ha="center",
    )

    box(
        0.3, 2.7, 2.5, 3.1,
        "React Frontend UI\n(frontend/src/App.jsx)",
        "• Photo Upload + 10 Samples\n• Free-Text Request Box\n• Side-by-Side Grad-CAM\n• Top-5 Stacked Rows + Modal\n• 2-Table Model Comparison\n(Served on :8000)",
        bg="#FFFFFF", border="#111114", lw=2.0,
    )

    box(
        3.7, 3.0, 2.1, 2.5,
        "FastAPI Backend\n(backend/main.py)",
        "• POST /api/predict\n• GET /api/samples\n• GET /api/health\n• StaticFiles (/dist)",
        bg="#F0EBFF", border="#4200FF",
    )
    arrow(2.85, 4.55, 3.65, 4.55, "HTTP Form")
    arrow(3.65, 3.95, 2.85, 3.95, "JSON + PNG", dy=-0.22)

    box(
        6.7, 5.4, 2.3, 1.9,
        "Vision Preprocessing\n(src/vision/dataset.py)",
        "• PIL RGB Convert\n• Resize(256) + CenterCrop(224)\n• ImageNet Normalize\n• Tensor [1, 3, 224, 224]",
        bg="#FFFFFF", border="#6B7280", lw=1.5,
    )
    arrow(5.85, 4.9, 6.65, 6.25, "Image")

    box(
        10.0, 5.1, 5.1, 2.4,
        "Area A & D · Vision Models (CPUExecutionProvider)",
        f"★ PRIMARY: ResNet18 Quantized (resnet18_int8.onnx · INT8)\n"
        f"   Top-1: {vq['top1_acc']*100:.2f}% | Top-5: {vq['top5_acc']*100:.2f}% | {vq['size_mb']} MB | {vq['latency_ms']} ms\n"
        "• Explainability: Grad-CAM Hook on layer4[-1] → Heatmap PNG\n"
        "• Comparison Ladder:\n"
        f"   1) ScratchCNN 4-Block (FP32 · {vs['top1_acc']*100:.2f}% · {vs['size_mb']} MB · {vs['latency_ms']} ms)\n"
        f"   2) Fine-Tuned ResNet18 (FP32 · {vf['top1_acc']*100:.2f}% · {vf['size_mb']} MB · {vf['latency_ms']} ms)",
        bg="#F0EBFF", border="#4200FF", lw=2.3,
    )
    arrow(9.05, 6.35, 9.95, 6.35, "Tensor")

    box(
        6.7, 1.2, 2.3, 2.3,
        "Area B · Classical NLP\n(src/retrieval/preprocess.py)",
        "• Lowercase & Clean Noise\n• Regex Constraint Parser:\n"
        "   max_time_min (≤ N min)\n"
        "   diet_tags (vegan, gf...)\n"
        "• PorterStemmer Tokens",
        bg="#FFFFFF", border="#6B7280", lw=1.5,
    )
    arrow(5.85, 3.6, 6.65, 2.35, "Query", dy=-0.22)
    arrow(12.2, 5.05, 7.85, 3.55, "Predicted Dish (Hard Filter)")

    box(
        10.0, 0.8, 5.1, 3.1,
        "Area A, B, C, D · Retrieval Ladder (213 Recipes Index)",
        f"★ PRIMARY: Quantized MiniLM BERT (bert_int8.onnx · INT8)\n"
        f"   Recall@5: {rq['recall_at_5']:.4f} | MRR: {rq['mrr']:.4f} | {rq['size_mb']} MB | {rq['latency_ms']} ms\n"
        "• Post-Processor: Hard Dish Filter + Time/Diet Penalties\n"
        "• Comparison Ladder:\n"
        f"   1) Classical: Regex + TF-IDF Cosine ({rt['latency_ms']} ms)\n"
        f"   2) Neural: Trained EmbeddingBag + MLP (FP32 · {rn['latency_ms']} ms)\n"
        f"   3) Transformer: Fine-Tuned all-MiniLM-L6-v2 (FP32 · {rf['latency_ms']} ms)",
        bg="#FFFFFF", border="#4200FF", lw=2.0,
    )
    arrow(9.05, 2.35, 9.95, 2.35, "Parsed")

    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"Saved architecture diagram to {out_path}")


def draw_benchmark_chart(out_path: Path = Path("docs/benchmark_chart.png")) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bench = _load_benchmarks()
    v_rows = bench["vision_results"]
    r_rows = bench["retrieval_results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=200)
    fig.patch.set_facecolor("#F9FAFB")

    v_models = ["ScratchCNN\n(FP32)", "ResNet18\nFine-Tuned (FP32)", "ResNet18\nQuantized (INT8)"]
    v_acc = [r["top1_acc"] * 100 for r in v_rows]
    v_lat = [r["latency_ms"] for r in v_rows]
    v_colors = ["#9CA3AF", "#7C4DFF", "#4200FF"]

    bars1 = ax1.bar(v_models, v_acc, color=v_colors, width=0.52)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Top-1 Accuracy (%)", fontweight="bold")
    ax1.set_title("Vision Ladder (2,500 Held-Out Test Images)", fontweight="bold", pad=12)
    for bar, acc, lat in zip(bars1, v_acc, v_lat):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{acc:.2f}%\n({lat} ms)", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    r_models = ["Regex +\nTF-IDF", "Trained\nEmbBag+MLP", "Fine-Tuned\nMiniLM (FP32)", "Quantized\nMiniLM (INT8)"]
    r_mrr = [r["mrr"] * 100 for r in r_rows]
    r_lat = [r["latency_ms"] for r in r_rows]
    r_colors = ["#9CA3AF", "#F59E0B", "#7C4DFF", "#4200FF"]

    bars2 = ax2.bar(r_models, r_mrr, color=r_colors, width=0.52)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("MRR × 100 (%)", fontweight="bold")
    ax2.set_title("Retrieval Ladder (46 Held-Out Test Queries)", fontweight="bold", pad=12)
    for bar, mrr, lat in zip(bars2, r_mrr, r_lat):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"MRR {mrr:.2f}%\n({lat} ms)", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved benchmark chart to {out_path}")


if __name__ == "__main__":
    draw_architecture_diagram()
    draw_benchmark_chart()
