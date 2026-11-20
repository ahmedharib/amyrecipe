"""Generate the 10-slide 16:9 Presentation PDF (`docs/SWE015_Presentation.pdf`) directly from benchmark_results.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _new_slide(section_label: str, title: str, slide_num: int, total_slides: int = 10):
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=150)
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    fig.patch.set_facecolor("#F9FAFB")

    ax.text(0.7, 6.92, section_label.upper(), fontsize=9.5, fontweight="bold", color="#4200FF")
    ax.text(0.7, 6.45, title, fontsize=20, fontweight="bold", color="#111114")
    ax.plot([0.7, 12.63], [6.18, 6.18], color="#E5E7EB", lw=1.2)

    ax.text(0.7, 0.35, "SWE015 – Introduction to Large Language Models · Use Case 3: Food Photo to Recipe Finder",
            fontsize=8.5, color="#6B7280")
    ax.text(12.63, 0.35, f"Slide {slide_num} / {total_slides}",
            fontsize=8.5, fontweight="bold", color="#4200FF", ha="right")
    return fig, ax


def _card(ax, x, y, w, h, title="", lines=None, bg="#FFFFFF", border="#E5E7EB", title_color="#4200FF", fontsize=9.3):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.16",
        facecolor=bg, edgecolor=border, linewidth=1.6,
    )
    ax.add_patch(rect)
    curr_y = y + h - 0.22
    if title:
        ax.text(x + 0.22, curr_y, title, fontsize=11.2, fontweight="bold", color=title_color, va="top")
        curr_y -= 0.34
    if lines:
        for ln in lines:
            if ln == "":
                curr_y -= 0.14
                continue
            ax.text(x + 0.22, curr_y, ln, fontsize=fontsize, color="#1F2937", va="top", linespacing=1.25)
            curr_y -= 0.28 * (ln.count("\n") + 1)


def build_presentation_pdf(
    out_pdf: Path = Path("docs/SWE015_Presentation.pdf"),
    student_name: str = "Ahmed [Update Name]",
    student_id: str = "[Update Student ID]",
    repo_url: str = "https://github.com/<your-username>/swe015-food-recipe-finder",
    video_url: str = "https://youtu.be/<your-2min-demo-link>",
) -> None:
    bench = json.loads(Path("docs/benchmark_results.json").read_text(encoding="utf-8"))
    v_map = {r["model"]: r for r in bench["vision_results"]}
    r_map = {r["model"]: r for r in bench["retrieval_results"]}

    vs = v_map["ScratchCNN (4-block)"]
    vf = v_map["ResNet18 Fine-Tuned"]
    vq = v_map["ResNet18 Quantized (ONNX)"]

    rt = r_map["Regex + TF-IDF Cosine"]
    rn = r_map["EmbeddingBag + MLP (Trained)"]
    rf = r_map["Fine-Tuned MiniLM (BERT)"]
    rq = r_map["Quantized MiniLM (ONNX)"]

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # SLIDE 1: Cover Slide
        fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=150)
        ax.set_xlim(0, 13.33)
        ax.set_ylim(0, 7.5)
        ax.axis("off")
        fig.patch.set_facecolor("#F9FAFB")

        hero = mpatches.FancyBboxPatch(
            (0.7, 0.8), 11.93, 5.9,
            boxstyle="round,pad=0.08,rounding_size=0.35",
            facecolor="#FFFFFF", edgecolor="#4200FF", linewidth=2.4,
        )
        ax.add_patch(hero)
        ax.text(1.3, 5.95, "SWE015 · INTRODUCTION TO LARGE LANGUAGE MODELS · FALL 2026",
                fontsize=10.5, fontweight="bold", color="#4200FF")
        ax.text(1.3, 5.15, "Food Photo to Recipe Finder",
                fontsize=27, fontweight="bold", color="#111114")
        ax.text(1.3, 4.55, "Use Case 3 — Multimodal Dish Classification, Grad-CAM & Quantized Transformer Retrieval",
                fontsize=13.5, color="#4B5563")

        _card(ax, 1.3, 1.35, 5.2, 2.7, "Student & Submission Details", [
            f"• Student Name:  {student_name}",
            f"• Student ID:    {student_id}",
            "• Chosen Case:   Use Case 3 (Food Photo to Recipe Finder)",
            "• Stack:         FastAPI + React + PyTorch + ONNX Runtime",
        ], bg="#F0EBFF", border="#4200FF")

        _card(ax, 6.9, 1.35, 5.1, 2.7, "Deliverable Links (Gate 1)", [
            "• Code Repository (GitHub):",
            f"  {repo_url}",
            "• 2-Minute Narrated Demo Video:",
            f"  {video_url}",
        ], bg="#FFFFFF", border="#111114", title_color="#111114", fontsize=9.8)

        pdf.savefig(fig)
        plt.close(fig)

        # SLIDE 2: Introduction
        fig, ax = _new_slide("1. Introduction", "The Problem, Target Users & What Success Looks Like", 2)
        _card(ax, 0.7, 3.55, 5.8, 2.35, "The Real-World Problem", [
            "• Home cooks often snap a photo of a dish they want to recreate",
            "  but need recipes matching strict dietary & time constraints",
            "  (e.g., 'vegetarian, gluten-free, under 30 minutes').",
            "• Pure keyword search fails on synonyms ('sweet cocoa treat'),",
            "  while FP32 deep models are slower & heavier on CPU.",
        ])
        _card(ax, 6.83, 3.55, 5.8, 2.35, "Who It Is For", [
            "• Home cooks & meal planners with dietary restrictions",
            "  (celiac/gluten-free, vegan, vegetarian, dairy-free).",
            "• Busy users needing fast weeknight filters ('< 20 minutes')",
            "  paired with instant visual dish recognition.",
            "• CPU-only deployments requiring low latency & small footprint.",
        ])
        _card(ax, 0.7, 0.85, 11.93, 2.4, "What Success Looks Like (Use Case 3 Objectives)", [
            f"1. Vision Pipeline: Classify 10 Food-101 classes with >90% Top-1 accuracy ({vq['top1_acc']*100:.2f}% INT8), visual",
            f"   Grad-CAM heatmaps confirming food focus, and ~4x compression ({vf['size_mb']} -> {vq['size_mb']} MB) on CPU.",
            f"2. Retrieval Pipeline: Parse time/diet constraints via Regex + Porter stemming and rank Top-5 recipes with",
            f"   INT8 MiniLM achieving {rq['recall_at_5']:.4f} Recall@5, {rq['mrr']:.4f} MRR in {rq['latency_ms']} ms on CPU.",
            "3. Interactive Web App: Deliver quantized models in a live React + FastAPI app with side-by-side comparison.",
        ], bg="#F0EBFF", border="#4200FF")
        pdf.savefig(fig)
        plt.close(fig)

        # SLIDE 3: Development — Data
        fig, ax = _new_slide("2. Development · Data", "Datasets, Train/Val/Test Splits & Preprocessing", 3)
        _card(ax, 0.7, 3.45, 5.8, 2.45, "Vision Dataset: ETHZ Food-101 (10-Class Subset)", [
            "• Source: Public ETHZ Food-101 (`ethz/food101` on HF Hub).",
            "• 10 Classes: caesar_salad, chocolate_cake, hamburger, lasagna,",
            "  pad_thai, pancakes, pizza, ramen, sushi, tacos.",
            "• Total: 10,000 images (1,000 per class):",
            "  - Train: 6,000 (600/class) | Val: 1,500 (150/class)",
            "  - Held-Out Test: 2,500 official test images (250/class)",
        ])
        _card(ax, 6.83, 3.45, 5.8, 2.45, "Recipe Corpus & Held-Out Test Queries", [
            "• Recipe Corpus (`data/recipes.json`): 213 structured recipes",
            "  - 153 extracted from public `corbt/all-recipes` (HF Hub)",
            "  - 60 curated constraint-diverse domain supplements",
            "  - Fields: title, dish_category, ingredients, steps, prep_time, diet_tags",
            "• Held-Out Queries (`data/test_queries.json`): 46 test queries",
            "  with ground-truth relevant recipe IDs (constraint + synonym cases).",
        ])
        _card(ax, 0.7, 0.85, 11.93, 2.3, "Preprocessing Pipelines (Vision Augmentation & Classical NLP)", [
            "• Image Preprocessing (`src/vision/dataset.py`): Train uses RandomResizedCrop(224), RandomHorizontalFlip,",
            "  RandomRotation(15°), ColorJitter, and ImageNet normalization. Eval uses Resize(256) + CenterCrop(224).",
            "• Text Preprocessing (`src/retrieval/preprocess.py`): Lowercasing, unit normalization ('mins' -> 'minutes'),",
            "  Regex extraction of `max_time_min` ('under 30 minutes') and 6 `diet_tags` (vegetarian, vegan, gluten_free,",
            "  dairy_free, spicy, high_protein), stopword removal, and NLTK PorterStemmer tokenization.",
        ], bg="#FFFFFF", border="#111114", title_color="#111114")
        pdf.savefig(fig)
        plt.close(fig)

        # SLIDE 4: Architecture Diagram (R6)
        fig, ax = _new_slide("2. Development · Architecture (R6)", "End-to-End System Architecture & Data Flow", 4)
        arch_img_path = Path("docs/architecture.png")
        if arch_img_path.exists():
            img = mpimg.imread(str(arch_img_path))
            ax.imshow(img, extent=[0.6, 12.73, 0.75, 6.05], aspect="auto")
        pdf.savefig(fig)
        plt.close(fig)

        # SLIDE 5: Technique Map (R2)
        fig, ax = _new_slide("2. Development · Technique Map (R2)", "Mapping Required Course Techniques to Application Modules", 5)
        _card(ax, 0.7, 3.55, 5.8, 2.35, "Area A · Neural Networks & Deep Learning", [
            "• ScratchCNN (`src/vision/models.py`): 4 Conv-BN-ReLU-MaxPool",
            "  blocks + Dropout(0.4) + 2-layer classifier trained with AdamW.",
            "• Transfer Learning (`build_transfer_resnet18`): ImageNet",
            "  ResNet18 — Phase 1 feature extraction (4 ep), Phase 2",
            "  fine-tuning `layer4 + fc` (7 ep) with label smoothing.",
            "• Grad-CAM (`src/vision/gradcam.py`): Saliency on `layer4[-1]`.",
        ], bg="#F0EBFF", border="#4200FF")
        _card(ax, 6.83, 3.55, 5.8, 2.35, "Area B · Classical NLP", [
            "• Regex Constraint Parser (`src/retrieval/preprocess.py`):",
            "  Extracts numerical time limits & dietary tags from free text.",
            "• Text Normalization & Stemming: Stopword filtering +",
            "  NLTK `PorterStemmer` token normalization.",
            "• TF-IDF + Cosine Baseline (`src/retrieval/tfidf_retriever.py`):",
            "  Sublinear bigram TF-IDF matrix + cosine similarity.",
        ])
        _card(ax, 0.7, 0.85, 5.8, 2.35, "Area C · Transformers", [
            "• WordPiece Subword Tokenization & Bi-Encoder Architecture:",
            "  `sentence-transformers/all-MiniLM-L6-v2` (6-layer BERT).",
            "• Domain Fine-Tuning (`src/retrieval/bert_retriever.py`):",
            "  Fine-tuned for 2 epochs on 893 `(query, recipe)` pairs using",
            "  in-batch `MultipleNegativesRankingLoss` + mean pooling.",
        ])
        _card(ax, 6.83, 0.85, 5.8, 2.35, "Area D · Efficient Models (Mandatory Quantization)", [
            "• Vision Static INT8 (`src/vision/quantize_cnn.py`): ONNX export",
            "  + `quantize_static` (QDQ, QUInt8 activations, QInt8 weights)",
            "  calibrated on 120 Food-101 training images.",
            "• Transformer Dynamic INT8 (`src/retrieval/quantize_bert.py`):",
            "  ONNX export + `quantize_dynamic` (QInt8 linear weights).",
        ], bg="#FFFFFF", border="#111114", title_color="#111114")
        pdf.savefig(fig)
        plt.close(fig)

        # SLIDE 6: 4-Model Ladder & Quantization
        fig, ax = _new_slide("2. Development · Model Ladder (R3)", "Four-Model Progression & ONNX INT8 Quantization", 6)
        _card(ax, 0.7, 3.5, 11.93, 2.4, "The Four-Model Ladder (Evaluated on Identical Held-Out Test Splits)", [
            "1. Classical Baseline: Sublinear (1,2)-gram TF-IDF over Porter-stemmed recipe docs + Regex constraint filtering.",
            "   (Vision counterpart: 4-block ScratchCNN trained from scratch on the 6,000 training images).",
            "2. Trained Neural Network: PyTorch `NeuralDualEncoder` (`EmbeddingBag(128)` + 2-layer ReLU/Dropout MLP)",
            "   trained for 35 epochs with temperature-scaled contrastive cross-entropy loss (`tau=0.1`).",
            "   (Vision counterpart: Two-phase transfer-learned ResNet18 fine-tuned in FP32).",
            "3. Transformer (FP32): Fine-tuned `all-MiniLM-L6-v2` BERT sentence encoder with normalized 384-d embeddings.",
        ])
        _card(ax, 0.7, 0.85, 11.93, 2.35, "4. Quantized Models in Production (`onnxruntime` CPUExecutionProvider — Requirement R5)", [
            "• Why ONNX Static INT8 for ResNet18? PyTorch `quantize_dynamic` only quantizes `nn.Linear`/`LSTM`, leaving",
            "  `Conv2d` in FP32. Exporting ResNet18 to ONNX opset 18 and calibrating over 120 images with `quantize_static`",
            f"  quantizes Conv2d layers to INT8: {vf['size_mb']} MB -> {vq['size_mb']} MB ({vf['latency_ms']} -> {vq['latency_ms']} ms).",
            "• Why ONNX Dynamic INT8 for MiniLM BERT? Transformer latency is dominated by dense `MatMul`/`Gemm` layers.",
            f"  `quantize_dynamic` converts weights to `QInt8`: {rf['size_mb']} MB -> {rq['size_mb']} MB ({rf['latency_ms']} -> {rq['latency_ms']} ms).",
        ], bg="#F0EBFF", border="#4200FF")
        pdf.savefig(fig)
        plt.close(fig)

        # SLIDE 7: Results Table (R4)
        fig, ax = _new_slide("2. Development · Results Table (R4)", "Measured Benchmark Results on Held-Out Test Data", 7)
        hw = bench.get("hardware", {})
        ax.text(0.7, 5.82,
                f"Hardware: {hw.get('processor', 'Intel CPU')} ({hw.get('platform', 'Windows 11')}) · Averaged over 120 CPU runs (15 warmup)",
                fontsize=9.2, fontweight="bold", color="#374151")

        cols = ["Pipeline", "Model", "Precision", "Held-Out Task Metrics", "Size (MB)", "CPU Latency"]
        rows_data = [
            ["Vision (2.5k)", "ScratchCNN (4-block)", vs["precision"], f"Top-1: {vs['top1_acc']*100:.2f}% | Top-5: {vs['top5_acc']*100:.2f}%", f"{vs['size_mb']} MB", f"{vs['latency_ms']} ms"],
            ["Vision (2.5k)", "ResNet18 Fine-Tuned", vf["precision"], f"Top-1: {vf['top1_acc']*100:.2f}% | Top-5: {vf['top5_acc']*100:.2f}%", f"{vf['size_mb']} MB", f"{vf['latency_ms']} ms"],
            ["Vision (2.5k)", "ResNet18 Quantized", vq["precision"], f"Top-1: {vq['top1_acc']*100:.2f}% | Top-5: {vq['top5_acc']*100:.2f}%", f"{vq['size_mb']} MB", f"{vq['latency_ms']} ms"],
            ["Retrieval (46)", "Regex + TF-IDF Cosine", rt["precision"], f"R@5: {rt['recall_at_5']:.4f} | MRR: {rt['mrr']:.4f}", f"{rt['size_mb']} MB", f"{rt['latency_ms']} ms"],
            ["Retrieval (46)", "EmbBag + MLP (Trained)", rn["precision"], f"R@5: {rn['recall_at_5']:.4f} | MRR: {rn['mrr']:.4f}", f"{rn['size_mb']} MB", f"{rn['latency_ms']} ms"],
            ["Retrieval (46)", "Fine-Tuned MiniLM", rf["precision"], f"R@5: {rf['recall_at_5']:.4f} | MRR: {rf['mrr']:.4f}", f"{rf['size_mb']} MB", f"{rf['latency_ms']} ms"],
            ["Retrieval (46)", "Quantized MiniLM", rq["precision"], f"R@5: {rq['recall_at_5']:.4f} | MRR: {rq['mrr']:.4f}", f"{rq['size_mb']} MB", f"{rq['latency_ms']} ms"],
        ]

        col_widths = [0.14, 0.22, 0.09, 0.29, 0.12, 0.14]
        table = ax.table(
            cellText=rows_data,
            colLabels=cols,
            colWidths=col_widths,
            loc="center",
            bbox=[0.5 / 13.33, 1.05 / 7.5, 12.33 / 13.33, 4.5 / 7.5],
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.6)
        for (r, c), cell in table.get_celld().items():
            cell.set_width(col_widths[c])
            cell.set_edgecolor("#E5E7EB")
            if r == 0:
                cell.set_facecolor("#4200FF")
                cell.get_text().set_color("#FFFFFF")
                cell.get_text().set_fontweight("bold")
            elif r in (3, 7):
                cell.set_facecolor("#F0EBFF")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor("#FFFFFF")

        pdf.savefig(fig)
        plt.close(fig)

        # SLIDE 8: Visual Charts
        fig, ax = _new_slide("2. Development · Accuracy vs. Latency Trade-Off", "Visualizing the Model Ladder Progression", 8)
        chart_path = Path("docs/benchmark_chart.png")
        if chart_path.exists():
            img = mpimg.imread(str(chart_path))
            ax.imshow(img, extent=[0.7, 12.63, 1.0, 5.95], aspect="auto")
        pdf.savefig(fig)
        plt.close(fig)

        # SLIDE 9: Conclusion
        fig, ax = _new_slide("3. Conclusion", "Model Comparison Insights, Failure Examples & Next Steps", 9)
        _card(ax, 0.7, 3.5, 5.8, 2.4, "Where Each Model Wins & Loses", [
            f"• TF-IDF Baseline: Ultra-fast ({rt['latency_ms']} ms) and strong on exact stems",
            f"  (MRR {rt['mrr']:.4f}), but fails on paraphrases with zero lexical overlap.",
            f"• Trained EmbeddingBag+MLP: Fastest neural run ({rn['latency_ms']} ms) but",
            f"  limited by small vocabulary trained only on 213 recipes ({rn['mrr']:.4f}).",
            f"• Quantized MiniLM (INT8): Best overall — {rq['hit_at_5']*100:.1f}% Hit@5, {rq['mrr']:.4f} MRR",
            f"  in {rq['latency_ms']} ms, retaining FP32 ranking quality at ~1/4th the size.",
        ])
        _card(ax, 6.83, 3.5, 5.8, 2.4, "Concrete Failure Examples (Measured on Test Set)", [
            "1. Classical NLP Failure (`test_queries.json` q_045):",
            "   Query: 'dairy free sweet cocoa treat' (no dish hint).",
            "   TF-IDF misses Chocolate Cake due to zero token overlap",
            "   ('cocoa treat' vs 'chocolate cake'); MiniLM ranks it #1.",
            "2. Vision Failure (`test/caesar_salad/130246.jpg`):",
            "   Quantized ResNet18 misclassifies a heavily topped Caesar",
            "   salad with tortilla strips as `tacos` (texture ambiguity).",
        ], bg="#FFFFFF", border="#111114", title_color="#111114")
        _card(ax, 0.7, 0.85, 11.93, 2.3, "Limitations & What We Would Do Next", [
            "• Current Limitations: (1) Closed 10-class classifier cannot recognize multi-dish platters or out-of-domain foods;",
            "  (2) Regex constraint parser only extracts predefined English patterns for time and 6 dietary tags.",
            "• Next Steps: (1) Expand to all 101 Food-101 classes with entropy thresholding for out-of-distribution rejection;",
            "  (2) Add cross-encoder re-ranking or vision-language embeddings (CLIP INT8) for open-vocabulary image retrieval.",
        ], bg="#F0EBFF", border="#4200FF")
        pdf.savefig(fig)
        plt.close(fig)

        # SLIDE 10: AI Usage Statement (R8)
        fig, ax = _new_slide("3. Conclusion · AI Usage Statement (R8)", "Transparency on AI Coding Assistant Usage", 10)
        _card(ax, 0.7, 1.0, 11.93, 4.9, "AI Tools Used & Scope of Assistance (Section 6 & Requirement R8)", [
            "• AI Assistant Used: Google Antigravity (AI pair-programming assistant).",
            "",
            "• Parts Assisted by AI:",
            "  1. Project Planning & Scaffolding: Structuring the FastAPI + React + PyTorch/ONNX repository layout.",
            "  2. Code Implementation & Debugging: Drafting PyTorch training loops (`train_cnn.py`, `neural_retriever.py`),",
            "     Grad-CAM hook utilities (`gradcam.py`), ONNX Runtime static/dynamic INT8 quantization scripts,",
            "     and the React + FastAPI web application (`backend/main.py`, `frontend/src/App.jsx`).",
            "  3. Slide & Diagram Formatting: Generating `docs/architecture.png`, charts, and this slide PDF.",
            "",
            "• Student Verification & Ownership:",
            "  - Every model (ScratchCNN, Fine-Tuned ResNet18, Quantized ResNet18, TF-IDF, EmbeddingBag+MLP, Fine-Tuned MiniLM,",
            "    and Quantized MiniLM) was genuinely trained, quantized, and benchmarked locally on my own hardware.",
            "  - Every metric in the Results Table comes directly from measured runs (`docs/benchmark_results.json`) over",
            "    the 2,500 held-out Food-101 test images and 46 held-out retrieval test queries.",
        ], bg="#FFFFFF", border="#4200FF", fontsize=9.8)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Generated 10-slide presentation PDF at {out_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="Ahmed")
    parser.add_argument("--student-id", default="[Your Student ID]")
    parser.add_argument("--repo-url", default="https://github.com/<your-username>/swe015-food-recipe-finder")
    parser.add_argument("--video-url", default="https://youtu.be/<your-2min-demo-link>")
    args = parser.parse_args()
    build_presentation_pdf(
        student_name=args.name,
        student_id=args.student_id,
        repo_url=args.repo_url,
        video_url=args.video_url,
    )
