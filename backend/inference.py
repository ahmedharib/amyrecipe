"""Unified inference service: loads INT8 ONNX models eagerly and comparison baselines lazily."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from src.retrieval.quantize_bert import QuantizedBertRetriever
from src.vision.dataset import get_transforms
from src.vision.gradcam import GradCAM, overlay_heatmap_on_image, pil_to_base64_png
from src.vision.models import build_transfer_resnet18


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)


class AppInferenceService:
    """Runs INT8 ONNX models by default; loads FP32/classical comparison models on demand."""

    def __init__(
        self,
        vision_dir: Path = Path("weights/vision"),
        bert_dir: Path = Path("weights/bert_retriever"),
        neural_dir: Path = Path("weights/neural_retriever"),
        recipes_path: Path = Path("data/recipes.json"),
    ) -> None:
        self.vision_dir = vision_dir
        self.bert_dir = bert_dir
        self.neural_dir = neural_dir
        self.classes: List[str] = json.loads((vision_dir / "classes.json").read_text(encoding="utf-8"))
        self.recipes: List[Dict[str, Any]] = json.loads(recipes_path.read_text(encoding="utf-8"))
        _, self.eval_tf = get_transforms(224)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_vision = ort.InferenceSession(
            str(vision_dir / "resnet18_int8.onnx"),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )

        # Fine-tuned ResNet18 kept on CPU for Grad-CAM saliency overlays
        self.resnet_fp32 = build_transfer_resnet18(num_classes=len(self.classes), pretrained=False)
        self.resnet_fp32.load_state_dict(
            torch.load(vision_dir / "resnet18_finetuned.pt", map_location="cpu", weights_only=True)
        )
        self.resnet_fp32.eval()
        self.gradcam = GradCAM(self.resnet_fp32, self.resnet_fp32.layer4[-1])

        self.qbert_retriever = QuantizedBertRetriever(self.recipes, bert_dir, "bert_int8.onnx")

        # Lazy-loaded only when compare=True
        self._baselines_loaded = False
        self.scratch_cnn = None
        self.bert_retriever = None
        self.neural_retriever = None
        self.tfidf_retriever = None

    def _ensure_baselines(self) -> None:
        if self._baselines_loaded:
            return
        from src.retrieval.bert_retriever import BertRetriever
        from src.retrieval.neural_retriever import NeuralRetriever
        from src.retrieval.tfidf_retriever import TfidfRetriever
        from src.vision.models import ScratchCNN

        self.scratch_cnn = ScratchCNN(num_classes=len(self.classes))
        self.scratch_cnn.load_state_dict(
            torch.load(self.vision_dir / "scratch_cnn.pt", map_location="cpu", weights_only=True)
        )
        self.scratch_cnn.eval()

        self.bert_retriever = BertRetriever(self.recipes, self.bert_dir)
        self.neural_retriever = NeuralRetriever(self.recipes, self.neural_dir)
        self.tfidf_retriever = TfidfRetriever(self.recipes)
        self._baselines_loaded = True

    def _preprocess_image(self, image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
        tensor = self.eval_tf(image.convert("RGB")).unsqueeze(0)
        return tensor, tensor.numpy().astype(np.float32)

    def _format_top3(self, probs: np.ndarray) -> List[Dict[str, Any]]:
        return [
            {
                "dish": self.classes[int(i)],
                "display_name": self.classes[int(i)].replace("_", " ").title(),
                "confidence": round(float(probs[int(i)]), 4),
            }
            for i in np.argsort(-probs)[:3]
        ]

    def predict(
        self,
        image: Image.Image,
        query: str,
        compare: bool = False,
        use_dish_hint: bool = True,
    ) -> Dict[str, Any]:
        img_tensor, img_np = self._preprocess_image(image)

        t0 = time.perf_counter()
        int8_logits = self.ort_vision.run(["logits"], {"input": img_np})[0][0]
        int8_lat = (time.perf_counter() - t0) * 1000.0
        int8_probs = _softmax(int8_logits)
        pred_idx = int(np.argmax(int8_probs))
        pred_dish = self.classes[pred_idx]
        pred_conf = float(int8_probs[pred_idx])

        heatmap, _, _ = self.gradcam.generate(img_tensor, class_idx=pred_idx)
        resized_orig = image.convert("RGB").resize((384, 384))
        overlay_img = overlay_heatmap_on_image(resized_orig, heatmap, alpha=0.45)

        dish_hint = pred_dish if use_dish_hint else None
        q_str = query or pred_dish.replace("_", " ")

        t1 = time.perf_counter()
        ret_out = self.qbert_retriever.search(query=q_str, dish_hint=dish_hint, top_k=5)
        ret_lat = (time.perf_counter() - t1) * 1000.0

        response: Dict[str, Any] = {
            "vision": {
                "model_name": "ResNet18 Quantized (ONNX)",
                "precision": "INT8",
                "predicted_dish": pred_dish,
                "display_name": pred_dish.replace("_", " ").title(),
                "confidence": round(pred_conf, 4),
                "top3": self._format_top3(int8_probs),
                "latency_ms": round(int8_lat, 2),
                "original_image": pil_to_base64_png(resized_orig),
                "gradcam_image": pil_to_base64_png(overlay_img),
            },
            "retrieval": {
                "model_name": "Quantized MiniLM (ONNX)",
                "precision": "INT8",
                "parsed_query": ret_out["parsed_query"],
                "results": ret_out["results"],
                "latency_ms": round(ret_lat, 2),
            },
        }

        if compare:
            self._ensure_baselines()

            t_s = time.perf_counter()
            with torch.no_grad():
                s_logits = self.scratch_cnn(img_tensor).numpy()[0]
            s_lat = (time.perf_counter() - t_s) * 1000.0
            s_probs = _softmax(s_logits)
            s_idx = int(np.argmax(s_probs))

            t_r = time.perf_counter()
            with torch.no_grad():
                r_logits = self.resnet_fp32(img_tensor).numpy()[0]
            r_lat = (time.perf_counter() - t_r) * 1000.0
            r_probs = _softmax(r_logits)
            r_idx = int(np.argmax(r_probs))

            comp_ret = []
            for label, prec, engine in [
                ("Regex + TF-IDF Cosine", "N/A", self.tfidf_retriever),
                ("EmbeddingBag + MLP (Trained)", "FP32", self.neural_retriever),
                ("Fine-Tuned MiniLM (BERT)", "FP32", self.bert_retriever),
                ("Quantized MiniLM (ONNX)", "INT8", self.qbert_retriever),
            ]:
                t_q = time.perf_counter()
                out = engine.search(q_str, dish_hint=dish_hint, top_k=5)
                q_lat = (time.perf_counter() - t_q) * 1000.0
                comp_ret.append({
                    "model_name": label,
                    "precision": prec,
                    "latency_ms": round(q_lat, 2),
                    "top_recipes": [
                        {
                            "id": r["id"],
                            "title": r["title"],
                            "score": r["score"],
                            "prep_time_min": r["prep_time_min"],
                        }
                        for r in out["results"]
                    ],
                })

            response["comparison"] = {
                "vision": [
                    {
                        "model_name": "ScratchCNN (4-block)",
                        "precision": "FP32",
                        "predicted_dish": self.classes[s_idx],
                        "display_name": self.classes[s_idx].replace("_", " ").title(),
                        "confidence": round(float(s_probs[s_idx]), 4),
                        "latency_ms": round(s_lat, 2),
                    },
                    {
                        "model_name": "ResNet18 Fine-Tuned",
                        "precision": "FP32",
                        "predicted_dish": self.classes[r_idx],
                        "display_name": self.classes[r_idx].replace("_", " ").title(),
                        "confidence": round(float(r_probs[r_idx]), 4),
                        "latency_ms": round(r_lat, 2),
                    },
                    {
                        "model_name": "ResNet18 Quantized (ONNX)",
                        "precision": "INT8",
                        "predicted_dish": pred_dish,
                        "display_name": pred_dish.replace("_", " ").title(),
                        "confidence": round(pred_conf, 4),
                        "latency_ms": round(int8_lat, 2),
                    },
                ],
                "retrieval": comp_ret,
            }

        return response
