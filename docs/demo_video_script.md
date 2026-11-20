# 2-Minute Narrated Demo Video Script (Deliverable 7.2)

> **Recording Rules (Gate 1 & Gate 2):**
> - Screen recording with **voice narration**, strictly **≤ 2:00** (only the first 2:00 are watched).
> - Start the server first: `.\.venv\Scripts\python.exe run.py` and open `http://127.0.0.1:8000` (wait for the loading screen to clear into the upload workspace).
> - Have one custom food photo ready on your Desktop (e.g., a pizza or sushi photo downloaded from Google Images) for the live upload demo, plus open `docs/SWE015_Presentation.pdf` in another browser tab on Slide 4 / Slide 7 / Slide 9 for the second half.
> - Upload to **YouTube (Unlisted)** or **Google Drive ("Anyone with the link")** and test in a private/incognito window.

---

## 0:00–0:15 — The Problem in One Sentence
- **On Screen:** The initial workspace (`http://127.0.0.1:8000`) showing the big **Upload a dish photo** box and sample pills.
- **Narration:**
  > "Hi, I'm [Your Name]. For Use Case 3, I built **Food Photo to Recipe Finder**, an application that lets users photograph a dish and type dietary or time constraints to retrieve matching recipes in milliseconds on a standard CPU using INT8 quantized ResNet18 and INT8 MiniLM models."

---

## 0:15–1:15 — Live Demo: At Least Two Different Inputs (Including One Difficult Case)

### Input 1: Custom Uploaded Photo + Time & Diet Constraints (0:15–0:45)
- **Action on Screen:**
  1. Click **Choose Photo** and upload your custom food photo (e.g., a pizza photo).
  2. In the text box, type: `vegetarian, under 30 minutes, gluten free` and click **Find Recipes** (or press `Enter`).
  3. Point to the right card: predicted dish (`Pizza` + confidence + INT8 ms), **Input RGB vs. Grad-CAM (`layer4[-1]`)** heatmap focusing on the food rather than the background, and the parsed constraint pills (`dish: pizza`, `max_time ≤ 30 min`, `vegetarian`, `gluten free`).
  4. Click the `#1` recipe row in **Top-5 Matching Recipes** to open the **modal** showing full ingredients and preparation steps, then press `Escape` / click **Close**.
- **Narration:**
  > "First, I'll upload an unseen photo of pizza and ask for *'vegetarian, under 30 minutes, gluten free'*. Our static-INT8 ONNX ResNet18 classifies it as Pizza, and the Grad-CAM heatmap shows the CNN focusing on the pizza crust and toppings. Classical regex parsing extracts the 30-minute time cap and dietary flags, hard-filters to pizza recipes, and our INT8 MiniLM encoder ranks the top-5 matches—clicking any recipe opens its full ingredients and steps."

### Input 2: Difficult Case — Misclassified / Ambiguous Photo or Synonym-Heavy Query (0:45–1:15)
- **Action on Screen:**
  1. Click **Ramen** (or upload an ambiguous/busy dish photo, e.g., `data/food101_10class/test/caesar_salad/130246.jpg` or a ramen photo with a synonym-heavy query like `fast plant based Japanese broth bowl in 15 min` — no words `"ramen"`, `"vegan"`, or `"noodles"`!).
  2. Click **Find Recipes** and scroll down to the **Side-by-Side Model Comparison** two-table grid.
- **Narration:**
  > "For a harder semantic case, I'll select Ramen and type *'fast plant based Japanese broth bowl in 15 min'*—notice it avoids the words *'ramen'* and *'vegan'*. Looking at the live Side-by-Side Model Comparison tables, we see all three vision models and all four retrieval models compared live: classical TF-IDF relies on exact stems, while the fine-tuned and INT8 quantized MiniLM models capture semantic synonyms like *'plant based'* and *'broth bowl'* at roughly 3× lower CPU latency than FP32 MiniLM."

---

## 1:15–1:45 — Architecture Diagram and Results Table
- **Action on Screen:** Switch to `docs/SWE015_Presentation.pdf` — show **Slide 4 (Architecture Diagram)** for ~10s, then **Slide 7 (Results Table)** for ~20s.
- **Narration:**
  > "Here is our end-to-end architecture matching the FastAPI and React code: uploaded images flow to static-INT8 ResNet18 and Grad-CAM, while text queries pass through regex constraint extraction and Porter stemming before INT8 MiniLM retrieval. On our held-out test sets of 2,500 Food-101 images and 46 queries over 213 recipes, transfer learning boosts Top-1 accuracy from 55.8% to 92.4%, and ONNX INT8 quantization cuts ResNet18 size by 4× to 10.76 MB and MiniLM size by 3.9× to 22.3 MB while cutting CPU latency down to a few milliseconds."

---

## 1:45–2:00 — Conclusion
- **Action on Screen:** Switch to **Slide 9 (Conclusion & Failure Examples)**.
- **Narration:**
  > "Overall, INT8 quantization preserves Transformer retrieval quality (0.9420 MRR) and 92.28% vision Top-1 accuracy with dramatic CPU speedups. Our main limitation is closed 10-class prediction on visually ambiguous mixed dishes like garnished Caesar salads, which we'd address next with open-vocabulary CLIP embeddings. Thanks for watching!"
