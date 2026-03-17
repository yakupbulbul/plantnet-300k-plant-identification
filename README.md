# PlantID Pro — AI Plant Identification (Capstone)

PlantID Pro is a capstone project focused on AI-based plant identification from images. The core system extracts visual features from a large plant dataset, indexes them for similarity search, and returns the most likely species with a short description.

**What This Repo Is**
- A publishable AI-focused snapshot of the project.
- Dataset structure only (images excluded from git).
- Mobile client code lives on a separate branch: `mobile-app`.

**Project Goal**
Identify a plant from a user-submitted image and return the biological species name with a short informational summary.

**Pipeline (From Report)**
1. Load images with `cv2` and PIL, resize to 224x224.
2. Apply `torchvision.transforms` for model compatibility.
3. Extract features using `ResNeXt50_32x4d` and `ViT-B/32`.
4. Fuse and normalize features into robust embeddings.
5. Index embeddings with FAISS `IndexFlatL2` for nearest-neighbor search.
6. Retrieve neighbors to infer species and return results via an API.

**Dataset**
- PlantNet-300K with approximately 306,146 images and 1,081 species.
- Long-tailed distribution and label uncertainty (per report).
- Local split layout: `plantnet_300K/images_train`, `plantnet_300K/images_val`, `plantnet_300K/images_test`.
- This branch keeps only the folder structure; images are excluded.

**Results (From Report)**
- Accuracy improved to over 70% after switching to TensorFlow.

**Repository Contents**
- `plantnet_300K/` dataset structure and dataset README.
- `yakup_bulbul_final_report_public.pdf` publishable report (personal details removed).
 

**Branches**
- `main` AI-focused, publishable assets and documentation.
- `mobile-app` Flutter client (kept separate to avoid mixing app code and AI assets).

**Code Status**
Training and inference code is not included in this branch yet. The full methodology is documented in the report. If you want code added, place the scripts here and I will update the README with exact run instructions.

**Privacy Notes**
- Do not publish `yakup_bulbul_final_report (2).pdf` because it contains personal and institutional details.
- The presentation video is kept local and is not included in the repository.
- Verify dataset redistribution rights before publishing images.

**License**
Add a license file before publishing.
