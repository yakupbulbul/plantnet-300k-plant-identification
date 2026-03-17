# PlantID Pro — AI Plant Identification (Capstone)

PlantID Pro is a capstone project focused on AI-based plant identification from images. The core system extracts visual features from a large plant dataset, indexes them for similarity search, and returns the most likely species with a short description.

**At a Glance**
- Task: plant species identification from images.
- Dataset: PlantNet-300K with approximately 306,146 images and 1,081 species.
- Models: ResNeXt50_32x4d and ViT-B/32 for feature extraction.
- Retrieval: FAISS `IndexFlatL2` for nearest-neighbor search.
- Reported accuracy: improved to over 70% after switching to TensorFlow (per report).
- This repo: AI-focused, publishable snapshot; dataset images and mobile app excluded.

**Project Goal**
Identify a plant from a user-submitted image and return the biological species name with a short informational summary.

**System Overview (From Report)**
The project processes plant images into feature embeddings, fuses representations from two pretrained models, normalizes the fused features, and indexes them for fast similarity search. An API layer is used to return predictions rather than direct database access.

**Pipeline (From Report)**
1. Load images with `cv2` and PIL to preserve color channels and formats.
2. Resize images to 224x224 using bilinear interpolation.
3. Apply `torchvision.transforms` for model compatibility.
4. Extract features using `ResNeXt50_32x4d` and `ViT-B/32`.
5. Fuse features and normalize the resulting embeddings.
6. Index embeddings using FAISS `IndexFlatL2` (L2 distance).
7. Retrieve nearest neighbors and return results via an API.

**Dataset**
- PlantNet-300K with approximately 306,146 images and 1,081 species.
- Long-tailed distribution and label uncertainty (per report).
- Local split layout: `plantnet_300K/images_train`, `plantnet_300K/images_val`, `plantnet_300K/images_test`.
- This branch keeps only the folder structure; images are excluded.

**Models and Tools (From Report)**
- Feature extraction: ResNeXt50_32x4d and ViT-B/32.
- Baseline reference: AlexNet (discussed in report).
- Training framework: TensorFlow (used to improve accuracy).
- Similarity search: FAISS.
- Database referenced: Hive.

**Results (From Report)**
- Accuracy improved to over 70% after switching to TensorFlow.
- Main challenge was the scale of the dataset and visually similar species.

**Repository Contents**
- `plantnet_300K/` dataset structure and dataset README.
- `yakup_bulbul_final_report_public.pdf` publishable report (personal details removed).
- `.gitignore` configured to exclude dataset images and private files.

**Not Included**
- Training and inference scripts.
- Mobile application code.
- Dataset images.
- Presentation video and the private report.

**How to Use This Repo**
1. Read `yakup_bulbul_final_report_public.pdf` for full methodology and references.
2. If you add training or inference code, place it in this repo and update this README with run commands.
3. If you publish dataset images, ensure you have redistribution rights and update `.gitignore` accordingly.

**Privacy and Publishing**
- Do not publish `yakup_bulbul_final_report (2).pdf` because it contains personal and institutional details.
- The presentation video is kept local and is not included in the repository.
- Verify dataset redistribution rights before publishing images.

**License**
Add a license file before publishing.
