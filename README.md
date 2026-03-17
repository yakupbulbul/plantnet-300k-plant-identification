# PlantNet-300K PlantID Pro — Model-Based Plant Identification (Capstone)

PlantID Pro is a capstone project focused on model-based plant identification from images. The core system extracts visual features from a large plant dataset, indexes them for similarity search, and returns the most likely species with a short description.

**At a Glance**
- Task: plant species identification from images.
- Dataset: PlantNet-300K with approximately 306,146 images and 1,081 species.
- Models: ResNeXt50_32x4d and ViT-B/32 for feature extraction.
- Retrieval: FAISS `IndexFlatL2` for nearest-neighbor search.
- Reported accuracy: improved to over 70% after switching to TensorFlow (per report).
- Code: feature extraction, indexing, and API in `model/`.
- This repo: model-focused, publishable snapshot; dataset images and mobile app excluded.

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
- This repo keeps lightweight metadata only.
- Larger metadata and index artifacts are referenced through the Drive folder in the external files section.
- Dataset reference (from original dataset README): Zenodo DOI
  ```text
  https://doi.org/10.5281/zenodo.4726653
  ```

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
- `model/` feature extraction, indexing, and API code.
- `plantnet_300K/` dataset structure and dataset README.
- `plantnet_300K/metaData/` includes `names.json` and `dataWithImages.csv`.
- `yakup_bulbul_final_report_public.pdf` publishable report (personal details removed).
- `.gitignore` configured to exclude dataset images and private files.

**Not Included**
- Mobile application code.
- Dataset images.
- Large metadata artifacts provided separately through Drive.
- FAISS index files (e.g., `indexedImagesFeaturesData.idx`) and large derived binaries.
- Presentation video and the private report.

**Model Code**
- `model/config.py` dataset paths and model settings (uses `PLANTNET_DATASET_DIR` if set).
- `model/utils.py` feature extraction, FAISS indexing, and helpers.
- `model/api.py` search functions (embedding + FAISS retrieval).
- `model/app.py` Flask API (`/predict` and `/images/<path>`).
- `model/analysis.py` local search test helper.
- `model/metrics.py` evaluation script.

**Quick Start (Local API)**
1. Set the dataset location with `PLANTNET_DATASET_DIR` (example: `export PLANTNET_DATASET_DIR=/path/to/plantnet_300K`).
2. Install dependencies (Python 3.10+ recommended): `pip install torch torchvision faiss-cpu pandas numpy flask pillow opencv-python scikit-learn tqdm statsmodels scipy`.
3. Run the API: `python3 model/app.py`.

**External Files (Optional Download)**
- `metadata.json` is not included in this repository to keep the GitHub version lightweight.
- `indexedImagesFeaturesData.idx` (precomputed FAISS index) is not included due to size.
- The original dataset README is in the same folder and points to the official dataset source.
- If you need the dataset, follow the instructions in that README.
- Access the folder here:
  ```text
  https://drive.google.com/drive/folders/1y1IaxsqYia8l5q87fd9z4dJW5PjRRNc9
  ```

**GitHub Upload (Main Branch Only)**
1. `git status` and confirm dataset images are not listed.
2. `git remote add origin <YOUR_GITHUB_REPO_URL>`
3. `git push -u origin main`

**How to Use This Repo**
1. Read `yakup_bulbul_final_report_public.pdf` for full methodology and references.
2. Place your dataset under `plantnet_300K/` (or set `PLANTNET_DATASET_DIR`).
3. If you publish dataset images, ensure you have redistribution rights and update `.gitignore` accordingly.

**Privacy and Publishing**
- Do not publish `yakup_bulbul_final_report (2).pdf` because it contains personal and institutional details.
- The presentation video is kept local and is not included in the repository.
- Verify dataset redistribution rights before publishing images.

**Contact**
If you want the mobile app code or have questions, email:
`yakupbulbul.vo@gmail.com`

**License**
Add a license file before publishing.
