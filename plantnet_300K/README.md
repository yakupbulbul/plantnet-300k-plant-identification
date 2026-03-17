# PlantNet-300K Dataset (Local Copy)

This folder contains a local copy of the PlantNet-300K image dataset used for the PlantID Pro capstone.

Structure
- `images_train/<class_id>/*.jpg`
- `images_val/<class_id>/*.jpg`
- `images_test/<class_id>/*.jpg`

Each class is a numeric folder ID. If you have a class-to-species mapping file, keep it alongside this README (for example `class_mapping.csv`).

Usage Notes
- This dataset is very large. If you plan to publish the repo on GitHub, consider Git LFS or external storage for the images.
- Make sure you have the right to redistribute the images before publishing.
- This repo keeps only the folder structure in git; images are intentionally excluded.

Suggested Additions
- A download or preparation script.
- A label mapping file.
- A short data card describing provenance, license, and intended use.
