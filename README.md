# Color-Invariant Saree Design Recognition

## Overview

This project implements a deep learning based image retrieval and verification system for saree designs.

The objective is to identify visually similar saree designs while reducing the effect of color differences. The system converts each saree image into a compact 256-dimensional embedding. Similar images should have embeddings that are close together in the embedding space.

The system supports:

- Image embedding generation
- Similarity-based retrieval
- Top-K identification
- Verification using embedding similarity
- Synthetic color-perturbation evaluation
- Model efficiency measurement

---

## Approach

The pipeline consists of the following stages:

1. Input saree images are resized to 224 × 224 pixels.
2. Color augmentation is applied during training to improve robustness to color changes.
3. A pretrained ConvNeXt-Atto backbone extracts visual features.
4. A linear projection layer converts the features into a 256-dimensional embedding.
5. L2 normalization is applied to the embedding.
6. Contrastive learning is used to make different augmented views of the same image similar.
7. Cosine similarity is used for image retrieval and verification.

### Pipeline

```text
Saree Image
     ↓
Resize to 224 × 224
     ↓
Color Augmentation
     ↓
ConvNeXt-Atto
     ↓
256-D Embedding
     ↓
L2 Normalization
     ↓
Cosine Similarity
     ↓
Retrieval / Verification