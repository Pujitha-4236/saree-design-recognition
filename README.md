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

## Approach Note

I use a pretrained ConvNeXt-Atto backbone with a 256-dimensional normalized embedding to capture saree surface patterns while improving robustness to color changes. Images are resized to 224×224 and training uses color augmentation with two-view contrastive learning and a temperature-scaled contrastive loss. Embeddings are L2-normalized and compared using cosine similarity for retrieval and verification.

### Pipeline

```text
                    Saree Image
                         ↓
                Resize to 224 × 224
                         ↓
                Training: Color Augmentation
                         ↓
                  ConvNeXt-Atto
                         ↓
                  256-D Embedding
                         ↓
                  L2 Normalization
                         ↓
                 Cosine Similarity
                    ↙          ↘
             Retrieval       Verification
                 ↓                 ↓
             Top-K Ranking    Similar / Different
```

### Retrieval

```text
Query Image
     ↓
Generate Embedding
     ↓
Compare with Gallery Embeddings
     ↓
Cosine Similarity
     ↓
Rank Similar Images
     ↓
Top-K Results
```

### Verification


Verification determines whether two saree images belong to the same category based on their embedding similarity.

The validation set was used to select the similarity threshold, and the selected threshold was then evaluated on the unseen test set.

### Validation

```text
Best validation threshold: 0.36
Validation accuracy: 74.40%
Validation pairs: 6555

### Test

```text
Threshold used: 0.36
Test verification accuracy: 77.18%
Test pairs: 1770