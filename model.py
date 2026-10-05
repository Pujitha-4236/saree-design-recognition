import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SareeEmbeddingModel(nn.Module):

    def __init__(self, embedding_dim=256):

        super().__init__()

        # Pretrained CNN backbone
        self.backbone = timm.create_model(
            "convnext_atto.d2_in1k",
            pretrained=True,
            num_classes=0
        )

        # Number of features produced by the backbone
        feature_dim = self.backbone.num_features

        # Convert features to 256-dimensional embedding
        self.embedding = nn.Linear(
            feature_dim,
            embedding_dim
        )

    def forward(self, x):

        # Extract visual features
        features = self.backbone(x)

        # Convert features to embedding
        embeddings = self.embedding(features)

        # Normalize embeddings
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        return embeddings


if __name__ == "__main__":

    model = SareeEmbeddingModel()

    print("Model created successfully!")
    print("Embedding size:", 256)

    # Test with one batch of fake images
    test_images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(test_images)

    print("Input shape:", test_images.shape)
    print("Output shape:", output.shape)