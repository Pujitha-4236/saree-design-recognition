import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):

        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Number of images in the batch
        batch_size = z1.size(0)

        # Combine both views
        embeddings = torch.cat([z1, z2], dim=0)

        # Calculate similarity
        similarity = torch.matmul(
            embeddings,
            embeddings.T
        )

        # Scale similarity using temperature
        similarity = similarity / self.temperature

        # Remove self-similarity
        mask = torch.eye(
            2 * batch_size,
            device=similarity.device
        ).bool()

        similarity = similarity.masked_fill(mask, -float("inf"))

        # Create positive-pair targets
        targets = torch.arange(
            2 * batch_size,
            device=similarity.device
        )

        targets = (targets + batch_size) % (2 * batch_size)

        # Calculate loss
        loss = F.cross_entropy(
            similarity,
            targets
        )

        return loss


if __name__ == "__main__":

    # Test the loss function
    z1 = torch.randn(16, 256)
    z2 = torch.randn(16, 256)

    loss_function = ContrastiveLoss()

    loss = loss_function(z1, z2)

    print("Contrastive loss test successful!")
    print("Loss:", loss.item())