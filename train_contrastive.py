import torch
import torch.optim as optim

from model import SareeEmbeddingModel
from contrastive_dataset import train_loader
from contrastive_loss import ContrastiveLoss


# Select device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Create model
model = SareeEmbeddingModel(
    embedding_dim=256
)

model = model.to(device)


# Create contrastive loss
criterion = ContrastiveLoss(
    temperature=0.07
)


# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# Number of training epochs
EPOCHS = 5


print("Starting training...")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_number, (view1, view2, labels) in enumerate(train_loader):

        # Move images to device
        view1 = view1.to(device)
        view2 = view2.to(device)

        # Get embeddings
        z1 = model(view1)
        z2 = model(view2)

        # Calculate contrastive loss
        loss = criterion(z1, z2)

        # Clear previous gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        total_loss += loss.item()

        if (batch_number + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] "
                f"Batch [{batch_number + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Average Loss: {average_loss:.4f}"
    )


# Save trained model
torch.save(
    model.state_dict(),
    "saree_contrastive_model.pth"
)

print("\nTraining completed!")
print("Model saved as: saree_contrastive_model.pth")