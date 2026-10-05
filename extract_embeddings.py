import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SareeEmbeddingModel


IMAGE_SIZE = 224

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# Test image transformation
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load test dataset
test_dataset = datasets.ImageFolder(
    "Data/kaggle_saree/test",
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# Create model
model = SareeEmbeddingModel(
    embedding_dim=256
)

# Load trained weights
model.load_state_dict(
    torch.load(
        "saree_contrastive_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


all_embeddings = []
all_labels = []


print("Extracting embeddings...")


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        embeddings = model(images)

        # Normalize embeddings
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

        all_embeddings.append(
            embeddings.cpu()
        )

        all_labels.append(labels)


# Combine all batches
all_embeddings = torch.cat(
    all_embeddings,
    dim=0
)

all_labels = torch.cat(
    all_labels,
    dim=0
)


# Save embeddings
torch.save(
    {
        "embeddings": all_embeddings,
        "labels": all_labels,
        "classes": test_dataset.classes
    },
    "test_embeddings.pth"
)


print("\nEmbedding extraction completed!")

print("Number of images:", len(all_embeddings))
print("Embedding shape:", all_embeddings.shape)
print("Classes:", test_dataset.classes)

print("\nSaved as: test_embeddings.pth")