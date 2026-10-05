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


# ------------------------------------------------
# Gallery transformation
# Keep gallery images in their original colors
# ------------------------------------------------

gallery_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------
# Query transformation
# Strongly change the colors
# ------------------------------------------------

color_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ColorJitter(
        brightness=0.8,
        contrast=0.8,
        saturation=1.0,
        hue=0.5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ------------------------------------------------
# Load gallery
# ------------------------------------------------

gallery_dataset = datasets.ImageFolder(
    "Data/kaggle_saree/train",
    transform=gallery_transform
)

gallery_loader = DataLoader(
    gallery_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------------
# Load test images with color transformation
# ------------------------------------------------

query_dataset = datasets.ImageFolder(
    "Data/kaggle_saree/test",
    transform=color_transform
)

query_loader = DataLoader(
    query_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------------
# Load trained model
# ------------------------------------------------

model = SareeEmbeddingModel(
    embedding_dim=256
)

model.load_state_dict(
    torch.load(
        "saree_contrastive_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ------------------------------------------------
# Extract embeddings
# ------------------------------------------------

def get_embeddings(loader):

    embeddings = []
    labels = []

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(device)

            batch_embeddings = model(images)

            batch_embeddings = F.normalize(
                batch_embeddings,
                p=2,
                dim=1
            )

            embeddings.append(
                batch_embeddings.cpu()
            )

            labels.append(batch_labels)

    embeddings = torch.cat(
        embeddings,
        dim=0
    )

    labels = torch.cat(
        labels,
        dim=0
    )

    return embeddings, labels


print("\nExtracting gallery embeddings...")

gallery_embeddings, gallery_labels = get_embeddings(
    gallery_loader
)

print(
    "Gallery embeddings:",
    gallery_embeddings.shape
)


print("\nExtracting color-changed query embeddings...")

query_embeddings, query_labels = get_embeddings(
    query_loader
)

print(
    "Query embeddings:",
    query_embeddings.shape
)


# ------------------------------------------------
# Calculate cosine similarity
# ------------------------------------------------

similarity_matrix = torch.mm(
    query_embeddings,
    gallery_embeddings.T
)


# ------------------------------------------------
# Top-1 identification
# ------------------------------------------------

correct = 0

for i in range(len(query_embeddings)):

    similarities = similarity_matrix[i]

    best_index = torch.argmax(
        similarities
    )

    predicted_label = gallery_labels[
        best_index
    ]

    actual_label = query_labels[i]

    if predicted_label == actual_label:
        correct += 1


accuracy = correct / len(query_embeddings)


print("\n================================")
print("COLOR-INVARIANCE RESULTS")
print("================================")

print(
    "Correct:",
    correct,
    "/",
    len(query_embeddings)
)

print(
    "Color-Invariant Top-1 Accuracy:",
    f"{accuracy * 100:.2f}%"
)