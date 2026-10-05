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


# Image transformation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------------
# Load gallery
# -------------------------

gallery_dataset = datasets.ImageFolder(
    "Data/kaggle_saree/train",
    transform=transform
)

gallery_loader = DataLoader(
    gallery_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# -------------------------
# Load test/query images
# -------------------------

query_dataset = datasets.ImageFolder(
    "Data/kaggle_saree/test",
    transform=transform
)

query_loader = DataLoader(
    query_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# -------------------------
# Load trained model
# -------------------------

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


# -------------------------
# Function to get embeddings
# -------------------------

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


# Get gallery embeddings
print("\nExtracting gallery embeddings...")

gallery_embeddings, gallery_labels = get_embeddings(
    gallery_loader
)

print(
    "Gallery embeddings:",
    gallery_embeddings.shape
)


# Get query embeddings
print("\nExtracting query embeddings...")

query_embeddings, query_labels = get_embeddings(
    query_loader
)

print(
    "Query embeddings:",
    query_embeddings.shape
)


# -------------------------
# Calculate similarity
# -------------------------

similarity_matrix = torch.mm(
    query_embeddings,
    gallery_embeddings.T
)


# -------------------------
# Top-1 identification
# -------------------------

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


print("\n==============================")
print("IDENTIFICATION RESULTS")
print("==============================")

print(
    "Correct:",
    correct,
    "/",
    len(query_embeddings)
)

print(
    "Top-1 Accuracy:",
    f"{accuracy * 100:.2f}%"
)