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


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Gallery
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


# Query
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


# Model
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


def get_embeddings(loader):

    embeddings = []
    labels = []

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(device)

            output = model(images)

            output = F.normalize(
                output,
                p=2,
                dim=1
            )

            embeddings.append(output.cpu())
            labels.append(batch_labels)

    return (
        torch.cat(embeddings),
        torch.cat(labels)
    )


print("\nExtracting gallery embeddings...")

gallery_embeddings, gallery_labels = get_embeddings(
    gallery_loader
)

print("Gallery:", gallery_embeddings.shape)


print("\nExtracting query embeddings...")

query_embeddings, query_labels = get_embeddings(
    query_loader
)

print("Queries:", query_embeddings.shape)


# Cosine similarity
similarity = torch.mm(
    query_embeddings,
    gallery_embeddings.T
)


top1 = 0
top5 = 0
top10 = 0


for i in range(len(query_embeddings)):

    scores = similarity[i]

    # Get highest similarity results
    top_indices = torch.argsort(
        scores,
        descending=True
    )

    actual_label = query_labels[i]

    # Top 1
    if gallery_labels[top_indices[0]] == actual_label:
        top1 += 1

    # Top 5
    top5_labels = gallery_labels[
        top_indices[:5]
    ]

    if actual_label in top5_labels:
        top5 += 1

    # Top 10
    top10_labels = gallery_labels[
        top_indices[:10]
    ]

    if actual_label in top10_labels:
        top10 += 1


total = len(query_embeddings)


print("\n==============================")
print("TOP-K RETRIEVAL RESULTS")
print("==============================")

print(
    "Top-1 Accuracy:",
    f"{top1 / total * 100:.2f}%"
)

print(
    "Top-5 Accuracy:",
    f"{top5 / total * 100:.2f}%"
)

print(
    "Top-10 Accuracy:",
    f"{top10 / total * 100:.2f}%"
)