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


# Load test dataset
test_dataset = datasets.ImageFolder(
    "Data/kaggle_saree/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# Load trained model
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


# Extract embeddings
embeddings = []
labels = []


print("\nExtracting test embeddings...")


with torch.no_grad():

    for images, batch_labels in test_loader:

        images = images.to(device)

        output = model(images)

        output = F.normalize(
            output,
            p=2,
            dim=1
        )

        embeddings.append(
            output.cpu()
        )

        labels.append(
            batch_labels
        )


embeddings = torch.cat(
    embeddings,
    dim=0
)

labels = torch.cat(
    labels,
    dim=0
)


print(
    "Embeddings:",
    embeddings.shape
)


# ------------------------------------------------
# Create positive and negative pairs
# ------------------------------------------------

positive_scores = []
negative_scores = []


number_of_images = len(embeddings)


for i in range(number_of_images):

    for j in range(i + 1, number_of_images):

        similarity = torch.dot(
            embeddings[i],
            embeddings[j]
        ).item()

        if labels[i] == labels[j]:

            positive_scores.append(
                similarity
            )

        else:

            negative_scores.append(
                similarity
            )


print("\n==============================")
print("VERIFICATION RESULTS")
print("==============================")

print(
    "Positive pairs:",
    len(positive_scores)
)

print(
    "Negative pairs:",
    len(negative_scores)
)

print(
    "Average positive similarity:",
    sum(positive_scores) / len(positive_scores)
)

print(
    "Average negative similarity:",
    sum(negative_scores) / len(negative_scores)
)