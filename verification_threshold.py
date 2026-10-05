import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import SareeEmbeddingModel


IMAGE_SIZE = 224

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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


# Load model
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


# Create pair scores
scores = []
targets = []


number_of_images = len(embeddings)


for i in range(number_of_images):

    for j in range(i + 1, number_of_images):

        similarity = torch.dot(
            embeddings[i],
            embeddings[j]
        ).item()

        scores.append(similarity)

        if labels[i] == labels[j]:
            targets.append(1)
        else:
            targets.append(0)


# Find best threshold
best_threshold = 0
best_accuracy = 0


for threshold in [
    x / 100 for x in range(-100, 101)
]:

    correct = 0

    for score, target in zip(scores, targets):

        if score >= threshold:
            prediction = 1
        else:
            prediction = 0

        if prediction == target:
            correct += 1

    accuracy = correct / len(scores)

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_threshold = threshold


print("\n================================")
print("VERIFICATION THRESHOLD")
print("================================")

print(
    "Best threshold:",
    best_threshold
)

print(
    "Verification accuracy:",
    f"{best_accuracy * 100:.2f}%"
)

print(
    "Total pairs:",
    len(scores)
)