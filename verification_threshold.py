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


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

valid_dataset = datasets.ImageFolder(
    "Data/kaggle_saree/valid",
    transform=transform
)

test_dataset = datasets.ImageFolder(
    "Data/kaggle_saree/test",
    transform=transform
)


valid_loader = DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

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


# --------------------------------------------------
# Function to extract embeddings
# --------------------------------------------------

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

    return embeddings, labels


# --------------------------------------------------
# Create pair scores
# --------------------------------------------------

def create_pairs(embeddings, labels):

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

    return scores, targets


# --------------------------------------------------
# Validation: find best threshold
# --------------------------------------------------

print("\n================================")
print("VALIDATION THRESHOLD SELECTION")
print("================================")

valid_embeddings, valid_labels = get_embeddings(
    valid_loader
)

valid_scores, valid_targets = create_pairs(
    valid_embeddings,
    valid_labels
)


best_threshold = 0
best_accuracy = 0


for threshold in [
    x / 100 for x in range(-100, 101)
]:

    correct = 0

    for score, target in zip(
        valid_scores,
        valid_targets
    ):

        if score >= threshold:
            prediction = 1
        else:
            prediction = 0

        if prediction == target:
            correct += 1

    accuracy = correct / len(valid_scores)

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_threshold = threshold


print(
    "Best validation threshold:",
    best_threshold
)

print(
    "Validation accuracy:",
    f"{best_accuracy * 100:.2f}%"
)

print(
    "Validation pairs:",
    len(valid_scores)
)


# --------------------------------------------------
# Test: evaluate using validation threshold
# --------------------------------------------------

print("\n================================")
print("TEST VERIFICATION")
print("================================")

test_embeddings, test_labels = get_embeddings(
    test_loader
)

test_scores, test_targets = create_pairs(
    test_embeddings,
    test_labels
)


correct = 0


for score, target in zip(
    test_scores,
    test_targets
):

    if score >= best_threshold:
        prediction = 1
    else:
        prediction = 0

    if prediction == target:
        correct += 1


test_accuracy = correct / len(test_scores)


print(
    "Threshold used:",
    best_threshold
)

print(
    "Test verification accuracy:",
    f"{test_accuracy * 100:.2f}%"
)

print(
    "Test pairs:",
    len(test_scores)
)

print("================================")