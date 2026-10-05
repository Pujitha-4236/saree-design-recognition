import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Image size used by the model
IMAGE_SIZE = 224


# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Change brightness, contrast, saturation and hue
    # This helps the model focus more on design than color
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.7,
        hue=0.1
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation and test transformations
# No random color changes here
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Dataset locations
TRAIN_PATH = "Data/kaggle_saree/train"
VALID_PATH = "Data/kaggle_saree/valid"
TEST_PATH = "Data/kaggle_saree/test"


# Load datasets
train_dataset = datasets.ImageFolder(
    TRAIN_PATH,
    transform=train_transform
)

valid_dataset = datasets.ImageFolder(
    VALID_PATH,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_PATH,
    transform=eval_transform
)


# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
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


print("Dataset loaded successfully!")

print("Training images:", len(train_dataset))
print("Validation images:", len(valid_dataset))
print("Test images:", len(test_dataset))

print("Classes:", train_dataset.classes)

print("Number of training batches:", len(train_loader))