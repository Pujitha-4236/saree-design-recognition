import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


IMAGE_SIZE = 224


# First view of the image
transform1 = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

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


# Second view of the SAME image
transform2 = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

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


class ContrastiveDataset(Dataset):

    def __init__(self, root):
        self.dataset = datasets.ImageFolder(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        image, label = self.dataset[index]

        # Create two different versions of the SAME image
        view1 = transform1(image)
        view2 = transform2(image)

        return view1, view2, label


# Training dataset
train_dataset = ContrastiveDataset(
    "Data/kaggle_saree/train"
)


# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)


if __name__ == "__main__":

    print("Contrastive dataset loaded successfully!")

    print("Number of images:", len(train_dataset))

    print("Number of batches:", len(train_loader))

    # Get one batch
    view1, view2, labels = next(iter(train_loader))

    print("View 1 shape:", view1.shape)
    print("View 2 shape:", view2.shape)
    print("Labels shape:", labels.shape)

    print("\nEach image now has two different views.")
    print("The two views belong to the same original image.")