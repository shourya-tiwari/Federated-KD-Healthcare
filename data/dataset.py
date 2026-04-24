import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class HospitalDataset(Dataset):
    """
    PyTorch Dataset for a single hospital's data.
    Takes a list of image paths and labels and returns
    transformed tensors ready for model training.
    """

    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths : list of full file paths to images
            labels      : list of ints (0 = Normal, 1 = Pneumonia)
            transform   : torchvision transforms to apply
        """
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform if transform else get_default_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image    = Image.open(img_path).convert('RGB')  # ensure 3 channels

        # Apply transforms
        image = self.transform(image)

        label = self.labels[idx]
        return image, label


def get_default_transform(image_size=64):
    """
    Standard transform pipeline for chest X-ray images.
    - Resize to 224x224 (standard for ResNet/EfficientNet)
    - Convert to tensor
    - Normalize with ImageNet mean/std (works well for medical images too)
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std =[0.229, 0.224, 0.225]   # ImageNet std
        )
    ])


def get_augmented_transform(image_size=64):
    """
    Augmented transform for training — adds random flips and slight rotation.
    Helps smaller hospitals generalize better with limited data.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


def get_dataloader(hospital_dict, batch_size=32, augment=False, shuffle=True):
    """
    Creates a DataLoader for a hospital dict returned by partition_into_hospitals().

    Args:
        hospital_dict : single hospital dict with 'images' and 'labels' keys
        batch_size    : number of images per batch
        augment       : use augmented transforms if True
        shuffle       : shuffle data each epoch

    Returns:
        DataLoader object
    """
    transform = get_augmented_transform() if augment else get_default_transform()

    dataset = HospitalDataset(
        image_paths=hospital_dict['images'],
        labels     =hospital_dict['labels'],
        transform  =transform
    )

    return DataLoader(
        dataset,
        batch_size =batch_size,
        shuffle    =shuffle,
        num_workers=0        # keep 0 on Windows to avoid multiprocessing issues
    )


if __name__ == "__main__":
    from data.partition import partition_into_hospitals

    DATA_DIR = r"G:\PROJECTS\FL with KD\Kaggle (Pneumonia) Chest X-ray Dataset\chest_xray\train"

    print("Partitioning data into hospitals...")
    hospitals = partition_into_hospitals(DATA_DIR)

    print("\nTesting DataLoader for each hospital:")
    print("=" * 55)

    for h in hospitals:
        loader = get_dataloader(h, batch_size=32)

        # Grab one batch and verify shape
        images, labels = next(iter(loader))

        expert_tag = "EXPERT    " if h['is_expert'] else "NON-EXPERT"
        print(f"Hospital {h['hospital_id']} [{expert_tag}] | "
              f"Batches: {len(loader):3d} | "
              f"Image shape: {tuple(images.shape)} | "
              f"Labels shape: {tuple(labels.shape)}")

    print("=" * 55)
    print("All DataLoaders working correctly.")