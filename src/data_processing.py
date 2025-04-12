from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
from datasets import load_dataset as load_dataset
import tqdm

TEST = False
TRAIN_PATH = "train" if not TEST else "train_test"
VAL_PATH = "val" if not TEST else "val_test"

class CocoHumanRGBDataset(Dataset):
    def __init__(self, image_paths, size=(256, 256)):
        self.image_paths = image_paths
        self.size = size
        
        self.input_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

        self.target_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        input_image = self.input_transform(image)
        target_image = self.target_transform(image)
        
        return input_image, target_image
    

def download_dataset(data_path: str, val_size: float = 0.2):
    """
    Download the dataset from Hugging Face Hub.
    """
    # If the dataset is not split into train and val, split it

    os.makedirs(os.path.join(data_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "val"), exist_ok=True)

    # Load dataset
    ds = load_dataset("UCSC-VLAA/Recap-COCO-30K")
    dataset = ds["train"].train_test_split(test_size=val_size)

    def download_images(split_name, split_dataset):
        for item in tqdm.tqdm(split_dataset, desc=f"Downloading {split_name} images"):
            image_id = item['image_id']
            image = item['image']

            image_path = os.path.join(data_path, split_name, f"{image_id}.jpg")

            image.save(image_path, format='JPEG')

    download_images("train", dataset["train"])
    download_images("val", dataset["test"])
    
    return dataset
    
    
def get_dataset(path, image_size=(256, 256), val_size=0.2):
    """
    Load the dataset from the specified path.
    """
    # If the data/train and data/val are not available, download the dataset
    if not os.path.exists(os.path.join(path, "train")) or not os.path.exists(os.path.join(path, "val")):    
        download_dataset(path, val_size=val_size)
    # Load the dataset
    train_dataset = CocoHumanRGBDataset(
        image_paths=[os.path.join(path, TRAIN_PATH,f) for f in os.listdir(os.path.join(path, TRAIN_PATH))],
        size=image_size,
    )
    val_dataset = CocoHumanRGBDataset(
        image_paths=[os.path.join(path, VAL_PATH,f) for f in os.listdir(os.path.join(path, VAL_PATH))],
        size=image_size,
    )
    return train_dataset, val_dataset
