import kagglehub
from kagglehub import KaggleDatasetAdapter
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os

TEST = True
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
    

def download_dataset(dataset_id, path):
    """
    Download the dataset from Hugging Face Hub.
    """
    # TODO: Check how to use Kaggle
    dataset = kagglehub.load_dataset(
        KaggleDatasetAdapter.HUGGING_FACE,
        "yanplayz08/coco-subset-for-pose-estimation",
        path,
        )
    
    return dataset
    
    
def load_dataset(path, size=(256, 256)):
    """
    Load the dataset from the specified path.
    """
    # If the data/train and data/val are not available, download the dataset
    if not os.path.exists(os.path.join(path, "train")) or not os.path.exists(os.path.join(path, "val")):    
        download_dataset("yanplayz08/coco-subset-for-pose-estimation", path)
    # Load the dataset
    train_dataset = CocoHumanRGBDataset(
        image_paths=[os.path.join(path, TRAIN_PATH,f) for f in os.listdir(os.path.join(path, TRAIN_PATH))],
        size=size,
    )
    val_dataset = CocoHumanRGBDataset(
        image_paths=[os.path.join(path, VAL_PATH,f) for f in os.listdir(os.path.join(path, VAL_PATH))],
        size=size,
    )
    return train_dataset, val_dataset
