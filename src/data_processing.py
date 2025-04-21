from torchvision import transforms
from torch.utils.data import Dataset
import torch
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import os
from datasets import load_dataset as load_dataset
import tqdm
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import copy

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



class CocoHumanCIELabDataset(Dataset):
    def __init__(self, image_paths, size=(256, 256), pts_file="pts_in_hull.npy", train=True):
        self.image_paths = image_paths
        self.size = size
        self.pts_file = pts_file

        # Input: grayscale
        self.input_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # Output: (1, H, W) in [0, 1]
        ])

        # Resize before Lab conversion
        self.rgb_transform = transforms.Compose([
            transforms.Resize(self.size)
        ])

        # Load or create pts_in_hull and nearest neighbor model
        self.get_pts_in_hull(self.pts_file)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Input: grayscale version
        input_image = self.input_transform(image)  # (1, H, W)

        # Convert resized RGB to Lab for ab target
        resized_image = self.rgb_transform(image)
        lab = rgb2lab(np.array(resized_image))
        ab = lab[:, :, 1:3]  # (H, W, 2)

        # Encode ab into bin indices (0–312)
        label_map = self.encode_ab(ab)  # (H, W)
        target_tensor = torch.from_numpy(label_map).long()  # required for CrossEntropyLoss

        return input_image, target_tensor

    def create_pts_in_hull(self, n_bins=313, name="pts_in_hull", subsample=None):
        if subsample is not None:
            image_paths = self.image_paths[:subsample]
        else:
            image_paths = self.image_paths
        
        sample_size=1000
        batch_size = 2048
        max_images = 10000
        kmeans = MiniBatchKMeans(n_clusters=n_bins, random_state=42, batch_size=batch_size, verbose=1)

        for i, image_path in enumerate(tqdm.tqdm(image_paths, desc="Fitting MiniBatchKMeans")):
            if max_images and i >= max_images:
                break

            image = Image.open(image_path).convert("RGB")
            image = self.rgb_transform(image)  # Make sure this resizes to 256x256 or lower
            lab = rgb2lab(np.array(image)).astype(np.float32)
            ab = lab[:, :, 1:3].reshape(-1, 2)

            if len(ab) < sample_size:
                continue  # avoid crashing on small/corrupt images
            indices = np.random.choice(len(ab), size=sample_size, replace=False)
            kmeans.partial_fit(ab[indices])

        self.pts_in_hull = kmeans.cluster_centers_
        np.save(name, self.pts_in_hull)
        return self.pts_in_hull

    def get_pts_in_hull(self, name="pts_in_hull"):
        if os.path.exists(name):
            self.pts_in_hull = np.load(name)
        elif os.path.exists(name + ".npy"):
            self.pts_in_hull = np.load(name + ".npy")
        else:
            self.pts_in_hull = self.create_pts_in_hull(name=name)

        self.neigh = NearestNeighbors(n_neighbors=1)
        self.neigh.fit(self.pts_in_hull)
        return self.pts_in_hull

    def encode_ab(self, ab):
        h, w, _ = ab.shape
        ab_flat = ab.reshape(-1, 2)
        _, indices = self.neigh.kneighbors(ab_flat)
        indices = indices.reshape(h, w)
        return indices


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
    
    
def get_dataset(path, image_size=(256, 256), val_size=0.2, type="RGB"):
    """
    Load the dataset from the specified path.
    """
    # If the data/train and data/val are not available, download the dataset
    if not os.path.exists(os.path.join(path, "train")) or not os.path.exists(os.path.join(path, "val")):    
        download_dataset(path, val_size=val_size)
    # Load the dataset
    if type == "CIELab":
        all_image_paths = [os.path.join(path, TRAIN_PATH, f) for f in os.listdir(os.path.join(path, TRAIN_PATH))]
        all_image_paths += [os.path.join(path, VAL_PATH, f) for f in os.listdir(os.path.join(path, VAL_PATH))]

        # Create the full dataset once (no train/test distinction here)
        train_dataset = CocoHumanCIELabDataset(
            image_paths=all_image_paths,
            size=image_size,
            train=False  # or `train=True`, doesn't matter now
        )
        val_dataset = copy.deepcopy(train_dataset)
        train_dataset.image_paths = [os.path.join(path, TRAIN_PATH, f) for f in os.listdir(os.path.join(path, TRAIN_PATH))]
        val_dataset.image_paths = [os.path.join(path, VAL_PATH, f) for f in os.listdir(os.path.join(path, VAL_PATH))]
    else:
        train_dataset = CocoHumanRGBDataset(
            image_paths=[os.path.join(path, TRAIN_PATH,f) for f in os.listdir(os.path.join(path, TRAIN_PATH))],
            size=image_size,
        )
        val_dataset = CocoHumanRGBDataset(
            image_paths=[os.path.join(path, VAL_PATH,f) for f in os.listdir(os.path.join(path, VAL_PATH))],
            size=image_size,
        )
    return train_dataset, val_dataset
