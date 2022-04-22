from PIL import Image
from pathlib import Path
import numpy as np
import seaborn as sns
from sklearn.datasets import make_circles
import albumentations

class ImageDataset:
    
    def __init__(self, path: Path, transforms):
        self.img_pathes = list(path.rglob("*.jpg"))
        self.transforms = transforms
        
    def __len__(self):
        return len(self.img_pathes)
    
    def __getitem__(self, index):
        img_path = self.img_pathes[index]
        img = Image.open(img_path)
        img_array = np.array(img)
        img = Image.fromarray(self.transforms(image=img_array)["image"])
        return {
            "image": img,
            "label": img_path.stem
        }