import glob
import json
import os
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


class CarDamageDataset(Dataset):
    def __init__(self, json_dir, img_dir, transform=None):
        self.json_files = glob.glob(os.path.join(json_dir, "*.json"))
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, 'r') as j:
            annotation = json.load(j)

        img_file_name = os.path.splitext(os.path.basename(json_path))[0] + '.jpg' # GT와 같은 파일명 찾기
        img_path = os.path.join(self.img_dir, img_file_name)

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"WUT??? Image file not found : {img_path}")
            raise

        mask = Image.new("L", (image.width, image.height), 0)
        draw = ImageDraw.Draw(mask)
        for seg in annotation['annotations'][0]['segmentation']:
            # GT 구조 = [[[x1, y1], [x2, y2], ..., [xn, yn]]]
            polygon = [(point[0], point[1]) for point in seg[0]]
            draw.polygon(polygon, outline=255, fill=255)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask>0).float()

        return image, mask