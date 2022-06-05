import torch
import json
from PIL import Image


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_lst, resolution=512, transform=None):
        with open(data_lst) as json_file:
            data = json.load(json_file)
        
        self.data = data
        self.resolution = resolution
        self.transform = transform
    
    def get_item_by_name(self, image_dir):
        img = Image.open(image_dir)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        data = self.data[index]
        image_dir = data['patch_dir']
        label = data['label']

        img = Image.open(image_dir)

        if self.transform is not None:
            img = self.transform(img)
        return img, label, image_dir

    def __len__(self):
        return len(self.data)