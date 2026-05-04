import os
from PIL import Image
from torch.utils.data import Dataset

class braindataset(Dataset):
    def __init__(self,image_dir,mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = os.path.join(self.image_dir,self.images[idx])
        mask = os.path.join(self.mask_dir,self.images[idx])

        img = Image.open(image).convert('RGB')
        mask = Image.open(mask).convert('L')
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img,mask


