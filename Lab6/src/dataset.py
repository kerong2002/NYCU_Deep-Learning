import os, json
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
def transform_img(img):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(img)
class iclevrDataset(Dataset):
    def __init__(self, root=None, mode="train"):
        super().__init__()
        assert mode in ['train', 'test', 'new_test'], "mode should be either 'train', 'test', or 'new_test'"
        with open(f'{mode}.json', 'r') as json_file:
            self.json_data = json.load(json_file)
            if mode == 'train':
                self.img_paths, self.labels = list(self.json_data.keys()), list(self.json_data.values())
            elif mode in ['test', 'new_test']:
                self.labels = self.json_data

        with open('objects.json', 'r') as json_file:
            self.objects_dict = json.load(json_file)
        self.labels_one_hot = torch.zeros(len(self.labels), len(self.objects_dict))
        for i, label in enumerate(self.labels):
            self.labels_one_hot[i][[self.objects_dict[j] for j in label]] = 1
        # initialize others
        self.root = root   
        self.mode = mode
            
    def __len__(self):
        return len(self.labels)      
    
    def __getitem__(self, index):
        if self.mode == 'train':
            img_path = os.path.join(self.root, self.img_paths[index])
            img = Image.open(img_path).convert('RGB')
            img = transform_img(img)
            label_one_hot = self.labels_one_hot[index]
            return img, label_one_hot
        
        elif self.mode in ['test', 'new_test']:
            label_one_hot = self.labels_one_hot[index]
            return label_one_hot

if __name__ == '__main__':
    dataset = iclevrDataset(root='iclevr', mode='train')
    print(len(dataset))
    x, y = dataset[0]
    print(x.shape, y.shape)
    dataset = iclevrDataset(root='iclevr', mode='test')
    print(len(dataset))
    y = dataset[0]
    print(y.shape)
