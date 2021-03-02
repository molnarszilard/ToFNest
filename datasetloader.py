import torch.utils.data as data
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import Resize, Compose, ToTensor
import torch, time, os
import random
import cv2

class DatasetLoader(data.Dataset):
    
    def __init__(self, root='/home/user/dataset/', seed=None, train=True):
        np.random.seed(seed)
        self.root = Path(root)

        self.depth_input_paths = [root+'depth_input/'+d for d in os.listdir(root+'depth_input/')]
        self.depth_input_paths = np.random.choice(self.depth_input_paths, len(self.depth_input_paths), False)
        
        self.length = len(self.depth_input_paths)
            
    def __getitem__(self, index):
        path = self.depth_input_paths[index]
        depth_input = cv2.imread(path,cv2.IMREAD_UNCHANGED )
        if len(depth_input.shape) < 3:
            print("Got 1 channel depth images, creating 3 channel depth images")
            combine_depth = np.empty((depth_input.shape[0],depth_input.shape[1], 3))
            combine_depth[:,:,0] = depth_input
            combine_depth[:,:,1] = depth_input
            combine_depth[:,:,2] = depth_input
            depth_input = combine_depth
        normalgt = Image.open(path.replace('depth_input', 'normalimages'))
        depth_input_mod = np.moveaxis(cv2.resize(depth_input,(depth_input.shape[1],depth_input.shape[0])).astype(np.float32),-1,0)
        normalgt_mod = Compose([Resize((depth_input.shape[0],depth_input.shape[1])), ToTensor()])(normalgt)
        return depth_input_mod, normalgt_mod

    def __len__(self):
        return self.length

if __name__ == '__main__':
    # Testing
    dataset = DatasetLoader()
    print(len(dataset))
    for item in dataset[0]:
        print(item.size())
