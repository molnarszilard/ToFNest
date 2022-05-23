import torch.utils.data as data
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms import Resize, Compose, ToTensor
import torch, time, os
import random
import cv2

class DatasetLoader(data.Dataset):
    
    def __init__(self, root='/dataset/', seed=None, train=True, ddir='depth3'):
        np.random.seed(seed)
        self.root = Path(root)
        self.ddir = ddir

        if train:
            self.depth_input_paths = [root+ddir+'/train/'+d for d in os.listdir(root+ddir+'/train')]
            # Randomly choose 50k images without replacement
            # self.rgb_paths = np.random.choice(self.rgb_paths, 4000, False)
        else:
            self.depth_input_paths = [root+ddir+'/test/'+d for d in os.listdir(root+ddir+'/test/')]
            # self.rgb_paths = np.random.choice(self.rgb_paths, 1000, False)
        
        self.length = len(self.depth_input_paths)
            
    def __getitem__(self, index):
        path = self.depth_input_paths[index]
        depth_input = cv2.imread(path,cv2.IMREAD_UNCHANGED).astype(np.float32)
        if len(depth_input.shape) < 3:
            print("Got 1 channel depth images, creating 3 channel depth images")
            combine_depth = np.empty((depth_input.shape[0],depth_input.shape[1], 3))
            combine_depth[:,:,0] = depth_input
            combine_depth[:,:,1] = depth_input
            combine_depth[:,:,2] = depth_input
            depth_input = combine_depth
        normalgt = Image.open(path.replace(self.ddir, 'normalimages'))
        depth_input_mod = np.moveaxis(depth_input,-1,0)
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
