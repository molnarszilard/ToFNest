import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
import shutil

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dir', default="",
                    help='the directory to the source files')
parser.add_argument('--extension',dest='ext', default=".png",
                    help='the extension that you want to find')
args = parser.parse_args()

directory=args.dir

dlist=os.listdir(directory)
dlist.sort()

n=0
for filename in dlist:
    if filename.endswith(args.ext):
        number=f'{n:05d}'
        os.rename(directory+filename,directory+number+args.ext)
        n=n+1
    else:
        continue
