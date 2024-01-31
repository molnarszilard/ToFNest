import os
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dir',dest='dir', default="",
                    help='the directory to the source files')
parser.add_argument('--extension',dest='ext', default=".png",
                    help='the extension that you want to find')
args = parser.parse_args()

directory=args.dir
dlist=os.listdir(directory)
dlist.sort()

f = open(directory+"filelist.txt", "w")
print(directory)
for filename in dlist:
    if filename.endswith(args.ext):
        f.write(filename[:-len(args.ext)]+"\n")
    else:
        continue
f.close()