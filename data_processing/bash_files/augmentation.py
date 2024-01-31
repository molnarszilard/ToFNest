import imageio
imageio.plugins.freeimage.download()
import imgaug as ia
import os
from imgaug import augmenters as iaa
import numpy as np
import argparse
import cv2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dir', default="",
                    help='the directory to the source files')
args = parser.parse_args()

directory=args.dir
dlist=os.listdir(directory)
dlist.sort()

def rotaterand(img, imgname):
    rotate = iaa.Affine(rotate=(-180, 180))
    image_rot = rotate(image=img)
    imageio.imwrite(directory+imgname+"_rotrand.png",image_rot, format='PNG-FI')

def translate(img, imgname):
    imgname=imgname[:-4]
    translate_right = iaa.Affine(translate_px={"x": (130, 150), "y": (0)})
    image_tr_r = translate_right(image=img)
    width_cutoff = 320
    im1 = image_tr_r[:, width_cutoff:]
    im_h = cv2.hconcat([im1, im1])
    imageio.imwrite(directory+imgname+"_duplicated.png",im_h, format='PNG-FI')

def rotatecw(img, imgname):
    rotate = iaa.Affine(rotate=(85,95))
    image_rot = rotate(image=img)    
    imageio.imwrite(directory+imgname+"_rotcw.png",image_rot, format='PNG-FI')
    translate(image_rot,imgname+"_rotcw.png")

def rotateccw(img, imgname):
    rotate = iaa.Affine(rotate=(-95,-85))
    image_rot = rotate(image=img)
    imageio.imwrite(directory+imgname+"_rotccw.png",image_rot, format='PNG-FI')
    translate(image_rot,imgname+"_rotccw.png")

def scale(img, imgname):
    scale = iaa.Affine(scale={"x": (0.6, 1.4), "y": (0.6, 1.4)})
    image_scale = scale(image=img)
    imageio.imwrite(directory+imgname+"_scale.png",image_scale, format='PNG-FI')

def fliplr(img, imgname):
    image_fliplr = iaa.flip.fliplr(img)
    imageio.imwrite(directory+imgname+"_fliplr.png",image_fliplr, format='PNG-FI')

def flipud(img, imgname):
    image_flipud = iaa.flip.flipud(img)
    imageio.imwrite(directory+imgname+"_flipud.png",image_flipud, format='PNG-FI')

for filename in dlist:
    if filename.endswith(".png"):
        #print(os.path.join(directory, filename))
        image_name=filename
        print("Image:"+image_name)
        img=imageio.imread(directory+image_name, format='PNG-FI')
        image_name=image_name[:-4]
        rotaterand(img,image_name)
        rotatecw(img,image_name)
        rotateccw(img,image_name)
        # scale(img,image_name)
        fliplr(img,image_name)
        flipud(img,image_name)
    else:
        continue    

f = open(directory+"filelist.txt", "w")

dlist=os.listdir(directory)
dlist.sort()
for filename in dlist:
    if filename.endswith(".jpg") or filename.endswith(".png"):
        #print(os.path.join(directory, filename))
        f.write(filename+"\n")
    else:
        continue
f.close()