from model_fpn import D2N
from torch.autograd import Variable
from torchvision.utils import save_image
import argparse
import cv2
import numpy as np
import os
import timeit
import torch

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Normal image estimation from ToF depth image')
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      default=True,
                      action='store_true')
    parser.add_argument('--num_workers', dest='num_workers',
                      help='num_workers',
                      default=1, type=int)  
    parser.add_argument('--depth_folder', dest='depth_folder',
                      help='path to a single input image for evaluation',
                      default='./dataset/depth_images/depth.png', type=str)
    parser.add_argument('--pred_folder', dest='pred_folder',
                      help='where to save the predicted images.',
                      default='./dataset/predicted_images/', type=str)
    parser.add_argument('--model_path', dest='model_path',
                      help='path to the model to use',
                      default='saved_models/d2n_1_9.pth', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    isExist = os.path.exists(args.pred_folder)
    if not isExist:
        os.makedirs(args.pred_folder)
        print("The new directory for saving images while training is created!")
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You might want to run with --cuda")
    
    # network initialization
    print('Initializing model...')
    d2n = D2N(fixed_feature_weights=False)
    if args.cuda:
        d2n = d2n.cuda()
        
    print('Done!')    
    
    load_name = os.path.join(args.model_path)
    print("loading checkpoint %s" % (load_name))
    state = d2n.state_dict()
    checkpoint = torch.load(load_name)
    checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
    state.update(checkpoint)
    d2n.load_state_dict(state)
    if 'pooling_mode' in checkpoint.keys():
        POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))
    del checkpoint
    torch.cuda.empty_cache()

    d2n.eval()

    img = Variable(torch.FloatTensor(1), volatile=True)

    print('evaluating...')
    with torch.no_grad():
        if not args.depth_folder.endswith('.png'):
            dlist=os.listdir(args.depth_folder)
            dlist.sort()
            time_sum = 0
            counter = 0
            for filename in dlist:
                if filename.endswith(".png"):
                    path=args.depth_folder+filename
                    print("Predicting for:"+filename)
                    depth = cv2.imread(path,cv2.IMREAD_UNCHANGED ).astype(np.float32)
                    if len(depth.shape) < 3:
                        print("Got 1 channel depth images, creating 3 channel depth images")
                        combine_depth = np.empty((depth.shape[0],depth.shape[1], 3))
                        combine_depth[:,:,0] = depth
                        combine_depth[:,:,1] = depth
                        combine_depth[:,:,2] = depth
                        depth = combine_depth
                    depth2 = np.moveaxis(depth,-1,0)
                    img = torch.from_numpy(depth2).float().unsqueeze(0)
                    start = timeit.default_timer()
                    z_fake = d2n(img.cuda())
                    stop = timeit.default_timer()
                    time_sum=time_sum+stop-start
                    counter=counter+1
                    zfv=z_fake*2-1
                    z_fake_norm=zfv.pow(2).sum(dim=1).pow(0.5).unsqueeze(1)
                    zfv=zfv/z_fake_norm
                    z_fake=(zfv+1)/2
                    save_path=args.pred_folder+filename[:-4]
                    save_image(z_fake[0], save_path +"_pred"+'.png')
                else:
                    continue
            print('Predicting '+str(counter)+' images took ', time_sum/counter)  
        else:
            depth = cv2.imread(args.depth_folder,cv2.IMREAD_UNCHANGED).astype(np.float32)
            if len(depth.shape) < 3:
                print("Got 1 channel depth images, creating 3 channel depth images")
                combine_depth = np.empty((depth.shape[0],depth.shape[1], 3))
                combine_depth[:,:,0] = depth
                combine_depth[:,:,1] = depth
                combine_depth[:,:,2] = depth
                depth = combine_depth
            depth2 = np.moveaxis(depth,-1,0)
            img = torch.from_numpy(depth2).float().unsqueeze(0)
            start = timeit.default_timer()
            z_fake = d2n(img.cuda())
            stop = timeit.default_timer()
            zfv=z_fake*2-1
            z_fake_norm=zfv.pow(2).sum(dim=1).pow(0.5).unsqueeze(1)
            zfv=zfv/z_fake_norm
            z_fake=(zfv+1)/2
            dirname, basename = os.path.split(args.depth_folder)
            save_path=args.pred_folder+basename[:-4]
            save_image(z_fake[0], save_path +"_pred"+'.png')
            print('Predicting the image took ', stop-start)
    
