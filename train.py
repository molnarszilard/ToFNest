import numpy as np
import os, sys
from model_fpn import D2N
import argparse, time
import torch
from torch.autograd import Variable
from datasetloader import DatasetLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from torchvision.utils import save_image
from collections import Counter
import matplotlib, cv2
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threading import Thread

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Normal image estimation from ToF depth image')
    parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=10, type=int)
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      default=True,
                      action='store_true')
    parser.add_argument('--bs', dest='bs',
                      help='batch_size',
                      default=4, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                      help='num_workers',
                      default=1, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='display interval',
                      default=10, type=int)
    parser.add_argument('--model_dir', dest='model_dir',
                      help='output directory',
                      default='saved_models', type=str)

# config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=1e-3, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)
    parser.add_argument('--loss_multiplier', dest='loss_multiplier',
                      help='increase the loss for much faster training',
                      default=200, type=int)
    parser.add_argument('--loss_md', dest='loss_md',
                      help='decrease the loss multiplier in every epoch',
                      default=True, type=bool)

# set training session
    parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
    parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
    parser.add_argument('--start_at', dest='start_epoch',
                      help='epoch to start with',
                      default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=9, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
    parser.add_argument('--save_epoch', dest='save_epoch',
                      help='after how many epochs do you want the model to be saved',
                      default=5, type=int)

# training parameters
    parser.add_argument('--gamma_sup', dest='gamma_sup',
                      help='factor of supervised loss',
                      default=1., type=float)
    parser.add_argument('--gamma_unsup', dest='gamma_unsup',
                      help='factor of unsupervised loss',
                      default=1., type=float)
    parser.add_argument('--gamma_reg', dest='gamma_reg',
                      help='factor of regularization loss',
                      default=10., type=float)
    parser.add_argument('--orient_normals', dest='orient_normals',
                      help='do you want to have your normals oriented?',
                      default=False, type=bool)
    parser.add_argument('--save_images', dest='save_images',
                      help='save every 100th image during the training to see its evolution',
                      default=True, type=bool)
    parser.add_argument('--dir_images', dest='dir_images',
                      help='directory where to save the training images',
                      default='training_images/', type=str)

    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']

class VectorLoss(nn.Module):
    def __init__(self):
        super(VectorLoss, self).__init__()

    def forward(self, pred, gt):
        pred=pred*2-1
        gt=gt*2-1
        inner_product = (pred * gt).sum(dim=1).unsqueeze(1)
        cos = inner_product / 2
        angle = torch.acos(cos)
        if not args.orient_normals:
            angle[angle>1.57]=3.14-angle[angle>1.57] 
        loss = torch.mean(angle)
        return loss
        
def resize_tensor(img, coords):
    return nn.functional.grid_sample(img, coords, mode='bilinear', padding_mode='zeros')

if __name__ == '__main__':

    args = parse_args()

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You might want to run with --cuda")

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        
    train_dataset = DatasetLoader()
    train_size = len(train_dataset)
    print(train_size)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs,
                            shuffle=True, num_workers=args.num_workers)

    # network initialization
    print('Initializing model...')
    d2n = D2N(fixed_feature_weights=False)
    if args.cuda:
        d2n = d2n.cuda()
        
    print('Done!')

    # hyperparams
    lr = args.lr
    bs = args.bs
    lr_decay_step = args.lr_decay_step
    lr_decay_gamma = args.lr_decay_gamma

    # params
    params = []
    for key, value in dict(d2n.named_parameters()).items():
      if value.requires_grad:
        if 'bias' in key:
            DOUBLE_BIAS=0
            WEIGHT_DECAY=4e-5
            params += [{'params':[value],'lr':lr*(DOUBLE_BIAS + 1), \
                  'weight_decay': 4e-5 and WEIGHT_DECAY or 0}]
        else:
            params += [{'params':[value],'lr':lr, 'weight_decay': 4e-5}]

    # optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=4e-5)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)

    vector_loss = VectorLoss()
    
    # resume
    if args.resume:
        load_name = os.path.join(args.model_dir,
          'd2n_1_{}.pth'.format(args.checkepoch))
        print("loading checkpoint %s" % (load_name))
        state = d2n.state_dict()
        checkpoint = torch.load(load_name)
        args.start_epoch = checkpoint['epoch']
        checkpoint = {k: v for k, v in checkpoint['model'].items() if k in state}
        state.update(checkpoint)
        d2n.load_state_dict(state)
        if 'pooling_mode' in checkpoint.keys():
            POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        del checkpoint
        torch.cuda.empty_cache()

    # constants
    iters_per_epoch = int(train_size / args.bs)
    loss_multiplier_decrease=1
    for epoch in range(args.start_epoch, args.max_epochs):
        
        # setting to train mode
        d2n.train()
        start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        img = Variable(torch.FloatTensor(1))
        z = Variable(torch.FloatTensor(1))
        if args.cuda:
            img = img.cuda()
            z = z.cuda()
        
        train_data_iter = iter(train_dataloader)
        for step in range(iters_per_epoch):
            if args.resume:
                delay_loss=args.checkepoch
            else:
                delay_loss = 0
            data = train_data_iter.next()
            
            img.resize_(data[0].size()).copy_(data[0])
            z.resize_(data[1].size()).copy_(data[1])
            optimizer.zero_grad()
            z_fake = d2n(img)
         
            zfv=z_fake*2-1
            z_fake_norm=zfv.pow(2).sum(dim=1).pow(0.5).unsqueeze(1)      
            zfv=zfv/z_fake_norm
            z_fake=(zfv+1)/2

            zv=z*2-1
            z_norm=zv.pow(2).sum(dim=1).pow(0.5).unsqueeze(1)
            zv=zv/z_norm
            z=(zv+1)/2

            vloss_train = vector_loss(z_fake, z)
            
            loss = vloss_train*args.loss_multiplier*loss_multiplier_decrease
            loss.backward()
            optimizer.step()            

            end = time.time()

            # info
            if step % args.disp_interval == 0:

                print("[epoch %2d][iter %4d] loss: %.4f vector_loss: %.4f" \
                                % (epoch, step, loss, vloss_train))
                if args.save_images and step%100==0:
                    if not os.path.exists(args.dir_images):
                        os.makedirs(args.dir_images)
                    save_image(z_fake[0], args.dir_images+'trainingpred_'+str(epoch)+'_'+str(step)+'.png')

        if args.loss_md:
            loss_multiplier_decrease=loss_multiplier_decrease*0.9
        if epoch%args.save_epoch==0 or epoch==args.max_epochs-1:
            if not os.path.exists(args.model_dir):
                        os.makedirs(args.model_dir)
            save_name = os.path.join(args.model_dir, 'd2n_{}_{}.pth'.format(args.session, epoch))
            torch.save({'epoch': epoch+1,
                    'model': d2n.state_dict(), 
                   },
                   save_name)

            print('save model: {}'.format(save_name))
        print('time elapsed: %fs' % (end - start))   
