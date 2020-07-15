from __future__ import division
import os, scipy.io
import numpy as np
import logging
import argparse
import sys
from datas import f_SonyDataset 
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models import UDDnet2 #Unet densenet(denoised + ori)
from models import preUnetv1,Bottleneck
from datetime import datetime
from tensorboardX import SummaryWriter

      
def get_detail(x,i,j):
    b = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    
    illumap = x[:,0,:,:] + x[:,1,:,:] + x[:,2,:,:] + x[:,3,:,:]
    illumap = illumap/3
    r = 5
    e = 0.01
    
    gt_guih = torch.pow((illumap[:,1:,:]-illumap[:,:h_x-1,:]),2)
    
    hzero = torch.full([b,1,512],0).cuda()
    gt_guih = torch.cat((gt_guih,hzero),1)
   
    gt_guiw = torch.pow((illumap[:,:,1:]-illumap[:,:,:w_x-1]),2)
    wzero = torch.full([b,512,1],0).cuda()
    gt_guiw = torch.cat((gt_guiw,wzero),2)

    gt_guih = gt_guih + gt_guiw
    gt_guih = gt_guih*1e3
    gt_guih = torch.unsqueeze(gt_guih, 1)
    
    gt_guihm = torch.cat((gt_guih,gt_guih,gt_guih,gt_guih),-3)
   
    
    return gt_guihm 

def train(args):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print('cuda ok')
     
    trainset = SonyDataset(args.input_dir, args.gt_dir, args.ps)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    logging.info("data loading okay")


    model = UDDnet2(Bottleneck)
    model.to(device)
    
    pretrain = preUnetv1()
    pretrain.load_state_dict(torch.load(args.model))
    pretrain.to(device)

    # loss function
    criterion = nn.L1Loss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # lr scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.1)

    # training
    running_loss = 0.0
    for epoch in range(args.num_epoch):
        scheduler.step()
        for i, databatch in enumerate(train_loader):
            # get the inputs
            input_patch, gt_patch, train_id, ratio = databatch
            input_patch, gt_patch = input_patch.to(device), gt_patch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pre = pretrain(input_patch)
            detail = get_detail(pre,train_id[0],ratio[0])
            
            inunet = torch.cat((pre,input_patch),1)
 
            outputs = model(inunet,detail)
            
            l1_loss = criterion(outputs, gt_patch)
            loss = l1_loss
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            if i % args.log_interval == (args.log_interval - 1):
                print('[%d, %5d] total_loss: %.4f   %s' %
                      (epoch, i, running_loss / args.log_interval,  datetime.now()))
                writer.add_scalar('Traindis/total_loss',running_loss / args.log_interval,i+160*epoch)
                running_loss = 0.0
                

            if epoch % args.save_freq == 0:
                if not os.path.isdir(os.path.join(args.result_dir, '%04d' % epoch)):
                    os.makedirs(os.path.join(args.result_dir, '%04d' % epoch))
                
                gt_patch = gt_patch.cpu().detach().numpy()
                outputs = outputs.cpu().detach().numpy()
                train_id = train_id.numpy()
                ratio = ratio.numpy()

                temp = np.concatenate((gt_patch[0, :, :, :], outputs[0, :, :, :]), axis=2)
                scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                    args.result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id[0], ratio[0]))

        # at the end of epoch
        if epoch % args.model_save_freq == 0:
            torch.save(model.state_dict(), args.checkpoint_dir + './model_%d.pl' % epoch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="command for training network")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='../dataset/Sony/short/')
    parser.add_argument('--gt_dir', type=str, default='../dataset/Sony/long/')
    parser.add_argument('--checkpoint_dir', type=str, default='result/')
    parser.add_argument('--result_dir', type=str, default='result/')
    parser.add_argument('--ps', type=int, default=512)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--save_freq', type=int, default=15) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epoch', type=int, default=3001)
    parser.add_argument('--model_save_freq', type=int, default=100)
    parser.add_argument('--model', type=str, default='pretrain_model.pl')

    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    writer = SummaryWriter('Sony')
    # Set Logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        filename=os.path.join(args.result_dir, 'log.txt'),
                        filemode='w')
    # Define a new Handler to log to console as well
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)

    # Start training
    logging.info(" ".join(sys.argv))
    logging.info(args)
    logging.info("using device %s" % str(args.gpu))
    train(args)
    writer.export_scalars_to_json("./all_saclars.json")
    writer.close()
