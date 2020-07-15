import argparse
import os
import torch
from datas import f_SonyTestDataset
from torch.utils.data import DataLoader
from models import UDDnet2,preUnetv1,Bottleneck
import scipy.io
from tqdm import tqdm
import numpy as np



def get_detail(x,i,j):
    h_x = x.size()[2]
    w_x = x.size()[3]
    
    illumap = x[:,0,:,:] + x[:,1,:,:] + x[:,2,:,:] + x[:,3,:,:]
    illumap = illumap/3
    
    gt_guih = torch.pow((illumap[:,1:,:]-illumap[:,:h_x-1,:]),2)

    hzero = torch.full([1,1,w_x],0).cuda()
    gt_guih = torch.cat((gt_guih,hzero),1)

    gt_guiw = torch.pow((illumap[:,:,1:]-illumap[:,:,:w_x-1]),2)
    wzero = torch.full([1,h_x,1],0).cuda()
    gt_guiw = torch.cat((gt_guiw,wzero),2)
    gt_guih = gt_guih + gt_guiw
    gt_guih = gt_guih*1e3
    
    gt_guihm = torch.cat((gt_guih,gt_guih,gt_guih,gt_guih),-3)
   
    return gt_guihm 


def test(args):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # data
    testset = f_SonyTestDataset(args.input_dir, args.gt_dir)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    pretrain = preUnetv1()
    pretrain.load_state_dict(torch.load(args.pre))
    pretrain.to(device)
    pretrain.eval()
    
    model = UDDnet2(Bottleneck3)
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()


    # testing
    for i, databatch in tqdm(enumerate(test_loader), total=len(test_loader)):
        input_full, scale_full, gt_full, test_id, sid,ratio = databatch
        scale_full, gt_full = torch.squeeze(scale_full), torch.squeeze(gt_full)

        # processing
        inputs = input_full.to(device)
        pre = pretrain(inputs)
        torch.cuda.empty_cache()
        detail = get_detail(pre,test_id[0],ratio[0])
        detail = torch.unsqueeze(detail, 0)
        inunet = torch.cat((pre,inputs),1)
        outputs = model(inunet,detail)
        outputs = outputs.cpu().detach()
        outputs = torch.squeeze(outputs)
        outputs = outputs.permute(1, 2, 0)

        # scaling can clipping
        outputs, scale_full, gt_full = outputs.numpy(), scale_full.numpy(), gt_full.numpy()
        scale_full = scale_full * np.mean(gt_full) / np.mean(
            scale_full)  # scale the low-light image to the same mean of the ground truth
        outputs = np.minimum(np.maximum(outputs, 0), 1)

        # saving
        if not os.path.isdir(os.path.join(args.result_dir, 'eval2')):
            os.makedirs(os.path.join(args.result_dir, 'eval2'))
        scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            os.path.join(args.result_dir, 'eval2', '%05d_%02d_train_%d_scale.jpg' % (test_id[0],sid[0], ratio[0])))
        scipy.misc.toimage(outputs * 255, high=255, low=0, cmin=0, cmax=255).save(
            os.path.join(args.result_dir, 'eval2', '%05d_%02d_train_%d_out.jpg' % (test_id[0],sid[0], ratio[0])))
        scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            os.path.join(args.result_dir, 'eval2', '%05d_%02d_train_%d_gt.jpg' % (test_id[0],sid[0], ratio[0])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluating model")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--input_dir', type=str, default='../dataset/Sony/short/')
    parser.add_argument('--gt_dir', type=str, default='../dataset/Sony/long/')
    parser.add_argument('--result_dir', type=str, default='test/')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0, help='multi-threads for data loading')
    #parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--pre', type=str, default='pretrain_model.pl')
    parser.add_argument('--model', type=str, default='enhance_model.pl')
    args = parser.parse_args()

    # Create Output Dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    test(args)
