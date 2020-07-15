import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

class DoubleConv(nn.Module):
    #  Conv--> LReLU **2 (without BN)
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.f(x)
        return x
 
class CALayer(nn.Module):
    def __init__(self,in_ch,reduction=16):
        super(CALayer,self).__init__()
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch,in_ch // reduction,1),
            nn.Conv2d(in_ch // reduction,in_ch,1),
            nn.Sigmoid(),
        )
        
    def forward(self,x):
        return x * self.a(x)

class RCAB(nn.Module):
    def __init__(self,in_ch,reduction=16):
        super(RCAB, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_ch,in_ch,3,padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch,in_ch,3,padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            CALayer(in_ch,reduction),
        #nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self,x):
        res = self.res(x) + x
        return res



class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.f = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.f(x)
        return x

    
class Up2(nn.Module):
    # upsample and concat

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.Conv2d(in_ch, in_ch//2,3, padding=1)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1,scale_factor=2,mode='bilinear',align_corners=False)
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    

class Up(nn.Module):
    # upsample and concat

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.f = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.f(x)
        return x


class DepthToSpace(nn.Module):
    # copied from https://gist.github.com/jalola/f41278bb27447bed9cd3fb48ec142aec
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

 

      
class preUnetv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(4, 32)
        self.d1 = Down(32, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)

        self.u1 = Up(512, 256)
        self.u2 = Up(256, 128)
        self.u3 = Up(128, 64)
        self.u4 = Up(64, 32)
        self.outc = OutConv(32, 4)
        

    def forward_f(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        x = self.outc(x)
        #x = torch.sigmoid(x)

        return x
    
    def forward(self,x):
        if self.training:
            return self.forward_f(x)
        else: 
            with torch.no_grad():
                return self.forward_f(x)

              
class preUnetv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(4, 32)
        self.d1 = Down(32, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)

        self.u1 = Up2(512, 256)
        self.u2 = Up2(256, 128)
        self.u3 = Up2(128, 64)
        self.u4 = Up2(64, 32)
        self.outc = OutConv(32, 4)
        

    def forward_f(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        x = self.outc(x)
        #x = torch.sigmoid(x)

        return x
    
    def forward(self,x):
        if self.training:
            return self.forward_f(x)
        else: 
            with torch.no_grad():
                return self.forward_f(x)
 
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(4, 32)
        self.d1 = Down(32, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)

        self.u1 = Up(512, 256)
        self.u2 = Up(256, 128)
        self.u3 = Up(128, 64)
        self.u4 = Up(64, 32)
        self.outc = OutConv(32, 12)
        self.d2s = DepthToSpace(2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        x = self.outc(x)
        x = self.d2s(x)
        x = torch.sigmoid(x)

        return x
    
    
       
class Unet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(4, 32)
        self.d1 = Down(32, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)

        self.u1 = Up2(512, 256)
        self.u2 = Up2(256, 128)
        self.u3 = Up2(128, 64)
        self.u4 = Up2(64, 32)
        self.outc = OutConv(32, 12)
        self.d2s = DepthToSpace(2)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        x = self.outc(x)
        x = self.d2s(x)
        x = self.relu(x)
        #x = torch.sigmoid(x)

        return x
    
    
    

class UDF_Unet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(8, 32)
        self.d1 = Down(32, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)

        self.u1 = Up2(512, 256)
        self.u2 = Up2(256, 128)
        self.u3 = Up2(128, 64)
        self.u4 = Up2(64, 32)
        self.outc = OutConv(32, 12)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        x5 = self.d4(x4)
        x = self.u1(x5, x4)
        x = self.u2(x, x3)
        x = self.u3(x, x2)
        x = self.u4(x, x1)
        x = self.outc(x)
        x = self.relu(x)

        return x

    
   
class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck3, self).__init__()
        #self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=3,padding = 1, bias=False)
        #self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=2,dilation=2, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(x))
        out = self.conv2(F.relu(out))
        
        # input and output are concatenated here
        out = torch.cat([out,x], 1)
        return out

    
class Transition(nn.Module):
    '''
        #transition layer is used for down sampling the feature i do not down sample 
        
        when compress rate is 0.5, out_planes is a half of in_planes
    '''
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        
        out = self.conv(F.relu(self.bn(x)))
        # use average pooling change the size of feature map here
        #out = F.avg_pool2d(out, 2)
        return out 

class DenseNet(nn.Module):
    def __init__(self, block, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        '''
        Args:
            block: bottleneck
            nblock: a list, the elements is number of bottleneck in each denseblock
            growth_rate: channel size of bottleneck's output
            reduction: 
        '''
        self.growth_rate = growth_rate

        num_planes = 12
        self.conv1 = nn.Conv2d(4, num_planes, kernel_size=3, padding=1, bias=False)
        
        # a DenseBlock and a transition layer
        self.dense1 = self._make_dense_layers(block, num_planes, 4)
        num_planes += 4*growth_rate
        #fuse the channel 
        self.fuse = nn.Conv2d(num_planes,24,kernel_size=1, bias=False)
        #out put
        self.out = nn.Conv2d(24,12,kernel_size=3, padding=1, bias=False)
        self.out2 = nn.Conv2d(12,3,kernel_size=3, padding=1, bias=False)
        # the channel size is superposed, mutiply by reduction to cut it down here, the reduction is also known as compress rate
        #out_planes = int(math.floor(num_planes*reduction))
       

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        
        # number of non-linear transformations in one DenseBlock depends on the parameter you set
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        out = self.conv1(x) # input 4 channel to 12
        out = self.dense1(out) #4 denseblocks 12channel to 60 channel 
        out = self.fuse(out)# 60-24
        out = self.out(out)#24-12
        out = self.out2(out)#12-3
        return out
    
def densenet():
    return DenseNet(Bottleneck)
    
    
class UDF_DenseNet4(nn.Module):
    def __init__(self, block, growth_rate=12, reduction=0.5, num_classes=10):
        super(UDF_DenseNet4, self).__init__()
        '''
        Args:
            block: bottleneck
            nblock: a list, the elements is number of bottleneck in each denseblock
            growth_rate: channel size of bottleneck's output
            reduction: 
        '''
        self.growth_rate = growth_rate

        num_planes = 12
        self.conv1 = nn.Conv2d(4, num_planes, kernel_size=3, padding=1, bias=False)
        
        # a DenseBlock and a transition layer
        self.dense1 = self._make_dense_layers(block, num_planes, 5)
        num_planes += 5*growth_rate
        #fuse the channel 
        self.fuse1 = nn.Conv2d(num_planes,24,kernel_size=3,padding=1, bias=False)
        num_planes = 24 
        self.dense2 = self._make_dense_layers(block, num_planes, 5)
        num_planes += 5*growth_rate
        self.fuse2 = nn.Conv2d(num_planes,60,kernel_size=3,padding=1, bias=False)
        self.fuse3 = nn.Conv2d(60,24,kernel_size=3,padding=1, bias=False)
        #out put
        self.out = nn.Conv2d(24,12,kernel_size=3, padding=1, bias=False)
        
        # the channel size is superposed, mutiply by reduction to cut it down here, the reduction is also known as compress rate
        #out_planes = int(math.floor(num_planes*reduction))
       

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        
        # number of non-linear transformations in one DenseBlock depends on the parameter you set
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        #x = F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        out = self.conv1(x) # input 4 channel to 12
        out = self.dense1(out) #4 denseblocks 12channel to 72 channel 
        out = self.fuse1(out)# 60-24
        out = self.dense2(out)#24-120
        out = self.fuse2(out)#120-60
        out = self.fuse3(out)#60-24
        out = self.out(out)#24-12
        return out
    

    


class UDDnet2(nn.Module):
    def __init__(self,block):
        super(UDDnet2, self).__init__()

        
        self.local_net = UDF_DenseNet4(block) #12channel

        self.mid_net = UDF_Unet2() #12channel


        self.end_net = nn.Sequential(
            nn.Conv2d(24, 12, kernel_size=1),  nn.Conv2d(12,12,kernel_size=3,padding=1),nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.up = DepthToSpace(2)
        
        
    def get_featureconv1(self,image):
        
        output = self.mid_net.inc(image)
        output = self.mid_net.d1(output)
        
        return output

    def forward_f(self, x,d):       
        local = self.local_net(d)
        mid = self.mid_net(x)
        fuse = torch.cat((local, mid), -3)
        fuse = self.end_net(fuse)
        fuse = self.up(fuse)
        
        
        return fuse
    
    def forward(self,x,d):
        if self.training:
            return self.forward_f(x,d)
        else: #测试模式 需要将输入切块
            with torch.no_grad():
                return self.forward_f(x,d)
 
