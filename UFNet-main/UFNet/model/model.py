import torch
import torch.nn as nn
import model.ops as ops
#import model.switchable_norm as sn

class noise(nn.Module):
    def __init__(self,in_channels,out_channels,groups=1):
        super(noise,self).__init__()
        kernel_size = 3 
        padding = 1 
        features = 64
        #sn.SwitchNorm2d(features)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=in_channels,out_channels=features,kernel_size=kernel_size,padding=padding, groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_10= nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_11= nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_12= nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_13= nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_14= nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_15= nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_16= nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features, kernel_size=kernel_size, padding= padding, groups=1,bias= False),nn.ReLU(inplace=True))
        self.conv1_17= nn.Conv2d(in_channels=features,out_channels=out_channels,kernel_size=kernel_size,padding=padding,groups=1,bias=False)
    def forward(self,input):
        x1 = self.conv1_1(input)
        x2 = self.conv1_2(x1)
        x2 = x1+x2
        x3 = self.conv1_3(x2)
        x4 = self.conv1_4(x3)
        x4 = x4+x3
        x5 = self.conv1_5(x4)
        x6 = self.conv1_6(x5)
        x6 = x6+x5
        x7 = self.conv1_7(x6)
        x8 = self.conv1_8(x7)
        x8=x8+x7
        x9 = self.conv1_9(x8)
        x10 = self.conv1_10(x9)
        x10 = x10+x9
        x11 = self.conv1_11(x10)
        x12 = self.conv1_12(x11)
        x12=x12+x11
        x13 = self.conv1_13(x12)
        x14 = self.conv1_14(x13)
        x14=x14+x13
        x15 = self.conv1_15(x14)
        x16 = self.conv1_16(x15)
        x16=x16+x15
        x17 = self.conv1_17(x16)
        out =  input-x17
        return out

class SR(nn.Module):
    def __init__(self,features,groups=1):
        super(SR,self).__init__()
        kernel_size = 3 
        padding = 1 
        features = 64
        distill_rate = 0.75
        remaining_rate = 1-distill_rate
        distill_channel = int(distill_rate*features)
        remaining_channel = int(remaining_rate*features)
        self.distill_channel = distill_channel
        self.remaining_channel = remaining_channel
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=distill_channel,out_channels=distill_channel,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=distill_channel,out_channels=distill_channel,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=distill_channel,out_channels=distill_channel,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=remaining_channel,out_channels=remaining_channel,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(in_channels=remaining_channel,out_channels=remaining_channel,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv2_4 = nn.Sequential(nn.Conv2d(in_channels=remaining_channel,out_channels=remaining_channel,kernel_size=kernel_size,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv1_5 = nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False)
        self.ReLU = nn.ReLU(inplace=True)
    def forward(self,input):
        input1 = self.ReLU(input)
        x1 = self.conv1_1(input1)
        dit1,remain1 = torch.split(x1,(self.distill_channel,self.remaining_channel),dim=1)
        remain2 = self.conv2_2(remain1)
        remain3 = self.conv2_3(remain2)
        remain4 = self.conv2_4(remain3)
        remain4 = remain2+remain4
        dit2 = self.conv1_2(dit1)
        dit3 = self.conv1_3(dit2)
        dit4 = self.conv1_4(dit3)
        dit4 = dit2+dit4
        out =  torch.cat([dit4,remain4],dim=1)
        out = self.conv1_5(out)
        out = out+input
        return out        

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        
        scale = kwargs.get("scale") #value of scale is scale. 
        multi_scale = kwargs.get("multi_scale") # value of multi_scale is multi_scale in args.
        group = kwargs.get("group", 1) #if valule of group isn't given, group is 1.
        kernel_size = 3 #tcw 201904091123
        kernel_size1 = 1 #tcw 201904091123
        padding1 = 0 #tcw 201904091124
        padding = 1     #tcw201904091123
        features = 64   #tcw201904091124
        groups = 1       #tcw201904091124
        channels = 3
        features1 = 64
        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.bt = SR(features)
        self.ReLU=nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=1,bias=False)
        self.b1 = SR(features)
        self.b2 = SR(features)
        self.b3 = SR(features)
        self.b4 = SR(features)
        self.b5 = SR(features)
        self.b6 = SR(features)
        self.b7 = SR(features)
        self.b8 = SR(features)
        self.denoiser = noise(channels,channels)
        #self.conv2 = nn.Conv2d(in_channels=features,out_channels=features,kernel_size=1,padding=0,groups=1,bias=False)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=3,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.conv6 = nn.Conv2d(in_channels=features,out_channels=channels,kernel_size=kernel_size,padding=padding,groups=1,bias=False)
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=3,padding=padding,groups=1,bias=False),nn.ReLU(inplace=True))
        self.ReLU = nn.ReLU(inplace=True)
        self.upsample = ops.UpsampleBlock(64, scale=scale, multi_scale=multi_scale,group=1)
    def forward(self,scale,lr_noise):
        xt = self.sub_mean(lr_noise)
        lr_clean = self.denoiser(xt)
        x1 = self.conv1_1(xt)
        b1 = self.b1(x1)
        b2 = self.b2(b1)
        b2 = b1+b2
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b4 = b3+b4
        b5 = self.b5(b4)
        b6 = self.b6(b5)
        b6 = b5+b6
        b7 = self.b7(b6)
        b8 = self.b8(b7)
        b8 = b7 + b8 + x1     
        #btt = self.ReLU(b8)
        #lr_clean,x16 = self.denoiser(xt)
        #lr_clean, lr_features = self.denoiser(lr_noise)
        #btt = btt+lr_features
        #denoiser_features = self.conv2(x16)
        #btt = b8*denoiser_features
        btt = self.ReLU(b8)
        xtcw = self.ReLU(lr_clean)
        xtcw = self.conv7(xtcw)
        btt = xtcw+btt
        temp = self.upsample(btt, scale=scale)
        #temp_t = self.upsample(x16,scale=scale)
        #temp = temp_t + temp
        temp2 = self.ReLU(temp)
        temp3 = self.conv3(temp2)
        temp4 = self.conv4(temp3)
        temp5 = self.conv5(temp4)
        temp6 = temp3+temp4+temp5
        out = self.conv6(temp6)
        out = self.add_mean(out)
        return lr_clean, out
        #return out
        #return out
