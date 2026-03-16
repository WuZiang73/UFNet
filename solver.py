import os
import random
import numpy as np
import scipy.misc as misc
import skimage.measure as measure
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
import cv2 #20190413029tcw
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

def add_noise(lr,mode,noise_level):
    if mode == 1:
    #certain noise, i.e.,0,25,35,50,75
        noise = torch.FloatTensor(lr.size()).normal_(mean=0,std=noise_level/255.)
    if mode == 0:
        noisel_b = [0,55]
        #noisel_b = 0
        noise = torch.zeros(lr.size())
        stdN = np.random.uniform(noisel_b[0],noisel_b[1],size=noise.size()[0])
        #print stdN
        for n in range(noise.size()[0]):
            sizeN = noise[0,:,:,:].size()
            noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0,std=stdN[n]/255.)
    lr_noise = lr+ noise
    return lr_noise 

class Solver():
    def __init__(self, model, cfg):
        if cfg.scale > 0: #there is only a scale, 201904081901
            self.refiner = model(scale=cfg.scale, 
                                 group=cfg.group)
        else: #there is mutile scales,201904081901
            self.refiner = model(multi_scale=True, 
                                 group=cfg.group)
        
        if cfg.loss_fn in ["MSE"]: 
            self.loss_fn = nn.MSELoss()
        elif cfg.loss_fn in ["L1"]: 
            self.loss_fn = nn.L1Loss()
        elif cfg.loss_fn in ["SmoothL1"]:
            self.loss_fn = nn.SmoothL1Loss()

        self.optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.refiner.parameters()), 
            cfg.lr)
        
        self.train_data = TrainDataset(cfg.train_data_path, 
                                       scale=cfg.scale, 
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=0,
                                       shuffle=True, drop_last=True)
        print("Dataset Information:")
        print(f"  Train Data Path: {cfg.train_data_path}")
        print(f"  Scale: {cfg.scale}")
        print(f"  Patch Size: {cfg.patch_size}")
        print(f"  Batch Size: {cfg.batch_size}")
        print(f"  Number of Workers: 0")  # 请注意，这里使用了硬编码的0，你可能需要根据实际情况进行调整
        print(f"  Shuffle: False")
        print(f"  Drop Last: True")
        
        #the ways of chosen GPU
        #the first way
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("cuda")
        # os.environ['CUDA_VISIBLE_DEVICES']='0,1'
        # self.device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu") #tcw201904100941, cuda:1 denotes the GPU of number 1. cuda:0 denotes the GPU of number 0.
        #automically choose the GPU, if torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #"The second way is as follows--------------------------"
        #self.device = torch.device('cuda',1) #the commod is added by tcw 201904100942
        #If torch.device('cuda',1), which chooses the GPU of number 1. If torch.device('cuda',0), which chooses the GPU of number 0.
        self.refiner = self.refiner.to(self.device) #load the model into the self.device
        self.loss_fn = self.loss_fn

        self.cfg = cfg
        self.step = 0
        
        self.writer = SummaryWriter(log_dir=os.path.join("runs", cfg.ckpt_name)) #log
        if cfg.verbose:
            num_params = 0
            for param in self.refiner.parameters():#model.parameters keep the parameters from all the layers.
                num_params += param.nelement() #.nelement() can count the number of all the parameters.
            print("# of params:", num_params)
        
        if not os.path.exists(cfg.ckpt_dir): #201904072208 tcw
            #os.makedirs(cfg.ckpt_dir, exist_ok=True) #201904072211tcw, it is given at first, but it is wrong. So, I mark it.
            os.makedirs(cfg.ckpt_dir, mode=0o777) #2019072211tcw

    def fit(self):
        cfg = self.cfg
        #last_epoch = 0
        #for ckpoint in os.listdir(cfg.ckpt_dir):
        #    epoch = int(ckpoint.replace(cfg.ckpt_name, "").replace(".pth", "").replace("_", ""))
        #    if epoch > last_epoch:
        #        last_epoch = epoch
        #self.load(os.path.join(
        #    cfg.ckpt_dir, "{}_{}.pth".format(cfg.ckpt_name, last_epoch)))

        refiner = nn.DataParallel(self.refiner,
                                  device_ids=range(cfg.num_gpu))

        learning_rate = cfg.lr
        while True:
            for inputs in tqdm(self.train_loader):
                self.refiner.train()
                if cfg.scale > 0:  # There is only a scale in the training processing.
                    scale = cfg.scale
                    hr, lr = inputs[-1][0], inputs[-1][1]  #
                else:
                    # only use one of multi-scale data
                    # i know this is stupid but just temporary. it is noticeable that scale is rand.
                    scale = random.randint(2, 4)
                    hr, lr = inputs[scale - 2][0], inputs[scale - 2][1]  # obatin hr,lr under differet scales.
                lr_noise = add_noise(lr, 1, 25)
                hr = hr.to(self.device)  # load hr on the self.device
                lr = lr.to(self.device)
                lr_noise = lr_noise.to(self.device)
                i = 0
                sr = refiner(scale, lr_noise)
                if isinstance(sr, tuple):
                    sr = sr[1]
                loss1 = self.loss_fn(sr, hr)
                # loss2 = self.loss_fn(lr_denoise, lr)
                # loss = loss1+0.5*loss2
                loss = loss1
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(self.refiner.parameters(),
                                        cfg.clip)  # tcw it is drop out, which can prevent overfitting.
                self.optim.step()

                learning_rate = self.decay_learning_rate()
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate

                self.step += 1
                if cfg.verbose and self.step % cfg.print_interval == 0:
                    if cfg.scale > 0:
                        psnr = self.evaluate("datasets/Urban100", scale=cfg.scale, num_step=self.step)
                        # print 'sdffffffffffff232'
                        self.writer.add_scalar("Urban100", psnr,
                                               self.step)  # save the data in the file of writer, which is shown via visual figures.
                        # The first parameter is figure name, the second parameter is axis Y, the third parameter is axis X.
                    else:
                        psnr = [self.evaluate("datasets/Urban100", scale=i, num_step=self.step) for i in range(2, 5)]
                        self.writer.add_scalar("Urban100_2x", psnr[0], self.step)
                        self.writer.add_scalar("Urban100_3x", psnr[1], self.step)
                        self.writer.add_scalar("Urban100_4x", psnr[2], self.step)

                    self.save(cfg.ckpt_dir, cfg.ckpt_name)

            if self.step > cfg.max_steps: break
    def evaluate(self, test_data_dir, scale=3, num_step=0):
        print("eva")
        global mean_psnr, mean_psnr1, mean_ssim
        mean_psnr = 0
        mean_psnr1 = 0
        mean_ssim = 0
        cfg = self.cfg
        self.refiner.eval()
        test_data = TestDataset(test_data_dir, scale=scale)
        test_loader = DataLoader(test_data,
                                batch_size=1,
                                num_workers=0,
                                shuffle=False)

        for step, inputs in enumerate(test_loader):
            hr = inputs[0].squeeze(0)
            lr = inputs[1].squeeze(0)
            name = inputs[2][0]
            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_noise = add_noise(lr_patch, 0, 0)
            lr_patch = lr_patch.to(self.device)
            lr_noise = lr_noise.to(self.device)

            with torch.no_grad():
                sr = self.refiner(scale, lr_noise)
                if isinstance(sr, tuple):
                    sr = sr[1]
                sr = sr.data

            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale

            # merge splited patch images
            result = torch.FloatTensor(3, h, w).to(self.device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result

            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()

            hr = rgb2ycbcr(hr)
            sr = rgb2ycbcr(sr)
            bnd = scale

            # 自动裁剪为最小公共区域，避免尺寸不一致
            min_h = min(hr.shape[0], sr.shape[0]) - bnd
            min_w = min(hr.shape[1], sr.shape[1]) - bnd
            im1 = hr[bnd:min_h, bnd:min_w]
            im2 = sr[bnd:min_h, bnd:min_w]

            # 再次保证最终尺寸一致
            final_h = min(im1.shape[0], im2.shape[0])
            final_w = min(im1.shape[1], im2.shape[1])
            im1 = im1[:final_h, :final_w]
            im2 = im2[:final_h, :final_w]

            mean_psnr += psnr(im1, im2) / len(test_data)
            mean_ssim += calculate_ssim(im1, im2) / len(test_data)

        print('epochs is %d, mean_psnr is %f, mean_ssim is %f' % (self.step, mean_psnr, mean_ssim))
        return mean_psnr
    def load(self, path):
        checkpoint = torch.load(path)
        self.refiner.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['epoch']
        self.decay_learning_rate()
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        #print 'sfdfsdsfsdf'
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step))
        torch.save({
            'epoch': self.step,
            'model_state_dict': self.refiner.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, save_path)

    def decay_learning_rate(self):
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay))
        return lr

def rgb2ycbcr(img, only_y=True):  #201904122348tcw
    '''
    same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)

def psnr(im1, im2):
    def im2double(im):
        min_val, max_val = 0, 255
        out = (im.astype(np.float64)-min_val) / (max_val-min_val)
        return out
        
    im1 = im2double(im1)
    im2 = im2double(im2)
    # 保证尺寸一致
    final_h = min(im1.shape[0], im2.shape[0])
    final_w = min(im1.shape[1], im2.shape[1])
    im1 = im1[:final_h, :final_w]
    im2 = im2[:final_h, :final_w]
    psnr = peak_signal_noise_ratio(im1, im2, data_range=1)
    return psnr

def calculate_ssim(img1, img2, border=0):
    if not img1.shape == img2.shape:
        # 自动裁剪为最小公共区域
        final_h = min(img1.shape[0], img2.shape[0])
        final_w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:final_h, :final_w]
        img2 = img2[:final_h, :final_w]
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
