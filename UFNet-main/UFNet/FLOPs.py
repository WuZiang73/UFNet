import torch
from thop import profile
from model.model import Net

if __name__ == "__main__":
    scale = 2
    model = Net(scale=scale)
    model = model.cuda()  
    model.eval()
    input_tensor = torch.randn(1, 3, 640, 360).cuda()
    lr_noise = torch.randn(1, 3, 640, 360).cuda()  # 和 input_tensor 形状一样
    with torch.no_grad():
        flops, params = profile(model, inputs=(input_tensor, lr_noise))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs, Params: {params / 1e6:.4f} M")