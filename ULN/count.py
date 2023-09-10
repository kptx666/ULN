from model.model import ULN
from thop import profile
import torch
from model_summary import get_model_flops

# input LR x2, HR size is 720p
# summary(model, torch.zeros((1, 3, 640, 360)))

# input LR x3, HR size is 720p
# summary(model, torch.zeros((1, 3, 426, 240)))

# input LR x4, HR size is 720p
# summary(model, torch.zeros((1, 3, 320, 180)))

model = ULN(upscale_factor=4)


net_cls_str = f'{model.__class__.__name__}'

# thop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = torch.randn(1, 3, 320, 180).to(device)
flops, params = profile(model, (inputs, ))
print(f'Network: {net_cls_str}, with flops(1280 x 720): {flops/1e9:.2f} GMac, with active parameters: {params/1e3} K.')


