import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from correlation import Correlation
import os
os.environ["CUDA_VISIBLE_DEVICES"]='6'

if __name__ == '__main__':
    b, c, h, w = 1, 256, 78, 78
    kH, kW = 9, 9
    x = torch.rand(b, c, h, w).cuda()
    y = torch.rand(b, c, h, w).cuda()
    m = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1)
    m.cuda()
    z = m(x,y)
    print(z.size())



