import torch
from torch import nn
import torch.nn.functional as F

# 定义量化和反量化函数
def quantize_tensor(x, num_bits=8):
    qmin = 0.
    qmax = 2.**num_bits - 1.
    min_val, max_val = x.min(), x.max()

    scale = (max_val - min_val) / (qmax - qmin)

    initial_zero_point = qmin - min_val / scale

    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    return q_x, scale, zero_point

def dequantize_tensor(q_x, scale, zero_point):
    return scale * (q_x.float() - zero_point)

# 定义量化卷积和量化全连接
class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_flag = False
        self.scale = None
        self.zero_point = None

    def linear_quant(self, quantize_bit=8):
        self.weight.data, self.scale, self.zero_point = quantize_tensor(self.weight.data, num_bits=quantize_bit)
        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            weight = dequantize_tensor(self.weight, self.scale, self.zero_point)
            return F.linear(x, weight, self.bias)
            # return F.linear(x, self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QuantConv2d, self).__init__(in_channels, out_channels,
                                          kernel_size, stride, padding, dilation, groups, bias)
        self.quant_flag = False
        self.scale = None
        self.zero_point = None

    def linear_quant(self, quantize_bit=8):
        self.weight.data, self.scale, self.zero_point = quantize_tensor(self.weight.data, num_bits=quantize_bit)
        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            weight = dequantize_tensor(self.weight, self.scale, self.zero_point)
            return F.conv2d(x, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
            # return F.conv2d(x, self.weight, self.bias, self.stride,
            #                 self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

# 定义网络模型
class LeNet(nn.Module):
    # 初始化网络
    def __init__(self):
        super(LeNet, self).__init__()

        self.c1 = QuantConv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Sigmoid = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = QuantConv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = QuantConv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = QuantLinear(120, 84)
        self.output = QuantLinear(84, 10)

    def forward(self, x):
        x = self.Sigmoid(self.c1(x))
        x = self.s2(x)
        x = self.Sigmoid(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x

    def linear_quant(self, quantize_bit=8):
        # Should be a less manual way to quantize
        # Leave it for the future
        self.c1.linear_quant(quantize_bit)
        self.c3.linear_quant(quantize_bit)
        self.c5.linear_quant(quantize_bit)
        self.f6.linear_quant(quantize_bit)
        self.output.linear_quant(quantize_bit)

if __name__ == "__main__":
    x = torch.rand([1, 1, 28, 28])
    model = LeNet()
    model.linear_quant()
    y = model(x)