import torch
from torch import nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义带掩模矩阵的卷积层
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = mask.clone().detach().requires_grad_(False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    # 保存剪枝后的权重
    def save_mask(self):
        self.weight.data = self.weight.data * self.mask.data

    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight * self.mask
            return F.conv2d(x, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

# 定义带掩模矩阵的全连接层
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask.clone().detach().requires_grad_(False)
        # print(self.weight.data.size())
        # print("分界线")
        # print(self.mask.data.size())
        self.weight.data = self.weight.data * self.mask.data

        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    # 保存剪枝后的权重
    def save_mask(self):
        self.weight.data = self.weight.data * self.mask.data

    def forward(self, x):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

# 定义网络模型
class LeNet(nn.Module):
    # 初始化网络
    def __init__(self):
        super(LeNet, self).__init__()

        self.c1 = MaskedConv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.Relu = nn.ReLU()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = MaskedConv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = MaskedConv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.f6 = MaskedLinear(120, 84)
        self.output = MaskedLinear(84, 10)

    def forward(self, x):
        x = self.Relu(self.c1(x))
        x = self.s2(x)
        x = self.Relu(self.c3(x))
        x = self.s4(x)
        x = self.c5(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = self.output(x)
        return x

    def set_linear_masks(self, masks):
        self.f6.set_mask(masks[0])
        self.output.set_mask(masks[1])

    def set_conv_masks(self, masks):
        self.c1.set_mask(torch.from_numpy(masks[0]))
        self.c3.set_mask(torch.from_numpy(masks[1]))
        self.c5.set_mask(torch.from_numpy(masks[2]))

    def save_masks(self):
        self.c1.save_mask()
        self.c3.save_mask()
        self.c5.save_mask()
        self.f6.save_mask()
        self.output.save_mask()

if __name__ == "__main__":
    x = torch.rand([1, 1, 28, 28])
    model = LeNet()
    y = model(x)