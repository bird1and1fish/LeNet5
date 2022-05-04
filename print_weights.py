import torch
from net import LeNet
from torchsummary import summary
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet().to(device)
model.load_state_dict(torch.load("D:/ws_pytorch/LeNet5/save_model/best_model.pth"))

summary(model, (1, 28, 28))

for name in model.state_dict():
    print(name)

print("##############c1_weight###############")
print(model.state_dict()['c1.weight'])
print("##############c1_bias###############")
print(model.state_dict()['c1.bias'])
print("##############c3_weight###############")
print(model.state_dict()['c3.weight'])
print("##############c3_bias###############")
print(model.state_dict()['c3.bias'])
print("##############c5_weight###############")
print(model.state_dict()['c5.weight'])
print("##############c5_bias###############")
print(model.state_dict()['c5.bias'])
print("##############f6_weight###############")
print(model.state_dict()['f6.weight'])
print("##############f6_bias###############")
print(model.state_dict()['f6.bias'])
print("##############output_weight###############")
print(model.state_dict()['output.weight'])
print("##############output_bias###############")
print(model.state_dict()['output.bias'])
