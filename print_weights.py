import torch
from net import LeNet
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LeNet().to(device)
model.load_state_dict(torch.load("D:/ws_pytorch/LeNet5/save_model/best_model.pth"))

summary(model, (1, 28, 28))

for name in model.state_dict():
    print("################" + name + "################")
    # print(model.state_dict()[name])
    file = open("weight/" + name + ".txt", "w")
    file.write(str(model.state_dict()[name]))
    file.close()


