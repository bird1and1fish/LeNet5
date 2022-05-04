import torch
from net import LeNet
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

show = ToPILImage()

# 数据转化为tensor格式
data_transform = transforms.Compose([transforms.ToTensor()])

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
# test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LeNet().to(device)

# 加载模型
model.load_state_dict(torch.load("D:/ws_pytorch/LeNet5/save_model/best_model.pth"))

# 获取预测结果
classes = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9"
]

# 进入到验证阶段
model.eval()
for i in range(80,90,1):
    x, y = test_dataset[i][0], test_dataset[i][1]
    show(x).show()
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True).to(device)
    # x = x.to(device)
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}", Actual:"{actual}"')

print('ending')
