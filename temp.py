import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from  torch.autograd import Variable

device = "cuda" if torch.cuda.is_available() else "cpu"
image = Image.open('data/mydata/0/0.jpg').convert('L')
resize = transforms.Resize([28, 28])
image = resize(image)
image = np.copy(image)
image = torch.tensor(image)
image = Variable(torch.unsqueeze(torch.unsqueeze(image, dim=0).byte(), dim=0).byte()).to(device)
image = image.clone().detach().to(device)

print(image)
print(image.size())

print("#" * 20)

ROOT_TEST = r'D:/ws_pytorch/LeNet5/data/mydata'

normalize = transforms.Normalize([0.1307], [0.3081])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()])

test_dataset = ImageFolder(ROOT_TEST, transform=test_transform)
x = test_dataset[0][0]
x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True).to(device)
x = x.clone().detach().to(device)

print(x)
print(x.size())