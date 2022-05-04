import torch
from net import LeNet
from torchvision.datasets import ImageFolder
from torchvision import transforms
from  torch.autograd import Variable
from torchvision.transforms import ToPILImage

show = ToPILImage()

ROOT_TEST = r'D:/ws_pytorch/LeNet5/data/mydata'

normalize = transforms.Normalize([0.1307], [0.3081])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    normalize])

test_dataset = ImageFolder(ROOT_TEST, transform=test_transform)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = LeNet().to(device)

model.load_state_dict(torch.load("D:/ws_pytorch/LeNet5/save_model/best_model.pth"))

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

model.eval()

num = len(test_dataset)

for i in range(num):
    x, y = test_dataset[i][0], test_dataset[i][1]
    show(x).show()
    x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=True).to(device)
    x = x.clone().detach().to(device)
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]
        print(f'predicted:"{predicted}", Actual:"{actual}"')
