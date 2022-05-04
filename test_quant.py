import torch
from torch import nn
from net_quant import LeNet
from torchvision import datasets, transforms
import time

normalize = transforms.Normalize([0.1307], [0.3081])
# 数据转化为tensor格式
data_transform = transforms.Compose([transforms.ToTensor(),
                                     normalize])

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data', train=False, transform=data_transform, download=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1000, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 调用net定义的模型
model = LeNet().to(device)

model.load_state_dict(torch.load("D:/ws_pytorch/LeNet5/save_model/best_model.pth"))

# 定义损失函数（交叉熵）
loss_fn = nn.CrossEntropyLoss()

def test(dataloader, model, loss_fn):
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # 前向传播
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
            test_loss = loss / n
            test_acc = current / n

            return test_loss, test_acc

# 量化前测试
start1 = time.time()
for i in range(20):
    test_loss, test_acc = test(test_dataloader, model, loss_fn)
end1 = time.time()
print("量化前测试")
print("test_loss" + str(test_loss))
print("test_acc" + str(test_acc))
print("耗时" + str(end1 - start1))
print("#" * 20)

# 量化后测试
model.linear_quant()
start2 = time.time()
for i in range(20):
    test_loss, test_acc = test(test_dataloader, model, loss_fn)
end2 = time.time()
print("量化后测试")
print("test_loss" + str(test_loss))
print("test_acc" + str(test_acc))
print("耗时" + str(end2 - start2))
print("#" * 20)


# # 模型保存
# folder = 'weight/quantization/'
# for name in model.state_dict():
#     # print("################" + name + "################")
#     # print(model.state_dict()[name])
#     file = open(folder + name + ".txt", "w")
#     file.write(str(model.state_dict()[name]))
#     file.close()
#
# file = open(folder + "c1_scale_zero.txt", "w")
# file.write(str(model.c1.scale))
# file.write("\n" + str(model.c1.zero_point))
# file.close()
#
# file = open(folder + "c3_scale_zero.txt", "w")
# file.write(str(model.c3.scale))
# file.write("\n" + str(model.c3.zero_point))
# file.close()
#
# file = open(folder + "c5_scale_zero.txt", "w")
# file.write(str(model.c5.scale))
# file.write("\n" + str(model.c5.zero_point))
# file.close()
#
# file = open(folder + "f6_scale_zero.txt", "w")
# file.write(str(model.f6.scale))
# file.write("\n" + str(model.f6.zero_point))
# file.close()
#
# file = open(folder + "output_scale_zero.txt", "w")
# file.write(str(model.output.scale))
# file.write("\n" + str(model.output.zero_point))
# file.close()
