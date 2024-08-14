import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root=".", transform=transform, train=True, download=True)
test_data = datasets.MNIST(root=".", transform=transform, train=False)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = CNN()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(20):
    running_loss = 0.0
    total_num, total_cor = 0, 0
    for i, data in enumerate(train_loader):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print("[{:d},{:4d}] loss :{:.3f}".format(epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

        correct = 0
        total = 0
        _, predicted = torch.max(outputs, 1)  # values, indices
        total = labels.size(0)
        total_num = total_num + total
        correct = (predicted == labels).sum().item()
        total_cor = total_cor + correct

    print("Accuracy: " + str(total_cor / total_num))

print("Finished Training")

torch.save(net.state_dict(), "mnist.pth")

print("Saved Model")
