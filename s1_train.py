import torch
import torch.nn.functional as F  # noqa
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
        super().__init__()
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
        # Training steps:
        # 1. Move data to GPU
        # 2. Zero gradient
        # 3. Forward pass
        # 4. Compute loss
        # 5. Backward pass
        # 6. Update weights
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(f"[{epoch + 1:d},{i + 1:4d}] loss :{running_loss / 100:.3f}")
            running_loss = 0.0

        _, predicted = torch.max(outputs, 1)  # values, indices
        total_num = total_num + labels.size(0)  # batch size
        total_cor = total_cor + (predicted == labels).sum().item()

    print("Accuracy: " + str(total_cor / total_num))

print("Finished Training")

torch.save(net.state_dict(), "mnist.pth")

print("Saved Model")
