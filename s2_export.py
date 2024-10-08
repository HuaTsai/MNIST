import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: Simplify the model
# import onnx
# import onnxsim


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
net.load_state_dict(torch.load("mnist.pth", weights_only=True))
net.eval()

torch_input = torch.randn(1, 1, 28, 28)
onnx_program = torch.onnx.export(
    net,
    torch_input,
    "mnist.onnx",
    export_params=True,
    input_names=["input"],
    output_names=["output"],
)

print("Save the model to mnist.onnx")

# model = onnx.load("mnist.onnx")
# model_sim, check = onnxsim.simplify(model)
# onnx.save(model_sim, "mnist_sim.onnx")
#
# print("Save the simplified model to mnist_sim.onnx")
