import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch import nn, optim
import ObserveModel

# torch.manual_seed(1)
# np.random.seed(1)

# Normalize our data around [-1,1] , the owners of the data have already calculated our mean and std and manually set them to 0.5 for both
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download our data - skips if already downloaded automatically
trainset = datasets.MNIST('Data/HardTruths', download=True, train=True, transform=transform)
valset = datasets.MNIST('Data/TestData', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

# Load our data into torch objects
dataiter = iter(trainloader)
images, labels = next(dataiter)

# See what dimensions our data has
"""
print("Number of Images: %d\nNumber of Channels: %d\n\t\t(in this case referring to # of "
      "colors represented)\nImage Height: %d\nImage Width: %d" % images.shape)
print("Number of Labels: %d" % labels.shape)
"""

# See what our computer is being handed
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
# plt.show()


# Make Our Network
input_size = 784  # (our images are 28x28 = 784)
hidden_sizes = [128, 64]
output_size = 10  # (maps to the 10 digits that exist directly)

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))


# print(model)


# Use GPU if we can
forceCPU = True
def isGPU():
    return torch.cuda.device_count() != 0 and not forceCPU


device = torch.device("cuda" if torch.cuda.is_available() and not forceCPU else "cpu")
print("Device: %s" % device)
model.to(device)

# Create our loss functions
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

if isGPU():
    logps = model(images.cuda())  # log probabilities
    loss = criterion(logps, labels.cuda())  # calculate the NLL loss
else:
    logps = model(images)  # log probabilities
    loss = criterion(logps, labels)  # calculate the NLL loss

# This shows our network being empty at the start
# print('Before backward pass: \n', model[0].weight.grad)
# loss.backward()
# print('After backward pass: \n', model[0].weight.grad)


# Create our optimizer
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
optimizer.zero_grad()  # ? says we need, citation needed

# Just make it easy to swap between GPU and CPU
if isGPU():
    def cudaWrap(o):
        return o.cuda()
else:
    def cudaWrap(o):
        return o

# This is where the real magic happens
time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()
        output = model(cudaWrap(images))
        loss = criterion(output, cudaWrap(labels))

        # This is where the model learns by backpropagation
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
print("\nTraining Time (in minutes) =", (time() - time0) / 60)

# Check how it went and see a real example
ObserveModel.main(model, valloader, forceCPU)
