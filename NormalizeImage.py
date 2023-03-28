import math

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

img = Image.open('Data/me.jpg')
# convert image to torch tensor
imgTensor = T.ToTensor()(img)


# Find our mean and Standard Deviation of MY dataset

# This lets us scale 0,255 to 0,1
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0,), (1,)),
                                ])


examples = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225),
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
            (1, 1, 1), (1, 1, 1),
            (0.1,), (0.1,)]

for i in range(0, len(examples), 2):
    transform = T.Normalize(mean=examples[i], std=examples[i + 1])

    normalized_imgTensor = transform(imgTensor)

    normalized_img = T.ToPILImage()(normalized_imgTensor)

    normalized_img.show()
