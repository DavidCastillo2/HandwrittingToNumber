import torch
import matplotlib.pyplot as plt
import numpy as np


def view_classify(imgTemp, psTempt):
    """ Function for viewing an image and it's predicted classes."""
    psTempt = psTempt.cpu().data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(imgTemp.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), psTempt)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()


def main(model, valloader, device):
    images, labels = next(iter(valloader))

    img = images[0].view(1, 784)
    with torch.no_grad():
        if torch.cuda.device_count() != 0:
            logps = model(img.cuda())
        else:
            logps = model(img)

    ps = torch.exp(logps)
    probab = list(ps.cpu().numpy()[0])
    print("Predicted Digit =", probab.index(max(probab)), end='\n\n')
    view_classify(img.view(1, 28, 28), ps)

    correct_count, all_count = 0, 0
    for images, labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                if torch.cuda.device_count() != 0:
                    logps = model(img.cuda())
                else:
                    logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.cpu().numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if true_label == pred_label:
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("Model Accuracy =", (correct_count / all_count))