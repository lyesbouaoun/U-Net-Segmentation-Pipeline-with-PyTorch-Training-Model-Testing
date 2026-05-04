import matplotlib.pyplot as plt
import torch

def show_result(image, pred):
    plt.figure(figsize=(10,6))

    img = image.squeeze().permute(1,2,0).cpu()

    plt.subplot(1,3,1)
    plt.title("Original Image")
    plt.imshow(img)

    plt.subplot(1,3,2)
    plt.title("Predicted Mask")
    plt.imshow(pred[0], cmap="gray")

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(img)
    plt.imshow(pred[0], cmap="jet", alpha=0.5)

    plt.show()

def affich_loss(train_loss):
    plt.figure()
    plt.plot(train_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("train_loss")
    plt.show()

def overlay(img, pred):
    pred = torch.sigmoid(pred)[0].detach()

    plt.imshow(img.permute(1,2,0))
    plt.imshow(pred, alpha=0.5, cmap='jet')
    plt.show()


