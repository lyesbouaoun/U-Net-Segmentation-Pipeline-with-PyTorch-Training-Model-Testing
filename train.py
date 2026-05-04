import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from U_NET import UNET
from dataset_util import braindataset
import torch.nn as nn
import pandas as pd
from affichage import affich_loss

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

dataset = braindataset(
    image_dir='./data/images',
    mask_dir='./data/masks',
    transform=transform
)


data_train = DataLoader(dataset=dataset, batch_size=16, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNET().to(device)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
u=0
train_loss = []
print("START TRAINING")
for epoch in range(20):

    model.train()

    epoch_loss = 0   # 🔥 reset à chaque epoch
    print("EPOCH START")
    for images, masks in data_train:

        images = images.to(device)
        masks = masks.to(device).float()

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    epoch_loss /= len(data_train)   # 🔥 moyenne

    train_loss.append(epoch_loss)

    print(f"epoch : {epoch+1} | loss : {epoch_loss:.4f}")

# sauvegarde modèle
torch.save(model.state_dict(), "./model.pth")

# sauvegarde CSV
df = pd.DataFrame({
    "epoch": list(range(1, len(train_loss)+1)),
    "train_loss": train_loss
})
df.to_csv("train_loss.csv", index=False)

# affichage
affich_loss(train_loss)