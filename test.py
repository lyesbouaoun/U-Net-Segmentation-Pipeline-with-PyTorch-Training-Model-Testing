import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from affichage import show_result
from U_NET import UNET
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNET().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output)

    return img_tensor, pred[0].cpu()



def load_image():
    image_path = filedialog.askopenfilename()
    if image_path:
        image, pred = predict_image(image_path)
        show_result(image, pred)


def load_folder():
    folder_path = filedialog.askdirectory()
    for file in os.listdir(folder_path):
        if file.endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder_path, file)
            image, pred = predict_image(path)
            show_result(image, pred)

root = tk.Tk()
root.title("U-Net Tester")
root.geometry("400x300")
root.configure(bg="#1e1e2f")  # fond moderne

# Titre
title = tk.Label(
    root,
    text="U-Net Segmentation",
    font=("Helvetica", 18, "bold"),
    fg="white",
    bg="#1e1e2f"
)
title.pack(pady=20)

# Frame pour centrer les boutons
frame = tk.Frame(root, bg="#1e1e2f")
frame.pack(pady=20)

# Style bouton
btn_style = {
    "font": ("Helvetica", 12),
    "bg": "#4CAF50",
    "fg": "white",
    "activebackground": "#45a049",
    "width": 20,
    "height": 2,
    "bd": 0,
    "cursor": "hand2"
}

btn1 = tk.Button(
    frame,
    text="📷 Choisir une image",
    command=load_image,
    **btn_style
)
btn1.pack(pady=10)

btn2 = tk.Button(
    frame,
    text="📁 Choisir un dossier",
    command=load_folder,
    **btn_style
)
btn2.pack(pady=10)

# Footer
footer = tk.Label(
    root,
    text="Deep Learning - U-Net",
    font=("Helvetica", 9),
    fg="gray",
    bg="#1e1e2f"
)
footer.pack(side="bottom", pady=10)

root.mainloop()
