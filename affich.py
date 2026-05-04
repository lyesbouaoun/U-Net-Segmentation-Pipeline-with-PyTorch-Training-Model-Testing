import pandas as pd
import matplotlib.pyplot as plt

# lire le fichier CSV
df = pd.read_csv("train_loss.csv")


# plot
plt.plot(df["epoch"], df["train_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()

plt.show()