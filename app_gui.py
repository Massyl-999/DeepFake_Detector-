# app_gui.py - Lance avec python app_gui.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models

# Chargement du modèle
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(path):
    img = Image.open(path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(input_tensor)).item()
    return "DEEPFAKE" if prob > 0.5 else "IMAGE RÉELLE", prob

# GUI
root = tk.Tk()
root.title("Détecteur Deepfake - 100% Images")
root.geometry("800x700")
root.configure(bg="#2c3e50")

tk.Label(root, text="Détecteur de Deepfake", font=("Arial", 24, "bold"), fg="#ecf0f1", bg="#2c3e50").pack(pady=20)

img_label = tk.Label(root, bg="white")
img_label.pack(pady=20)

result_label = tk.Label(root, text="Charge une image pour analyser", font=("Arial", 18), fg="#ecf0f1", bg="#2c3e50")
result_label.pack(pady=10)

def open_image():
    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
    if not path:
        return
    # Afficher l'image
    img = Image.open(path).resize((400, 400))
    photo = ImageTk.PhotoImage(img)
    img_label.config(image=photo)
    img_label.image = photo

    # Prédiction
    result_label.config(text="Analyse en cours...", fg="yellow")
    root.update()
    label, prob = predict_image(path)
    confidence = prob * 100 if "DEEPFAKE" in label else (1 - prob) * 100
    color = "#e74c3c" if "DEEPFAKE" in label else "#2ecc71"
    result_label.config(
        text=f"{label}\nConfiance : {confidence:.1f}%",
        fg=color, font=("Arial", 22, "bold")
    )

tk.Button(root, text="Choisir une image", command=open_image,
          font=("Arial", 16), bg="#3498db", fg="white", height=2, width=20).pack(pady=20)

tk.Label(root, text="Fonctionne 100% hors ligne - Seulement des images", fg="#95a5a6", bg="#2c3e50").pack(side="bottom", pady=20)

root.mainloop()