import os
import threading
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import joblib

# ================= CONFIG =================
CNN_IMG_SIZE = 224
ANN_IMG_SIZE = 64

CLASS_NAMES = [
    'butterfly','cat','chicken','cow','dog',
    'elephant','horse','sheep','spider','squirrel'
]

MODEL_CNN  = "models/cnn_animals_best.h5"
MODEL_ANN  = "models/ann_model_multiclass.h5"
SCALER_ANN = "models/scaler_multiclass.pkl"
PCA_ANN    = "models/pca_multiclass.pkl"

# ================= GLOBAL STATE =================
cnn_model = None
ann_model = None
ann_scaler = None
ann_pca = None
img_path = None

# ================= MODEL LOADING =================
def load_cnn():
    global cnn_model
    if cnn_model is None:
        update_status("Loading CNN model...")
        cnn_model = tf.keras.models.load_model(MODEL_CNN)
        update_status("CNN loaded ✔")
    return cnn_model

def load_ann():
    global ann_model, ann_scaler, ann_pca
    if ann_model is None:
        update_status("Loading ANN model...")
        ann_model = tf.keras.models.load_model(MODEL_ANN)
        ann_scaler = joblib.load(SCALER_ANN)
        ann_pca = joblib.load(PCA_ANN)
        update_status("ANN loaded ✔")
    return ann_model

# ================= PREDICTION =================
def predict(filepath, model_type):
    img = Image.open(filepath).convert("RGB")

    if model_type == "CNN":
        model = load_cnn()
        img = img.resize((CNN_IMG_SIZE, CNN_IMG_SIZE))
        arr = np.array(img) / 255.0
        arr = np.expand_dims(arr, 0)
        pred = model.predict(arr, verbose=0)[0]
    else:
        model = load_ann()
        img = img.resize((ANN_IMG_SIZE, ANN_IMG_SIZE)).convert("L")
        arr = np.array(img).flatten().reshape(1, -1)
        arr = ann_scaler.transform(arr)
        arr = ann_pca.transform(arr)
        pred = model.predict(arr, verbose=0)[0]

    top3 = np.argsort(pred)[-3:][::-1]
    return [(CLASS_NAMES[i], pred[i] * 100) for i in top3]

# ================= THREAD HANDLING =================
def run_prediction_async():
    threading.Thread(target=run_prediction, daemon=True).start()

def run_prediction():
    if not img_path:
        messagebox.showwarning("No Image", "Please load an image first.")
        return

    disable_buttons()
    update_status("Running inference...")

    model_type = model_choice.get()
    results = predict(img_path, model_type)

    display_results(results)
    update_status("Prediction completed ✔")
    enable_buttons()

# ================= UI HELPERS =================
def update_status(msg):
    status_lbl.config(text=f"Status: {msg}")
    root.update_idletasks()

def disable_buttons():
    predict_btn.config(state=tk.DISABLED)
    batch_btn.config(state=tk.DISABLED)

def enable_buttons():
    predict_btn.config(state=tk.NORMAL)
    batch_btn.config(state=tk.NORMAL)

def display_results(results):
    conf_bar.delete("all")
    text = ""

    for i, (label, conf) in enumerate(results):
        text += f"{label.upper():10s} : {conf:5.2f}%\n"
        if i == 0:
            conf_bar.create_rectangle(0, 0, 3*conf, 20, fill="lime")

    result_lbl.config(text=text)

# ================= IMAGE LOAD =================
def load_image():
    global img_path
    file = filedialog.askopenfilename(
        filetypes=[("Images","*.jpg *.jpeg *.png *.jfif *.bmp *.webp")]
    )
    if not file:
        return

    img_path = file
    img = Image.open(file)
    img.thumbnail((350,350))
    tk_img = ImageTk.PhotoImage(img)

    canvas.delete("all")
    canvas.create_image(175,175,image=tk_img)
    canvas.image = tk_img

    enable_buttons()
    update_status(f"Loaded {os.path.basename(file)}")

# ================= BATCH MODE =================
def batch_test():
    folder = filedialog.askdirectory()
    if not folder:
        return

    disable_buttons()
    update_status("Running batch inference...")

    model_type = model_choice.get()
    results = []

    for f in os.listdir(folder):
        if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp",".jfif")):
            path = os.path.join(folder, f)
            label, conf = predict(path, model_type)[0]
            results.append((f, label, conf))

    show_batch_results(results)
    update_status("Batch testing completed ✔")
    enable_buttons()

def show_batch_results(results):
    win = tk.Toplevel(root)
    win.title("Batch Results")

    txt = tk.Text(win, width=70, height=30, font=("Segoe UI", 11))
    txt.pack(padx=10, pady=10)

    for f, lbl, conf in results:
        txt.insert(tk.END, f"{f:25s} → {lbl.upper():10s} ({conf:5.2f}%)\n")

# ================= GUI =================
root = tk.Tk()
root.title("Animal Classification System")
root.geometry("780x720")
root.configure(bg="#1e1e1e")
root.resizable(False, False)

title = tk.Label(
    root, text="Animal Classifier (ANN & CNN)",
    font=("Segoe UI", 20, "bold"),
    fg="white", bg="#1e1e1e"
)
title.pack(pady=10)

canvas = tk.Canvas(root, width=350, height=350, bg="black")
canvas.pack(pady=10)

# ======= FIXED RADIO BUTTONS =======
model_choice = tk.StringVar(value="CNN")

tk.Radiobutton(
    root, text="CNN (Recommended)",
    variable=model_choice, value="CNN",
    bg="#1e1e1e", fg="white",
    activebackground="#1e1e1e",
    activeforeground="white",
    selectcolor="#3a3a3a",   # <<< CRITICAL FIX
    font=("Segoe UI", 11)
).pack()

tk.Radiobutton(
    root, text="ANN (Baseline)",
    variable=model_choice, value="ANN",
    bg="#1e1e1e", fg="white",
    activebackground="#1e1e1e",
    activeforeground="white",
    selectcolor="#3a3a3a",   # <<< CRITICAL FIX
    font=("Segoe UI", 11)
).pack()

btn_frame = tk.Frame(root, bg="#1e1e1e")
btn_frame.pack(pady=15)

load_btn = ttk.Button(btn_frame, text="Load Image", command=load_image)
predict_btn = ttk.Button(btn_frame, text="Predict", command=run_prediction_async)
batch_btn = ttk.Button(btn_frame, text="Batch Folder", command=batch_test)

load_btn.grid(row=0, column=0, padx=10)
predict_btn.grid(row=0, column=1, padx=10)
batch_btn.grid(row=0, column=2, padx=10)

predict_btn.config(state=tk.DISABLED)
batch_btn.config(state=tk.DISABLED)

result_lbl = tk.Label(
    root, text="Prediction Output",
    font=("Consolas", 14),
    fg="cyan", bg="#1e1e1e",
    justify="left"
)
result_lbl.pack(pady=10)

conf_bar = tk.Canvas(root, width=300, height=20, bg="gray")
conf_bar.pack(pady=5)

status_lbl = tk.Label(
    root, text="Status: Waiting...",
    font=("Segoe UI", 11),
    fg="lightgray", bg="#1e1e1e"
)
status_lbl.pack(pady=10)

root.mainloop()
