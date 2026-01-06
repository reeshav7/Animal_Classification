import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import joblib   

# --- Application Configuration ---
# We define different target sizes since the CNN handles high-res color images, 
# while the ANN was optimized for lower-res grayscale data.
CNN_IMG_SIZE = 224        
ANN_IMG_SIZE = 64         

# Supported animal categories based on the trained dataset
CLASS_NAMES = ['butterfly','cat','chicken','cow','dog',
               'elephant','horse','sheep','spider','squirrel']

# File paths for pre-trained models and preprocessing artifacts
MODEL_CNN = "models/cnn_animals_best.h5"
MODEL_ANN = "models/ann_model_multiclass.h5"
SCALER_ANN = "models/scaler_multiclass.pkl"
PCA_ANN    = "models/pca_multiclass.pkl"

# Initialize model placeholders for lazy loading to save startup time
cnn_model = None
ann_model = None
ann_scaler = None
ann_pca = None


# --- Model Management ---

def load_model_cnn():
    """
    Loads the CNN model into memory only when first requested.
    This keeps the GUI responsive during initial launch.
    """
    global cnn_model
    if cnn_model is None:
        status("Loading CNN Model...")
        cnn_model = tf.keras.models.load_model(MODEL_CNN)
        status("CNN Loaded ✔")
    return cnn_model


def load_model_ann():
    """
    Loads the ANN along with its specific Scaler and PCA components.
    The ANN requires precise preprocessing steps to match its training state.
    """
    global ann_model, ann_scaler, ann_pca
    if ann_model is None:
        status("Loading ANN + Preprocessing Files...")
        ann_model = tf.keras.models.load_model(MODEL_ANN)
        ann_scaler = joblib.load(SCALER_ANN)
        ann_pca = joblib.load(PCA_ANN)
        status("ANN Loaded ✔")
    return ann_model


# --- Inference Logic ---

def predict_image(filepath, use_model="CNN"):
    """
    Processes an image through the selected neural network architecture.
    Handles resizing, normalization, and dimensionality reduction where necessary.
    """
    img = Image.open(filepath).convert("RGB")

    if use_model == "CNN":
        # Standard CNN pipeline: Resize -> Normalize -> Expand Dimensions
        model = load_model_cnn()
        img_resized = img.resize((CNN_IMG_SIZE, CNN_IMG_SIZE))
        arr = np.array(img_resized) / 255.0
        arr = np.expand_dims(arr, 0)
        pred = model.predict(arr, verbose=0)[0]

    else:
        # ANN pipeline: Grayscale -> Flatten -> Scale -> PCA Transformation
        model = load_model_ann()
        img_resized = img.resize((ANN_IMG_SIZE, ANN_IMG_SIZE)).convert("L")
        arr = np.array(img_resized).flatten().reshape(1,-1)

        arr = ann_scaler.transform(arr)  
        arr = ann_pca.transform(arr)     # Reduce dimensionality to the top 600 features

        pred = model.predict(arr, verbose=0)[0]

    # Map the highest probability score to its corresponding class name
    idx = np.argmax(pred)
    conf = pred[idx] * 100
    return CLASS_NAMES[idx], conf


# --- GUI Event Handlers ---

def load_image():
    """
    Triggers a file dialog for the user to select an image and displays it on the canvas.
    """
    file = filedialog.askopenfilename(
        filetypes=[("Image Files","*.jpg *.jpeg *.png *.jfif *.webp *.bmp")]
    )
    if not file:
        return

    global img_path
    img_path = file

    # Generate a thumbnail for the UI while maintaining aspect ratio
    img = Image.open(file)
    img.thumbnail((350,350))
    tk_img = ImageTk.PhotoImage(img)

    canvas.img = tk_img
    canvas.create_image(175,175,image=tk_img)

    status(f"Loaded: {os.path.basename(file)}")


def run_prediction():
    """
    Executes the classification based on the currently loaded image and selected model.
    """
    if img_path == "":
        return messagebox.showwarning("No Image", "Please load an image first")

    model_type = model_var.get()
    label, conf = predict_image(img_path, model_type)

    result_lbl.config(text=f"Prediction: {label.upper()} ({conf:.2f}%)")
    status("Prediction Completed ✔")


def batch_test():
    """
    Iterates through all images in a selected directory and displays results in a new window.
    """
    folder = filedialog.askdirectory()
    if not folder:
        return

    model_type = model_var.get()
    results = []

    # Filter for common image extensions
    for img_name in os.listdir(folder):
        if img_name.lower().endswith((".jpg",".jpeg",".png",".jfif",".webp",".bmp")):
            path = os.path.join(folder,img_name)
            label, conf = predict_image(path, model_type)
            results.append(f"{img_name:<25} → {label.upper()} ({conf:.2f}%)")

    # Create a secondary window to display the batch report
    result_window = tk.Toplevel(root)
    result_window.title("Batch Results")
    text = tk.Text(result_window,width=60,height=30,font=("Calibri",11))
    text.pack()

    for r in results:
        text.insert(tk.END,r+"\n")

    status("Batch Testing Completed ✔")


def status(msg):
    """Updates the status bar at the bottom of the GUI."""
    status_lbl.config(text="Status: " + msg)


# --- UI Construction ---

root = tk.Tk()
root.title("Animal Classifier - ANN & CNN")
root.geometry("750x650")
root.resizable(False,False)
root.configure(bg="#23272f") # Modern dark theme background

img_path = ""

# Header Section
title = tk.Label(root,text="Animal Classification System",
                 font=("Arial",20,"bold"),bg="#23272f",fg="white")
title.pack(pady=10)

frame = tk.Frame(root,bg="#1e222a")
frame.pack(pady=10)

canvas = tk.Canvas(frame,width=350,height=350,bg="black")
canvas.pack()

# Model Choice
model_var = tk.StringVar(value="CNN")
tk.Radiobutton(root,text="Use CNN Model",variable=model_var,value="CNN",
               bg="#23272f",fg="white",selectcolor="black").pack()
tk.Radiobutton(root,text="Use ANN Model",variable=model_var,value="ANN",
               bg="#23272f",fg="white",selectcolor="black").pack()

# Buttons
btn_frame = tk.Frame(root,bg="#23272f")
btn_frame.pack(pady=15)

tk.Button(btn_frame,text="Load Image",font=("Arial",12),width=12,
          command=load_image).grid(row=0,column=0,padx=10)

tk.Button(btn_frame,text="Predict",font=("Arial",12),width=12,
          command=run_prediction).grid(row=0,column=1,padx=10)

tk.Button(btn_frame,text="Batch Test Folder",font=("Arial",12),width=16,
          command=batch_test).grid(row=0,column=2,padx=10)

result_lbl = tk.Label(root,text="Prediction Output...",
                      font=("Arial",14),bg="#23272f",fg="cyan")
result_lbl.pack(pady=10)

status_lbl = tk.Label(root,text="Status: Waiting...",
                      font=("Arial",11),bg="#23272f",fg="lightgrey")
status_lbl.pack(pady=5)

root.mainloop()
