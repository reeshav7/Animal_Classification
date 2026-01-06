import os, shutil, random

data_dir = "../data/animals"   # folder that contains 10 animal folders
train_dir = "../data/train"
test_dir = "../data/test"

split_ratio = 0.8  # 80% training, 20% testing

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Auto-detect class folders
classes = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))]
print("\nDetected Classes:", classes)

for cls in classes:
    src = os.path.join(data_dir, cls)
    imgs = os.listdir(src)
    random.shuffle(imgs)

    split = int(len(imgs)*split_ratio)
    train_imgs = imgs[:split]
    test_imgs = imgs[split:]

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(train_dir, cls, img))

    for img in test_imgs:
        shutil.copy(os.path.join(src, img), os.path.join(test_dir, cls, img))

print("\n=== Dataset Split Completed Successfully ===")
