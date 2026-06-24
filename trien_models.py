# """
# Passive FM Radar - Drone Identification
# ========================================
# Train a CNN to classify 37 drone types from RF spectrogram images.
# Beginner-friendly, CPU-optimized.
# """

# import os
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms, models
# import matplotlib.pyplot as plt
# import json

# # ─────────────────────────────────────────
# # CONFIGURATION — Edit these paths/values
# # ─────────────────────────────────────────
# DATA_DIR   = "D:/gp data/RFUAV_valid/"          # Folder containing 37 drone subfolders
# MODEL_DIR  = "saved_models/"  # Where trained model is saved
# RESULTS_DIR= "results/"       # Where plots and metrics are saved

# IMG_SIZE   = 224              # ResNet expects 224x224
# BATCH_SIZE = 16               # Small batch for CPU
# EPOCHS     = 20               # Increase for better accuracy
# LR         = 0.001            # Learning rate
# VAL_SPLIT  = 0.15             # 15% validation
# TEST_SPLIT = 0.15             # 15% test
# NUM_CLASSES= 37               # Number of drone types

# DEVICE = torch.device("cpu")
# print(f"Using device: {DEVICE}")

# # ─────────────────────────────────────────
# # STEP 1: DATA LOADING & AUGMENTATION
# # ─────────────────────────────────────────
# def get_data_loaders(data_dir):
#     """Load images from 37 drone folders and split into train/val/test."""

#     # Training augmentation (makes model more robust)
#     train_transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
#                              [0.229, 0.224, 0.225])    # ImageNet std
#     ])

#     # Validation/test — no augmentation, just resize & normalize
#     eval_transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ])

#     # Load full dataset (uses folder names as class labels automatically)
#     full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    
#     # Print class info
#     print(f"\n✅ Found {len(full_dataset)} images across {len(full_dataset.classes)} drone classes")
#     print(f"   Classes: {full_dataset.classes[:5]}... (showing first 5)\n")
    
#     # Save class names for later prediction use
#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     with open(os.path.join(RESULTS_DIR, "class_names.json"), "w") as f:
#         json.dump(full_dataset.classes, f)

#     # Split dataset
#     total = len(full_dataset)
#     val_size  = int(total * VAL_SPLIT)
#     test_size = int(total * TEST_SPLIT)
#     train_size = total - val_size - test_size

#     train_ds, val_ds, test_ds = random_split(
#         full_dataset, [train_size, val_size, test_size],
#         generator=torch.Generator().manual_seed(42)
#     )

#     # Apply eval transform to val and test
#     val_ds.dataset.transform  = eval_transform
#     test_ds.dataset.transform = eval_transform

#     print(f"📊 Dataset split:")
#     print(f"   Train: {train_size} images")
#     print(f"   Val:   {val_size} images")
#     print(f"   Test:  {test_size} images\n")

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
#     val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#     test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#     return train_loader, val_loader, test_loader, full_dataset.classes


# # ─────────────────────────────────────────
# # STEP 2: MODEL — ResNet18 (Transfer Learning)
# # ─────────────────────────────────────────
# def build_model(num_classes):
#     """
#     Use ResNet18 pretrained on ImageNet, replace final layer for 37 classes.
#     Transfer learning = faster training, better accuracy, less data needed.
#     """
#     model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

#     # Freeze early layers (they already learned useful features)
#     for param in list(model.parameters())[:-10]:
#         param.requires_grad = False

#     # Replace final layer: 512 → 37 classes
#     model.fc = nn.Sequential(
#         nn.Dropout(0.4),
#         nn.Linear(model.fc.in_features, num_classes)
#     )

#     return model.to(DEVICE)


# # ─────────────────────────────────────────
# # STEP 3: TRAINING LOOP
# # ─────────────────────────────────────────
# def train_one_epoch(model, loader, optimizer, criterion):
#     model.train()
#     total_loss, correct, total = 0, 0, 0

#     for images, labels in loader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         _, predicted = outputs.max(1)
#         correct += predicted.eq(labels).sum().item()
#         total += labels.size(0)

#     return total_loss / len(loader), 100.0 * correct / total


# def evaluate(model, loader, criterion):
#     model.eval()
#     total_loss, correct, total = 0, 0, 0

#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             correct += predicted.eq(labels).sum().item()
#             total += labels.size(0)

#     return total_loss / len(loader), 100.0 * correct / total


# # ─────────────────────────────────────────
# # STEP 4: PLOT TRAINING CURVES
# # ─────────────────────────────────────────
# def plot_history(history):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#     epochs = range(1, len(history["train_acc"]) + 1)

#     ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#e74c3c")
#     ax1.plot(epochs, history["val_loss"],   label="Val Loss",   color="#3498db")
#     ax1.set_title("Loss over Epochs")
#     ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
#     ax1.legend(); ax1.grid(True, alpha=0.3)

#     ax2.plot(epochs, history["train_acc"], label="Train Acc", color="#e74c3c")
#     ax2.plot(epochs, history["val_acc"],   label="Val Acc",   color="#3498db")
#     ax2.set_title("Accuracy over Epochs")
#     ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
#     ax2.legend(); ax2.grid(True, alpha=0.3)

#     plt.tight_layout()
#     path = os.path.join(RESULTS_DIR, "training_curves.png")
#     plt.savefig(path, dpi=150)
#     print(f"\n📈 Training curves saved to: {path}")
#     plt.show()


# # ─────────────────────────────────────────
# # MAIN
# # ─────────────────────────────────────────
# def main():
#     os.makedirs(MODEL_DIR, exist_ok=True)
#     os.makedirs(RESULTS_DIR, exist_ok=True)

#     # Load data
#     train_loader, val_loader, test_loader, class_names = get_data_loaders(DATA_DIR)

#     # Build model
#     model = build_model(NUM_CLASSES)
#     print(f"🧠 Model: ResNet18 with Transfer Learning")
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"   Trainable parameters: {total_params:,}\n")

#     # Loss & optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

#     # Training loop
#     best_val_acc = 0.0
#     history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

#     print("=" * 55)
#     print("🚀 Starting Training")
#     print("=" * 55)

#     for epoch in range(1, EPOCHS + 1):
#         t0 = time.time()

#         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
#         val_loss,   val_acc   = evaluate(model, val_loader, criterion)
#         scheduler.step()

#         history["train_loss"].append(train_loss)
#         history["val_loss"].append(val_loss)
#         history["train_acc"].append(train_acc)
#         history["val_acc"].append(val_acc)

#         elapsed = time.time() - t0
#         print(f"Epoch [{epoch:02d}/{EPOCHS}] "
#               f"| Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% "
#               f"| Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% "
#               f"| Time: {elapsed:.0f}s")

#         # Save best model
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             path = os.path.join(MODEL_DIR, "best_model.pth")
#             torch.save(model.state_dict(), path)
#             print(f"   💾 Best model saved! (Val Acc: {best_val_acc:.1f}%)")

#     # Final test evaluation
#     print("\n" + "=" * 55)
#     print("📋 Final Test Evaluation")
#     print("=" * 55)
#     model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))
#     test_loss, test_acc = evaluate(model, test_loader, criterion)
#     print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

#     # Save history & plot
#     with open(os.path.join(RESULTS_DIR, "history.json"), "w") as f:
#         json.dump(history, f)

#     plot_history(history)
#     print("\n✅ Training complete!")


# if __name__ == "__main__":
#     main()



#################################
######### overfiting ############
#################################

# """
# Passive FM Radar - Drone Identification
# ========================================
# Train a CNN to classify 37 drone types from RF spectrogram images.
# Supports inference on new images without completing all epochs.
# CPU-friendly.
# """

# import os, time, json
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms, models
# from PIL import Image

# # ────────────────────── CONFIG ──────────────────────
# DATA_DIR = "D:/gp data/RFUAV_valid/"         # 37 drone folders
# MODEL_DIR   = "saved_models/"
# RESULTS_DIR = "results/"
# IMG_SIZE    = 224
# BATCH_SIZE  = 16
# EPOCHS      = 20
# LR          = 0.001
# VAL_SPLIT   = 0.15
# TEST_SPLIT  = 0.15
# NUM_CLASSES = 37
# DEVICE      = torch.device("cpu")

# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)
# print(f"Using device: {DEVICE}")

# # ────────────────────── DATA LOADERS ──────────────────────
# def get_data_loaders(data_dir):
#     train_transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#     ])
#     eval_transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#     ])

#     dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
#     print(f"✅ Found {len(dataset)} images across {len(dataset.classes)} classes")
#     with open(os.path.join(RESULTS_DIR, "class_names.json"), "w") as f:
#         json.dump(dataset.classes, f)

#     total = len(dataset)
#     val_size  = int(total*VAL_SPLIT)
#     test_size = int(total*TEST_SPLIT)
#     train_size = total - val_size - test_size

#     train_ds, val_ds, test_ds = random_split(
#         dataset, [train_size, val_size, test_size],
#         generator=torch.Generator().manual_seed(42)
#     )
#     val_ds.dataset.transform = eval_transform
#     test_ds.dataset.transform = eval_transform

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#     val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#     test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

#     return train_loader, val_loader, test_loader, dataset.classes

# # ────────────────────── MODEL ──────────────────────
# def build_model(num_classes):
#     model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#     for param in list(model.parameters())[:-10]:
#         param.requires_grad = False
#     model.fc = nn.Sequential(nn.Dropout(0.4),
#                              nn.Linear(model.fc.in_features, num_classes))
#     return model.to(DEVICE)

# # ────────────────────── TRAINING ──────────────────────
# def train_one_epoch(model, loader, optimizer, criterion):
#     model.train()
#     total_loss, correct, total = 0,0,0
#     for images, labels in loader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         _, pred = outputs.max(1)
#         correct += pred.eq(labels).sum().item()
#         total += labels.size(0)
#     return total_loss/len(loader), 100*correct/total

# def evaluate(model, loader, criterion):
#     model.eval()
#     total_loss, correct, total = 0,0,0
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             _, pred = outputs.max(1)
#             correct += pred.eq(labels).sum().item()
#             total += labels.size(0)
#     return total_loss/len(loader), 100*correct/total

# # ────────────────────── INFERENCE ──────────────────────
# def predict_image(model, img_path, class_names):
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
#     ])
#     img = Image.open(img_path).convert("RGB")
#     img_tensor = transform(img).unsqueeze(0)
#     model.eval()
#     with torch.no_grad():
#         output = model(img_tensor.to(DEVICE))
#         pred_class = output.argmax(dim=1).item()
#     return class_names[pred_class]

# def predict_folder(model, folder_path, class_names):
#     files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".png",".jpeg"))]
#     for f in files:
#         path = os.path.join(folder_path, f)
#         pred = predict_image(model, path, class_names)
#         print(f"{f} → {pred}")

# # ────────────────────── MAIN ──────────────────────
# def main():
#     # إذا عايز تكمل تدريب
#     train_loader, val_loader, test_loader, class_names = get_data_loaders(DATA_DIR)
#     model = build_model(NUM_CLASSES)
#     print(f"🧠 Model built with trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

#     best_val_acc = 0.0
#     for epoch in range(1, EPOCHS+1):
#         t0 = time.time()
#         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
#         val_loss, val_acc = evaluate(model, val_loader, criterion)
#         scheduler.step()
#         print(f"Epoch [{epoch}/{EPOCHS}] | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Time: {int(time.time()-t0)}s")
#         # حفظ أفضل نموذج
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
#             print(f"💾 Best model saved! Val Acc: {best_val_acc:.1f}%")

#     # تحميل أفضل نموذج لأي inference
#     model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))

#     # --- تجربة صورة واحدة ---
#     test_image = "data/new_drone_image.jpg"  # غيره للصورة عندك
#     print("Prediction:", predict_image(model, test_image, class_names))

#     # --- تجربة فولدر كامل ---
#     test_folder = "data/test_images"  # فولدر الصور عندك
#     predict_folder(model, test_folder, class_names)

# if __name__ == "__main__":
#     main()






# import os, time, json, random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# from torchvision import datasets, transforms
# from PIL import Image

# # ────────────────────── CONFIG ──────────────────────
# DATA_DIR = "D:/gp data/RFUAV_valid/"
# MODEL_DIR = "saved_models/"
# RESULTS_DIR = "results/"

# IMG_SIZE = 224
# BATCH_SIZE = 16
# EPOCHS = 15
# LR = 5e-4
# VAL_SPLIT = 0.15
# TEST_SPLIT = 0.15
# NUM_CLASSES = 37

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)

# print(f"🚀 Using device: {DEVICE}")

# # ────────────────────── SPEC AUGMENT ──────────────────────
# class SpecAugment:
#     def __call__(self, img):
#         img = img.clone()
#         # Time masking
#         t = random.randint(0, img.shape[1]//4)
#         t0 = random.randint(0, img.shape[1]-t)
#         img[:, t0:t0+t] = 0

#         # Frequency masking
#         f = random.randint(0, img.shape[2]//4)
#         f0 = random.randint(0, img.shape[2]-f)
#         img[:, :, f0:f0+f] = 0

#         return img

# # ────────────────────── DATA ──────────────────────
# def get_data_loaders(data_dir):

#     base_dataset = datasets.ImageFolder(root=data_dir)

#     print(f"✅ Found {len(base_dataset)} images across {len(base_dataset.classes)} classes")

#     with open(os.path.join(RESULTS_DIR, "class_names.json"), "w") as f:
#         json.dump(base_dataset.classes, f)

#     total = len(base_dataset)
#     val_size = int(total * VAL_SPLIT)
#     test_size = int(total * TEST_SPLIT)
#     train_size = total - val_size - test_size

#     indices = torch.randperm(total)
#     train_idx = indices[:train_size]
#     val_idx   = indices[train_size:train_size+val_size]
#     test_idx  = indices[train_size+val_size:]

#     # Transforms
#     train_transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomVerticalFlip(),
#         transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
#         transforms.GaussianBlur(3),
#         transforms.ToTensor(),
#         SpecAugment(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])

#     eval_transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])

#     train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_transform), train_idx)
#     val_ds   = Subset(datasets.ImageFolder(data_dir, transform=eval_transform), val_idx)
#     test_ds  = Subset(datasets.ImageFolder(data_dir, transform=eval_transform), test_idx)

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
#     test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

#     return train_loader, val_loader, test_loader, base_dataset.classes

# # ────────────────────── MODEL ──────────────────────
# class RFNet(nn.Module):
#     def __init__(self, num_classes=37):
#         super(RFNet, self).__init__()

#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.bn1   = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2   = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn3   = nn.BatchNorm2d(128)

#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#         self.bn4   = nn.BatchNorm2d(256)

#         self.gap = nn.AdaptiveAvgPool2d((1,1))

#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = torch.max_pool2d(x, 2)

#         x = torch.relu(self.bn2(self.conv2(x)))
#         x = torch.max_pool2d(x, 2)

#         x = torch.relu(self.bn3(self.conv3(x)))
#         x = torch.max_pool2d(x, 2)

#         x = torch.relu(self.bn4(self.conv4(x)))

#         x = self.gap(x)
#         x = x.view(x.size(0), -1)

#         return self.fc(x)

# # ────────────────────── TRAIN ──────────────────────
# def train_one_epoch(model, loader, optimizer, criterion):
#     model.train()
#     total_loss, correct, total = 0, 0, 0

#     for images, labels in loader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         _, pred = outputs.max(1)
#         correct += pred.eq(labels).sum().item()
#         total += labels.size(0)

#     return total_loss/len(loader), 100*correct/total

# def evaluate(model, loader, criterion):
#     model.eval()
#     total_loss, correct, total = 0, 0, 0

#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             total_loss += loss.item()
#             _, pred = outputs.max(1)
#             correct += pred.eq(labels).sum().item()
#             total += labels.size(0)

#     return total_loss/len(loader), 100*correct/total

# # ────────────────────── INFERENCE ──────────────────────
# def predict_image(model, img_path, class_names):
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5]*3, [0.5]*3)
#     ])

#     img = Image.open(img_path).convert("RGB")
#     img = transform(img).unsqueeze(0).to(DEVICE)

#     model.eval()
#     with torch.no_grad():
#         output = model(img)
#         probs = torch.softmax(output, dim=1)
#         top5 = torch.topk(probs, 5)

#     results = []
#     for i in range(5):
#         cls = class_names[top5.indices[0][i]]
#         conf = top5.values[0][i].item()*100
#         results.append((cls, conf))

#     return results

# # ────────────────────── MAIN ──────────────────────
# def main():

#     train_loader, val_loader, test_loader, class_names = get_data_loaders(DATA_DIR)

#     model = RFNet(NUM_CLASSES).to(DEVICE)

#     print(f"🧠 Params: {sum(p.numel() for p in model.parameters())}")

#     criterion = nn.CrossEntropyLoss()

#     optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

#     best_val = 0
#     patience = 5
#     counter = 0

#     for epoch in range(1, EPOCHS+1):

#         t0 = time.time()

#         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
#         val_loss, val_acc = evaluate(model, val_loader, criterion)

#         print(f"Epoch {epoch}: Train {train_acc:.1f}% | Val {val_acc:.1f}% | {int(time.time()-t0)}s")

#         if val_acc > best_val:
#             best_val = val_acc
#             counter = 0
#             torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pth"))
#             print("💾 Saved best model")
#         else:
#             counter += 1

#         if counter >= patience:
#             print("⛔ Early stopping")
#             break

#     # Test
#     model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth")))
#     test_loss, test_acc = evaluate(model, test_loader, criterion)
#     print(f"\n🧪 Test Accuracy: {test_acc:.2f}%")

#     # Predict example
#     test_image = "data/test.jpg"
#     print("\n🔍 Prediction:")
#     print(predict_image(model, test_image, class_names))


# if __name__ == "__main__":
#     main()




# ##########################################
# ########## trine RFnet v1 with acc 80%####
# ##########################################
# import os, time, json, random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Subset
# from torchvision import datasets, transforms
# from PIL import Image
# import numpy as np

# # ────────────────────── CONFIG ──────────────────────
# DATA_DIR = "D:/gp data/RFUAV_valid/"
# MODEL_DIR = "saved_models/"
# RESULTS_DIR = "results/"

# IMG_SIZE = 224
# BATCH_SIZE = 16
# EPOCHS = 15
# LR = 5e-4
# VAL_SPLIT = 0.15
# TEST_SPLIT = 0.15
# NUM_CLASSES = 37


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)

# print(f"🚀 Using device: {DEVICE}")

# # ────────────────────── SPEC AUGMENT ──────────────────────
# class SpecAugment:
#     def __call__(self, img):
#         img = img.clone()
#         # Time masking 
#         t = random.randint(0, img.shape[1]//4)
#         t0 = random.randint(0, img.shape[1]-t)
#         img[:, t0:t0+t] = 0

#         # Frequency masking 
#         f = random.randint(0, img.shape[2]//4)
#         f0 = random.randint(0, img.shape[2]-f)
#         img[:, :, f0:f0+f] = 0

#         return img

# # ────────────────────── DATA ──────────────────────
# def get_data_loaders(data_dir):
    
#     base_dataset = datasets.ImageFolder(root=data_dir)
#     print(f"✅ Found {len(base_dataset)} images across {len(base_dataset.classes)} classes")

#     with open(os.path.join(RESULTS_DIR, "class_names.json"), "w") as f:
#         json.dump(base_dataset.classes, f)

#     total = len(base_dataset)
#     val_size = int(total * VAL_SPLIT)
#     test_size = int(total * TEST_SPLIT)
#     train_size = total - val_size - test_size

#     indices = torch.randperm(total)
#     train_idx = indices[:train_size]
#     val_idx   = indices[train_size:train_size+val_size]
#     test_idx  = indices[train_size+val_size:]

#     # --- Transforms Grayscale ---
   
#     train_transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1), 
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.RandomHorizontalFlip(), 
#         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
#         transforms.GaussianBlur(3),
#         transforms.ToTensor(),
#         SpecAugment(),
#         transforms.Normalize([0.5], [0.5]) 
#     ])

#     eval_transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1), 
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]) 
#     ])

   
#     train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_transform), train_idx)
#     val_ds   = Subset(datasets.ImageFolder(data_dir, transform=eval_transform), val_idx)
#     test_ds  = Subset(datasets.ImageFolder(data_dir, transform=eval_transform), test_idx)

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
#     test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

#     return train_loader, val_loader, test_loader, base_dataset.classes

# # ────────────────────── MODEL (RFNet) ──────────────────────
# class RFNet(nn.Module):
#     def __init__(self, num_classes=37):
#         super(RFNet, self).__init__()

#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1) 
#         self.bn1   = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2   = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn3   = nn.BatchNorm2d(128)

#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#         self.bn4   = nn.BatchNorm2d(256)

#         self.gap = nn.AdaptiveAvgPool2d((1,1))

#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = torch.max_pool2d(x, 2)

#         x = torch.relu(self.bn2(self.conv2(x)))
#         x = torch.max_pool2d(x, 2)

#         x = torch.relu(self.bn3(self.conv3(x)))
#         x = torch.max_pool2d(x, 2)

#         x = torch.relu(self.bn4(self.conv4(x)))

#         x = self.gap(x)
#         x = x.view(x.size(0), -1)

#         return self.fc(x)

# # ────────────────────── TRAIN & EVAL ──────────────────────
# def train_one_epoch(model, loader, optimizer, criterion):
#     model.train()
#     total_loss, correct, total = 0, 0, 0

#     for images, labels in loader:
#         images, labels = images.to(DEVICE), labels.to(DEVICE)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         _, pred = outputs.max(1)
#         correct += pred.eq(labels).sum().item()
#         total += labels.size(0)

#     return total_loss/len(loader), 100*correct/total

# def evaluate(model, loader, criterion):
#     model.eval()
#     total_loss, correct, total = 0, 0, 0

#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)

#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             total_loss += loss.item()
#             _, pred = outputs.max(1)
#             correct += pred.eq(labels).sum().item()
#             total += labels.size(0)

#     return total_loss/len(loader), 100*correct/total

# # ────────────────────── INFERENCE ──────────────────────
# def predict_image(model, img_path, class_names):
    
#     inference_transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])

#     if not os.path.exists(img_path):
#         return "Image not found."

#     img = Image.open(img_path).convert("L") 
#     img = inference_transform(img).unsqueeze(0).to(DEVICE)

#     model.eval()
#     with torch.no_grad():
#         output = model(img)
#         probs = torch.softmax(output, dim=1)
#         top5 = torch.topk(probs, min(5, NUM_CLASSES))

#     results = []
#     for i in range(top5.indices.size(1)):
#         cls = class_names[top5.indices[0][i]]
#         conf = top5.values[0][i].item()*100
#         results.append((cls, conf))

#     return results

# # ────────────────────── MAIN ──────────────────────
# def main():
   
#     if not os.path.exists(DATA_DIR):
#         print(f"❌ Error: Data directory {DATA_DIR} not found!")
#         return

#     train_loader, val_loader, test_loader, class_names = get_data_loaders(DATA_DIR)

#     model = RFNet(NUM_CLASSES).to(DEVICE)

#     print(f"🧠 Total Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

#     best_val_acc = 0
#     patience = 5
#     counter = 0

#     print("\n--- Starting Training (Grayscale Mode) ---")
#     for epoch in range(1, EPOCHS+1):
#         t0 = time.time()

#         train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
#         val_loss, val_acc = evaluate(model, val_loader, criterion)

#         duration = int(time.time() - t0)
#         print(f"Epoch {epoch}/{EPOCHS}: Train Loss {train_loss:.4f}, Acc {train_acc:.1f}% | Val Loss {val_loss:.4f}, Acc {val_acc:.1f}% | Time: {duration}s")

       
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             counter = 0
#             torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model_gray.pth"))
#             print("💾 Saved best model (Grayscale)")
#         else:
#             counter += 1

#         if counter >= patience:
#             print("⛔ Early stopping triggered.")
#             break

#     model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model_gray.pth")))
#     test_loss, test_acc = evaluate(model, test_loader, criterion)
#     print(f"\n🧪 Final Test Accuracy: {test_acc:.2f}%")

# if __name__ == "__main__":
#     main()








# ###################################
# ########## RFnet v3 ###############
# ###################################

# import os, json, time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
# from torchvision import datasets, transforms

# # ───────────────── CONFIG ─────────────────
# DATA_DIR = "D:/gp data/RFUAV_valid/"
# MODEL_DIR = "saved_models/"
# RESULTS_DIR = "results/"

# IMG_SIZE = 224
# BATCH_SIZE = 16
# EPOCHS = 20
# LR = 3e-4
# NUM_CLASSES = 37

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(RESULTS_DIR, exist_ok=True)

# print("🚀 Device:", DEVICE)

# # ───────────────── RFNET V3 MODEL ─────────────────
# import torch.nn.functional as F

# class CNNBackbone(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)

#         self.pool = nn.MaxPool2d(2)

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)

#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x)

#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool(x)

#         x = F.relu(self.bn4(self.conv4(x)))

#         return x


# class PatchEmbedding(nn.Module):
#     def __init__(self, in_ch=256, dim=128):
#         super().__init__()
#         self.proj = nn.Conv2d(in_ch, dim, 1)

#     def forward(self, x):
#         x = self.proj(x)
#         B, C, H, W = x.shape
#         x = x.flatten(2).transpose(1, 2)
#         return x


# class TransformerBlock(nn.Module):
#     def __init__(self, dim=128, heads=4):
#         super().__init__()

#         self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

#         self.norm1 = nn.LayerNorm(dim)
#         self.norm2 = nn.LayerNorm(dim)

#         self.mlp = nn.Sequential(
#             nn.Linear(dim, dim * 2),
#             nn.ReLU(),
#             nn.Linear(dim * 2, dim)
#         )

#     def forward(self, x):
#         attn_out, _ = self.attn(x, x, x)
#         x = self.norm1(x + attn_out)

#         mlp_out = self.mlp(x)
#         x = self.norm2(x + mlp_out)

#         return x


# class RFNetV3(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.cnn = CNNBackbone()
#         self.patch = PatchEmbedding(256, 128)

#         self.transformer = nn.Sequential(
#             TransformerBlock(128, 4),
#             TransformerBlock(128, 4)
#         )

#         self.classifier = nn.Sequential(
#             nn.LayerNorm(128),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.cnn(x)
#         x = self.patch(x)
#         x = self.transformer(x)
#         x = x.mean(dim=1)
#         return self.classifier(x)

# # ───────────────── DATA ─────────────────
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# # save class names
# with open(os.path.join(RESULTS_DIR, "class_names.json"), "w") as f:
#     json.dump(dataset.classes, f)

# NUM_CLASSES = len(dataset.classes)

# # split
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size

# train_ds, val_ds = random_split(dataset, [train_size, val_size])

# val_ds.dataset.transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# print(f"Dataset: {len(dataset)} images | Classes: {NUM_CLASSES}")

# # ───────────────── TRAIN ─────────────────
# model = RFNetV3(NUM_CLASSES).to(DEVICE)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# best_acc = 0

# def evaluate(model, loader):
#     model.eval()
#     correct, total = 0, 0

#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(DEVICE), y.to(DEVICE)

#             out = model(x)
#             pred = out.argmax(1)

#             correct += (pred == y).sum().item()
#             total += y.size(0)

#     return 100 * correct / total


# print("\n🔥 Training RFNet V3 Started...\n")

# for epoch in range(EPOCHS):

#     model.train()
#     total_loss = 0

#     for x, y in train_loader:
#         x, y = x.to(DEVICE), y.to(DEVICE)

#         optimizer.zero_grad()

#         out = model(x)
#         loss = criterion(out, y)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     val_acc = evaluate(model, val_loader)

#     print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

#     if val_acc > best_acc:
#         best_acc = val_acc
#         torch.save(model.state_dict(), os.path.join(MODEL_DIR, "rfnet_v3_best.pth"))
#         print("💾 Saved Best RFNet V3")

# print("\n✅ Training Finished!")
# print("Best Accuracy:", best_acc)





###################################
######## ANALYSIS ONLY ############
###################################

import os, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import collections

# ───────────────── CONFIG ─────────────────
DATA_DIR = "D:/gp data/RFUAV_valid/"
MODEL_PATH = "saved_models/rfnet_v3_best.pth"
RESULTS_DIR = "results/"

IMG_SIZE = 224
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🚀 Device:", DEVICE)

# ───────────────── MODEL ─────────────────
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, in_ch=256, dim=128):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, 1)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim=128, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x


class RFNetV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNBackbone()
        self.patch = PatchEmbedding(256, 128)
        self.transformer = nn.Sequential(
            TransformerBlock(128, 4),
            TransformerBlock(128, 4)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.patch(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

# ───────────────── DATA ─────────────────
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes

test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ───────────────── LOAD MODEL ─────────────────
model = RFNetV3(len(class_names)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model Loaded Successfully")

# ───────────────── EVALUATE ─────────────────
def evaluate(model, loader):
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

test_acc = evaluate(model, test_loader)
print(f"\n🔥 Test Accuracy: {test_acc:.2f}%")

# ───────────────── PREDICTIONS ─────────────────
def get_preds(model, loader):
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(DEVICE)).argmax(1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(pred)
    return np.array(y_true), np.array(y_pred)

y_true, y_pred = get_preds(model, test_loader)

# ───────────────── CONFUSION MATRIX ─────────────────
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# ───────────────── REPORT ─────────────────
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# ───────────────── ERRORS ─────────────────
errors = y_true[y_true != y_pred]
counter = collections.Counter(errors)

print("\n⚠️ Most Confused Classes:")
for cls, count in counter.most_common(10):
    print(f"{class_names[cls]}: {count}")