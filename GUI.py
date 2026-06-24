# """
# Drone RF Fingerprint Predictor - GUI
# ====================================
# GUI لتجربة نموذج ResNet18 على أي صورة تريدها
# باستخدام PyQt5
# """

# import sys, os, json, torch
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout
# from PyQt5.QtGui import QPixmap
# from torchvision import models, transforms
# from PIL import Image

# # ───── CONFIG ─────
# DEVICE = torch.device("cpu")
# MODEL_DIR = "saved_models/"
# RESULTS_DIR = "results/"
# IMG_SIZE = 224
# NUM_CLASSES = 37

# # ───── LOAD MODEL ─────
# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# for param in list(model.parameters())[:-10]:
#     param.requires_grad = False
# model.fc = torch.nn.Sequential(torch.nn.Dropout(0.4),
#                                torch.nn.Linear(model.fc.in_features, NUM_CLASSES))

# model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth"), map_location=DEVICE))
# model.to(DEVICE)
# model.eval()

# # load class names
# with open(os.path.join(RESULTS_DIR, "class_names.json"), "r") as f:
#     class_names = json.load(f)

# # ───── IMAGE TRANSFORM ─────
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# # ───── GUI ─────
# class DronePredictor(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Drone RF Fingerprint Predictor")
#         self.setGeometry(100, 100, 600, 500)

#         self.label_img = QLabel("No Image Selected")
#         self.label_img.setFixedSize(400, 300)
#         self.label_img.setStyleSheet("border: 1px solid black;")
#         self.label_result = QLabel("Prediction: None")

#         self.btn_open = QPushButton("Open Image")
#         self.btn_open.clicked.connect(self.open_image)

#         layout = QVBoxLayout()
#         layout.addWidget(self.label_img)
#         layout.addWidget(self.label_result)
#         layout.addWidget(self.btn_open)

#         self.setLayout(layout)

#     def open_image(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
#         if file_path:
#             pixmap = QPixmap(file_path)
#             pixmap = pixmap.scaled(self.label_img.width(), self.label_img.height())
#             self.label_img.setPixmap(pixmap)

#             pred = self.predict_image(file_path)
#             self.label_result.setText(f"Prediction: {pred}")

#     def predict_image(self, img_path):
#         img = Image.open(img_path).convert("RGB")
#         img_tensor = transform(img).unsqueeze(0)
#         with torch.no_grad():
#             output = model(img_tensor.to(DEVICE))
#             pred_class = output.argmax(dim=1).item()
#         return class_names[pred_class]

# # ───── RUN APP ─────
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = DronePredictor()
#     window.show()
#     sys.exit(app.exec_())









#########################################
################## V2 ###################
#########################################

# """
# Advanced Drone RF Fingerprint Predictor - GUI
# ============================================
# GUI متقدم لتجربة نموذج ResNet18 على أي صورة
# ويعرض أعلى 5 احتمالات مع الثقة + حفظ النتائج مع الوقت
# """

# import sys, os, json, torch, csv
# from datetime import datetime
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout, QTextEdit
# from PyQt5.QtGui import QPixmap
# from torchvision import models, transforms
# from PIL import Image

# # ───── CONFIG ─────
# DEVICE = torch.device("cpu")
# MODEL_DIR = "saved_models/"
# RESULTS_DIR = "results/"
# IMG_SIZE = 224
# NUM_CLASSES = 37
# CSV_FILE = os.path.join(RESULTS_DIR, "predictions.csv")

# # ───── LOAD MODEL ─────
# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# for param in list(model.parameters())[:-10]:
#     param.requires_grad = False
# model.fc = torch.nn.Sequential(
#     torch.nn.Dropout(0.4),
#     torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
# )

# model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth"), map_location=DEVICE))
# model.to(DEVICE)
# model.eval()

# # load class names
# with open(os.path.join(RESULTS_DIR, "class_names.json"), "r") as f:
#     class_names = json.load(f)

# # ───── IMAGE TRANSFORM ─────
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# # ───── GUI ─────
# class DronePredictor(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Advanced Drone RF Fingerprint Predictor")
#         self.setGeometry(100, 100, 700, 600)

#         # Widgets
#         self.label_img = QLabel("No Image Selected")
#         self.label_img.setFixedSize(400, 300)
#         self.label_img.setStyleSheet("border: 1px solid black;")

#         self.text_result = QTextEdit()
#         self.text_result.setReadOnly(True)
#         self.text_result.setFixedHeight(200)

#         self.btn_open = QPushButton("Open Image")
#         self.btn_open.clicked.connect(self.open_image)

#         # Layout
#         layout = QVBoxLayout()
#         layout.addWidget(self.label_img)
#         layout.addWidget(self.text_result)
#         layout.addWidget(self.btn_open)

#         self.setLayout(layout)

#         # Ensure CSV exists
#         os.makedirs(RESULTS_DIR, exist_ok=True)
#         if not os.path.exists(CSV_FILE):
#             with open(CSV_FILE, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["time", "image", "prediction", "top5_probs"])

#     def open_image(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
#         if file_path:
#             # Show image
#             pixmap = QPixmap(file_path)
#             pixmap = pixmap.scaled(self.label_img.width(), self.label_img.height())
#             self.label_img.setPixmap(pixmap)

#             # Predict
#             top_pred, top5 = self.predict_image(file_path)

#             # Display results
#             text = f"Prediction: {top_pred}\nTop 5 Predictions:\n"
#             for cls, prob in top5:
#                 text += f"  {cls}: {prob:.2f}%\n"
#             self.text_result.setText(text)

#             # Save to CSV with timestamp
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             with open(CSV_FILE, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([timestamp, file_path, top_pred, json.dumps(top5)])

#     def predict_image(self, img_path):
#         img = Image.open(img_path).convert("RGB")
#         img_tensor = transform(img).unsqueeze(0)
#         with torch.no_grad():
#             output = model(img_tensor.to(DEVICE))
#             probs = torch.softmax(output, dim=1)
#             top5_prob, top5_idx = probs.topk(5)
#             top5 = [(class_names[i], p.item()*100) for i,p in zip(top5_idx[0], top5_prob[0])]
#         return top5[0][0], top5  # Return top prediction + top5 list

# # ───── RUN APP ─────
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = DronePredictor()
#     window.show()
#     sys.exit(app.exec_())








# #########################################
# ############# V3  ResNet18 ##############
# #########################################
# """
# Drone RF Fingerprint GUI with Integrated Plots
# ===============================================
# كل شيء في نفس الـ GUI:
# - اختيار صورة وتجربة النموذج
# - عرض اسم الدرون + أعلى 5 احتمالات
# - Timeline و Bar Chart مدمجين
# """

# import sys, os, json, torch, csv
# from datetime import datetime
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout, QTextEdit, QHBoxLayout, QSizePolicy
# from PyQt5.QtGui import QPixmap
# from torchvision import models, transforms
# from PIL import Image
# import pandas as pd
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure

# # ───── CONFIG ─────
# DEVICE = torch.device("cpu")
# MODEL_DIR = "saved_models/"
# RESULTS_DIR = "results/"
# IMG_SIZE = 224
# NUM_CLASSES = 37
# CSV_FILE = os.path.join(RESULTS_DIR, "predictions.csv")

# # ───── LOAD MODEL ─────
# model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# for param in list(model.parameters())[:-10]:
#     param.requires_grad = False
# model.fc = torch.nn.Sequential(
#     torch.nn.Dropout(0.4),
#     torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
# )
# model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pth"), map_location=DEVICE))
# model.to(DEVICE)
# model.eval()

# # load class names
# with open(os.path.join(RESULTS_DIR, "class_names.json"), "r") as f:
#     class_names = json.load(f)

# # ───── IMAGE TRANSFORM ─────
# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# # ───── GUI ─────
# class DronePredictorGUI(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Drone RF Fingerprint Predictor + Timeline")
#         self.setGeometry(100, 100, 1000, 800)

#         # Widgets
#         self.label_img = QLabel("No Image Selected")
#         self.label_img.setFixedSize(400, 300)
#         self.label_img.setStyleSheet("border: 1px solid black;")

#         self.text_result = QTextEdit()
#         self.text_result.setReadOnly(True)
#         self.text_result.setFixedHeight(150)

#         self.btn_open = QPushButton("Open Image")
#         self.btn_open.clicked.connect(self.open_image)

#         # Matplotlib Canvas
#         self.figure = Figure(figsize=(8,5))
#         self.canvas = FigureCanvas(self.figure)
#         self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

#         # Layout
#         layout = QVBoxLayout()
#         layout.addWidget(self.label_img)
#         layout.addWidget(self.text_result)
#         layout.addWidget(self.canvas)
#         layout.addWidget(self.btn_open)
#         self.setLayout(layout)

#         # Ensure CSV exists
#         os.makedirs(RESULTS_DIR, exist_ok=True)
#         if not os.path.exists(CSV_FILE):
#             with open(CSV_FILE, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(["time", "image", "prediction", "top5_probs"])

#     # ─── Open & Predict Image ───
#     def open_image(self):
#         file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
#         if file_path:
#             # Show image
#             pixmap = QPixmap(file_path)
#             pixmap = pixmap.scaled(self.label_img.width(), self.label_img.height())
#             self.label_img.setPixmap(pixmap)

#             # Predict
#             top_pred, top5 = self.predict_image(file_path)

#             # Display results
#             text = f"Prediction: {top_pred}\nTop 5 Predictions:\n"
#             for cls, prob in top5:
#                 text += f"  {cls}: {prob:.2f}%\n"
#             self.text_result.setText(text)

#             # Save to CSV with timestamp
#             timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             with open(CSV_FILE, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([timestamp, file_path, top_pred, json.dumps(top5)])

#             # Update plots
#             self.update_plots()

#     # ─── Prediction Logic ───
#     def predict_image(self, img_path):
#         img = Image.open(img_path).convert("RGB")
#         img_tensor = transform(img).unsqueeze(0)
#         with torch.no_grad():
#             output = model(img_tensor.to(DEVICE))
#             probs = torch.softmax(output, dim=1)
#             top5_prob, top5_idx = probs.topk(5)
#             top5 = [(class_names[i], p.item()*100) for i,p in zip(top5_idx[0], top5_prob[0])]
#         return top5[0][0], top5  # Top prediction + top5 list

#     # ─── Update Timeline + Bar Chart ───
#     def update_plots(self):
#         if not os.path.exists(CSV_FILE):
#             return

#         df = pd.read_csv(CSV_FILE)
#         df['time'] = pd.to_datetime(df['time'])
#         drones = df['prediction'].unique()

#         self.figure.clear()

#         ax1 = self.figure.add_subplot(2,1,1)  # Timeline
#         for drone in drones:
#             times = df[df['prediction']==drone]['time']
#             ax1.scatter(times, [drone]*len(times), alpha=0.6)
#         ax1.set_xlabel("Time")
#         ax1.set_ylabel("Drone Type")
#         ax1.set_title("Drone Appearance Timeline")
#         ax1.tick_params(axis='x', rotation=45)

#         ax2 = self.figure.add_subplot(2,1,2)  # Bar chart
#         counts = df['prediction'].value_counts()
#         ax2.bar(counts.index, counts.values, color='skyblue')
#         ax2.set_ylabel("Count")
#         ax2.set_xlabel("Drone Type")
#         ax2.set_title("Number of Appearances per Drone")
#         ax2.tick_params(axis='x', rotation=45)

#         self.figure.tight_layout()
#         self.canvas.draw()


# # ───── RUN APP ─────
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = DronePredictorGUI()
#     window.show()
#     sys.exit(app.exec_())








# import sys, os, json, csv
# from datetime import datetime

# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QLabel, QPushButton,
#     QFileDialog, QVBoxLayout, QTextEdit, QSizePolicy
# )
# from PyQt5.QtGui import QPixmap

# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image

# # ───────────────── CONFIG ─────────────────
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_PATH = "saved_models/best_model_gray.pth"
# RESULTS_DIR = "results/"
# CSV_FILE = os.path.join(RESULTS_DIR, "predictions.csv")
# IMG_SIZE = 224
# NUM_CLASSES = 37

# os.makedirs(RESULTS_DIR, exist_ok=True)

# # ───────────────── MODEL ─────────────────
# class RFNet(nn.Module):
#     def __init__(self, num_classes=37):
#         super().__init__()

#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)

#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)

#         self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)

#         self.gap = nn.AdaptiveAvgPool2d((1, 1))
#         # self.fc = nn.Linear(256, num_classes)
#         self.fc = nn.Sequential(
#                     nn.Dropout(0.5),
#                     nn.Linear(256, num_classes)
#                 )

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

# # ───────────────── LOAD CLASS NAMES ─────────────────
# with open(os.path.join(RESULTS_DIR, "class_names.json"), "r") as f:
#     class_names = json.load(f)

# # ───────────────── LOAD MODEL ─────────────────
# model = RFNet(NUM_CLASSES).to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# model.eval()

# # ───────────────── TRANSFORM (MATCH TRAINING) ─────────────────
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])

# # ───────────────── GUI ─────────────────
# class DroneGUI(QWidget):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("RF Drone Classifier")
#         self.setGeometry(100, 100, 900, 600)

#         self.img_label = QLabel("No Image")
#         self.img_label.setFixedSize(400, 300)
#         self.img_label.setStyleSheet("border: 1px solid black;")

#         self.result_box = QTextEdit()
#         self.result_box.setReadOnly(True)

#         self.btn = QPushButton("Open Image")
#         self.btn.clicked.connect(self.load_image)

#         layout = QVBoxLayout()
#         layout.addWidget(self.img_label)
#         layout.addWidget(self.result_box)
#         layout.addWidget(self.btn)

#         self.setLayout(layout)

#         # CSV init
#         if not os.path.exists(CSV_FILE):
#             with open(CSV_FILE, "w", newline="") as f:
#                 csv.writer(f).writerow(["time", "image", "prediction"])

#     # ───────── PREDICT ─────────
#     def predict(self, img_path):

#         img = Image.open(img_path).convert("L")
#         img = transform(img).unsqueeze(0).to(DEVICE)

#         with torch.no_grad():
#             out = model(img)
#             probs = torch.softmax(out, dim=1)

#             top_prob, top_idx = torch.topk(probs, 5)

#             results = []
#             for i in range(5):
#                 cls = class_names[top_idx[0][i]]
#                 conf = top_prob[0][i].item() * 100
#                 results.append((cls, conf))

#         return results

#     # ───────── LOAD IMAGE ─────────
#     def load_image(self):
#         path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")

#         if path:
#             pixmap = QPixmap(path).scaled(400, 300)
#             self.img_label.setPixmap(pixmap)

#             results = self.predict(path)

#             text = "Top Predictions:\n\n"
#             for c, p in results:
#                 text += f"{c}: {p:.2f}%\n"

#             self.result_box.setText(text)

#             # save log
#             with open(CSV_FILE, "a", newline="") as f:
#                 csv.writer(f).writerow([
#                     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#                     path,
#                     results[0][0]
#                 ])

# # ───────────────── RUN ─────────────────
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     win = DroneGUI()
#     win.show()
#     sys.exit(app.exec_())






















# import sys, json
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QLabel, QPushButton,
#     QFileDialog, QVBoxLayout, QTextEdit
# )
# from PyQt5.QtGui import QPixmap
# from PIL import Image

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import transforms


# # ───────── CONFIG ─────────
# MODEL_PATH = "saved_models/rfnet_v3_best.pth"
# CLASS_PATH = "results/class_names.json"

# IMG_SIZE = 224
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # ───────── MODEL ─────────
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
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = F.relu(self.bn4(self.conv4(x)))
#         return x


# class PatchEmbedding(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.proj = nn.Conv2d(256, 128, 1)

#     def forward(self, x):
#         x = self.proj(x)
#         B, C, H, W = x.shape
#         return x.flatten(2).transpose(1, 2)


# class TransformerBlock(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
#         self.norm1 = nn.LayerNorm(128)
#         self.norm2 = nn.LayerNorm(128)

#         self.mlp = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )

#     def forward(self, x):
#         attn_out, _ = self.attn(x, x, x)
#         x = self.norm1(x + attn_out)
#         x = self.norm2(x + self.mlp(x))
#         return x


# class RFNetV3(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.cnn = CNNBackbone()
#         self.patch = PatchEmbedding()

#         self.transformer = nn.ModuleList([
#             TransformerBlock(),
#             TransformerBlock()
#         ])

#         self.classifier = nn.Sequential(
#             nn.LayerNorm(128),
#             nn.Dropout(0.5),
#             nn.Linear(128, num_classes)
#         )

#     def forward(self, x):
#         x = self.cnn(x)
#         x = self.patch(x)

#         for block in self.transformer:
#             x = block(x)

#         x = x.mean(dim=1)
#         return self.classifier(x)


# # ───────── LOAD CLASS NAMES ─────────
# with open(CLASS_PATH, "r") as f:
#     class_names = json.load(f)

# # ───────── LOAD MODEL ─────────
# model = RFNetV3(len(class_names)).to(DEVICE)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.eval()

# print("✅ Model Loaded")


# # ───────── TRANSFORM ─────────
# transform = transforms.Compose([
#     transforms.Grayscale(1),
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5], [0.5])
# ])


# # ───────── GUI ─────────
# class App(QWidget):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle("RFNet v3 Drone Predictor")
#         self.setGeometry(100, 100, 600, 600)

#         self.image_label = QLabel("No Image Selected")
#         self.image_label.setFixedSize(400, 300)

#         self.result_text = QTextEdit()
#         self.result_text.setReadOnly(True)

#         self.btn = QPushButton("Open Image")
#         self.btn.clicked.connect(self.load_image)

#         layout = QVBoxLayout()
#         layout.addWidget(self.image_label)
#         layout.addWidget(self.result_text)
#         layout.addWidget(self.btn)

#         self.setLayout(layout)

#     def load_image(self):
#         file_path, _ = QFileDialog.getOpenFileName(
#             self, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
#         )

#         if not file_path:
#             return

#         # عرض الصورة
#         pixmap = QPixmap(file_path).scaled(
#             400, 300,
#             aspectRatioMode=1
#         )
#         self.image_label.setPixmap(pixmap)

#         # Prediction
#         try:
#             pred, top5 = self.predict(file_path)

#             text = f"🎯 Prediction: {pred}\n\n"
#             text += "🔝 Top 5 Results:\n\n"

#             for cls, prob in top5:
#                 text += f"• {cls}: {prob:.2f}%\n"

#             self.result_text.setText(text)

#         except Exception as e:
#             self.result_text.setText(f"Error:\n{str(e)}")

#     def predict(self, path):
#         img = Image.open(path).convert("L")
#         img = transform(img).unsqueeze(0).to(DEVICE)

#         with torch.no_grad():
#             out = model(img)
#             probs = torch.softmax(out, dim=1)

#             top5_prob, top5_idx = probs.topk(5)

#             results = []
#             for i in range(5):
#                 idx = top5_idx[0][i].item()
#                 cls = class_names[str(idx)]
#                 prob = top5_prob[0][i].item() * 100
#                 results.append((cls, prob))

#         return results[0][0], results


# # ───────── RUN ─────────
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = App()
#     window.show()
#     sys.exit(app.exec_())











import sys
import os
import json
import csv
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QTextEdit,
    QSizePolicy
)

from PyQt5.QtGui import QPixmap

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

from torchvision import transforms


# =========================================================
# CONFIG
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "saved_models"
RESULTS_DIR = "results"

CSV_FILE = os.path.join(RESULTS_DIR, "predictions.csv")

IMG_SIZE = 224
NUM_CLASSES = 37

BEST_MODEL = os.path.join(MODEL_DIR, "rfnet_v3_best.pth")


# =========================================================
# TRANSFORMER BLOCK
# =========================================================

class TransformerBlock(nn.Module):

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):

        attn_output, _ = self.attn(x, x, x)

        x = self.norm1(x + attn_output)

        mlp_output = self.mlp(x)

        x = self.norm2(x + mlp_output)

        return x


# =========================================================
# RFNET V3
# =========================================================

class RFNet(nn.Module):

    def __init__(
        self,
        num_classes=37,
        patch_size=16,
        embed_dim=128,
        num_heads=4,
        depth=2
    ):
        super(RFNet, self).__init__()

        # ---------------- CNN Backbone ----------------

        self.cnn = nn.Sequential()

        self.cnn.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.cnn.bn1 = nn.BatchNorm2d(32)

        self.cnn.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.cnn.bn2 = nn.BatchNorm2d(64)

        self.cnn.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.cnn.bn3 = nn.BatchNorm2d(128)

        self.cnn.conv4 = nn.Conv2d(128, embed_dim, 3, padding=1)
        self.cnn.bn4 = nn.BatchNorm2d(embed_dim)

        # ---------------- Patch Embedding ----------------

        self.patch = nn.Sequential()

        self.patch.proj = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # ---------------- Transformer ----------------

        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        # ---------------- Classifier ----------------

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = torch.relu(self.cnn.bn1(self.cnn.conv1(x)))
        x = torch.max_pool2d(x, 2)

        x = torch.relu(self.cnn.bn2(self.cnn.conv2(x)))
        x = torch.max_pool2d(x, 2)

        x = torch.relu(self.cnn.bn3(self.cnn.conv3(x)))
        x = torch.max_pool2d(x, 2)

        x = torch.relu(self.cnn.bn4(self.cnn.conv4(x)))

        # Patch Embedding
        x = self.patch.proj(x)

        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)

        # Transformer
        for block in self.transformer:
            x = block(x)

        # Global Average Pooling
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)

        return x


# =========================================================
# LOAD CLASS NAMES
# =========================================================

with open(os.path.join(RESULTS_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)


# =========================================================
# LOAD MODEL
# =========================================================

model = RFNet(num_classes=NUM_CLASSES).to(DEVICE)

state_dict = torch.load(BEST_MODEL, map_location=DEVICE)

model.load_state_dict(state_dict)

model.eval()


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


# =========================================================
# GUI
# =========================================================

class DronePredictorGUI(QWidget):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("RFNet V3 Drone Predictor")
        self.setGeometry(100, 100, 1000, 800)

        # ---------------- IMAGE LABEL ----------------

        self.label_img = QLabel("No Image Selected")

        self.label_img.setFixedSize(400, 300)

        self.label_img.setStyleSheet(
            "border: 1px solid black;"
        )

        # ---------------- RESULT TEXT ----------------

        self.text_result = QTextEdit()

        self.text_result.setReadOnly(True)

        self.text_result.setFixedHeight(170)

        # ---------------- BUTTON ----------------

        self.btn_open = QPushButton("Open Image")

        self.btn_open.clicked.connect(self.open_image)

        # ---------------- MATPLOTLIB ----------------

        self.figure = Figure(figsize=(8, 5))

        self.canvas = FigureCanvas(self.figure)

        self.canvas.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        # ---------------- LAYOUT ----------------

        layout = QVBoxLayout()

        layout.addWidget(self.label_img)
        layout.addWidget(self.text_result)
        layout.addWidget(self.canvas)
        layout.addWidget(self.btn_open)

        self.setLayout(layout)

        # ---------------- CREATE CSV ----------------

        os.makedirs(RESULTS_DIR, exist_ok=True)

        if not os.path.exists(CSV_FILE):

            with open(
                CSV_FILE,
                "w",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    "time",
                    "image",
                    "prediction",
                    "top5_probs"
                ])

    # =====================================================
    # OPEN IMAGE
    # =====================================================

    def open_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if file_path:

            pixmap = QPixmap(file_path)

            pixmap = pixmap.scaled(
                self.label_img.width(),
                self.label_img.height()
            )

            self.label_img.setPixmap(pixmap)

            top_pred, top5 = self.predict_image(file_path)

            # ---------------- SHOW RESULT ----------------

            text = f"Prediction: {top_pred}\n\n"

            text += "Top 5 Predictions:\n"

            for cls, prob in top5:
                text += f"{cls}: {prob:.2f}%\n"

            self.text_result.setText(text)

            # ---------------- SAVE CSV ----------------

            timestamp = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            with open(
                CSV_FILE,
                "a",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    timestamp,
                    file_path,
                    top_pred,
                    json.dumps(top5)
                ])

            self.update_plots()

    # =====================================================
    # PREDICT IMAGE
    # =====================================================

    def predict_image(self, img_path):

        img = Image.open(img_path).convert("L")

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        print("Input shape:", img_tensor.shape)

        with torch.no_grad():

            output = model(img_tensor)

            probs = torch.softmax(output, dim=1)

            top5_prob, top5_idx = probs.topk(5)

            top5 = [
                (
                    class_names[i],
                    p.item() * 100
                )
                for i, p in zip(
                    top5_idx[0],
                    top5_prob[0]
                )
            ]

        return top5[0][0], top5

    # =====================================================
    # UPDATE PLOTS
    # =====================================================

    def update_plots(self):

        if not os.path.exists(CSV_FILE):
            return

        df = pd.read_csv(CSV_FILE)

        df['time'] = pd.to_datetime(
            df['time'],
            format='mixed',
            errors='coerce'
        )

        df = df.dropna(subset=['time'])

        drones = df['prediction'].unique()

        self.figure.clear()

        # ---------------- TIMELINE ----------------

        ax1 = self.figure.add_subplot(2, 1, 1)

        for drone in drones:

            times = df[
                df['prediction'] == drone
            ]['time']

            ax1.scatter(
                times,
                [drone] * len(times),
                alpha=0.6
            )

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Drone Type")
        ax1.set_title("Drone Appearance Timeline")

        ax1.tick_params(
            axis='x',
            rotation=45
        )

        # ---------------- BAR CHART ----------------

        ax2 = self.figure.add_subplot(2, 1, 2)

        counts = df['prediction'].value_counts()

        ax2.bar(
            counts.index,
            counts.values
        )

        ax2.set_ylabel("Count")
        ax2.set_xlabel("Drone Type")

        ax2.set_title(
            "Number of Appearances per Drone"
        )

        ax2.tick_params(
            axis='x',
            rotation=45
        )

        self.figure.tight_layout()

        self.canvas.draw()


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = DronePredictorGUI()

    window.show()

    sys.exit(app.exec_())