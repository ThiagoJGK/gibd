---
aliases: [FasterRCNNClaude_21_11]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-11-28
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/E.C de imágenes a color/Faster RCNN/FasterRCNNClaude_21_11.ipynb"
tamanio_bytes: 465436
---

# Notebook: FasterRCNNClaude_21_11.ipynb

Ruta interna: `GIBD/E.C de imágenes a color/Faster RCNN/FasterRCNNClaude_21_11.ipynb`

---

# **Imports y montar el drive**


**[Celda 2 - Código]**
```python
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from google.colab import drive
drive.mount('/content/drive')
```


*Salida:*
```text
Mounted at /content/drive
```

# **Definición de rutas**


**[Celda 4 - Código]**
```python
# Definir las rutas correctas
BASE_DIR = "/content/drive/MyDrive/GIBD/E.C de imágenes a color/Faster RCNN/badgesbbfromroboflow"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "_annotations.coco.json")
```

# **Funciones**


**[Celda 6 - Código]**
```python

class ClubLogosDataset(Dataset):
    def __init__(self, annotation_path=ANNOTATIONS_FILE, img_folder=TRAIN_DIR, transforms=None):
        self.coco = COCO(annotation_path)
        self.img_folder = img_folder
        self.transforms = transforms
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        for ann in anns:
            x, y, width, height = ann['bbox']
            boxes.append([x, y, x + width, y + height])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform():
    def transform(image, target):
        image = F.to_tensor(image)
        return image, target

    return transform

def train_model(train_loader, num_classes):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 1
    model.to(device)

    print("Iniciando entrenamiento...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i}], Loss: {losses.item():.4f}")

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss/len(train_loader):.4f}")

    return model

def inference(model, image_path, confidence_threshold=0.5):
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_transform()
    image_tensor, _ = transform(image_rgb, {})
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(image_tensor)

    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    filtered_boxes = boxes[scores >= confidence_threshold]
    filtered_labels = labels[scores >= confidence_threshold]

    for box, label in zip(filtered_boxes, filtered_labels):
        cv2.rectangle(image,
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 255, 0), 2)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
```

# **Entrenamiento**


**[Celda 8 - Código]**
```python
# Iniciar entrenamiento
print("Preparando dataset...")
train_dataset = ClubLogosDataset(transforms=get_transform())
train_loader = DataLoader(train_dataset,
                         batch_size=2,
                         shuffle=True,
                         collate_fn=lambda x: tuple(zip(*x)))

num_classes = len(train_dataset.coco.cats) + 1  # +1 para el background
print(f"Número de clases: {num_classes}")

# Entrenar modelo
model = train_model(train_loader, num_classes)

# Ejemplo de inferencia
test_image_path = os.path.join(TRAIN_DIR, "camiseta_202_jpg.rf.2d8d1bd06cf423018756b2e48db8fdf3"+".jpg")  # Puedes cambiar a cualquier imagen de prueba
inference(model, test_image_path)
```


*Salida:*
```text
Preparando dataset...
loading annotations into memory...
Done (t=1.32s)
creating index...
index created!
Número de clases: 2
Using device: cuda
/usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/usr/local/lib/python3.10/dist-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1`. You can also use `weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to /root/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
100%|██████████| 160M/160M [00:01<00:00, 113MB/s]
Iniciando entrenamiento...
Epoch [1/1], Batch [0], Loss: 0.8499
Epoch [1/1], Batch [10], Loss: 0.2919
Epoch [1/1], Batch [20], Loss: 0.1770
Epoch [1/1], Batch [30], Loss: 0.1502
Epoch [1/1], Batch [40], Loss: 0.2308
Epoch [1/1], Batch [50], Loss: 0.2026
Epoch [1/1], Batch [60], Loss: 0.1794
Epoch [1/1], Batch [70], Loss: 0.1408
Epoch [1/1], Batch [80], Loss: 0.2121
Epoch [1/1], Batch [90], Loss: 0.1269
Epoch [1/1], Batch [100], Loss: 0.1539
Epoch [1/1], Batch [110], Loss: 0.1622
Epoch [1/1], Batch [120], Loss: 0.1465
Epoch [1/1], Batch [130], Loss: 0.1624
Epoch [1/1], Batch [140], Loss: 0.1815
Epoch [1/1], Batch [150], Loss: 0.1946
Epoch [1/1], Batch [160], Loss: 0.1439
Epoch [1/1], Batch [170], Loss: 0.0877
Epoch [1/1], Batch [180], Loss: 0.1682
Epoch [1/1], Batch [190], Loss: 0.1367
Epoch [1/1], Batch [200], Loss: 0.1333
Epoch [1/1], Batch [210], Loss: 0.1426
Epoch [1/1], Batch [220], Loss: 0.2065
Epoch [1/1], Batch [230], Loss: 0.1604
Epoch [1/1], Batch [240], Loss: 0.1899
Epoch [1/1], Batch [250], Loss: 0.0982
Epoch [1/1], Batch [260], Loss: 0.1345
Epoch [1/1], Batch [270], Loss: 0.1368
Epoch [1/1], Average Loss: 0.1655
<Figure size 1200x800 with 1 Axes>
```

# **Guardar modelo**


**[Celda 10 - Código]**
```python
# Guardar modelo entrenado
torch.save(model.state_dict(), 'modelo_logos_clubes.pth')
```

# **Cargar modelo entrenado**


**[Celda 12 - Código]**
```python
#Ruta del modelo
MODEL_PATH = "modelo_logos_clubes.pth"

# Cargar el modelo entrenado
num_classes = 2  # 1 clase de logo + 1 clase de fondo
model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Transformar la imagen al formato esperado por el modelo
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    return image_tensor

def detect_logos(image_path, confidence_threshold=0.5):
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        predictions = model(image_tensor)

    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    filtered_boxes = boxes[scores >= confidence_threshold]
    filtered_labels = labels[scores >= confidence_threshold]

    return filtered_boxes, filtered_labels

# Ejemplo de uso
test_image_path = os.path.join(TRAIN_DIR, "camiseta_202_jpg.rf.2d8d1bd06cf423018756b2e48db8fdf3"+".jpg")
boxes, labels = detect_logos(test_image_path)

# Visualizar las predicciones
image = cv2.imread(test_image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = [int(coord) for coord in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.figure(figsize=(12, 8))
plt.imshow(image)
plt.axis('off')
plt.show()
```


*Salida:*
```text
<ipython-input-17-5e67378d00ac>:7: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(MODEL_PATH))
<Figure size 1200x800 with 1 Axes>
```
