---
aliases: [FasterRCNN_17_10]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-10-31
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/E.C de imágenes a color/Faster RCNN/FasterRCNN_17_10.ipynb"
tamanio_bytes: 597293
---

# Notebook: FasterRCNN_17_10.ipynb

Ruta interna: `GIBD/E.C de imágenes a color/Faster RCNN/FasterRCNN_17_10.ipynb`

---

# **Imports**


**[Celda 2 - Código]**
```python
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

```

# **Cargar datos**

Cargar el modelo


**[Celda 5 - Código]**
```python
# Modelo preentrenado de torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Ajustar el número de clases (por ejemplo, 2 para un solo objeto + background)
num_classes = 2  # Background + 1 clase de interés
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

```


**[Celda 6 - Código]**
```python
import torch
import torch.optim as optim

# Definir el optimizador, usando Adam en este caso
optimizer = optim.Adam(model.parameters(), lr=0.001)  # lr es la tasa de aprendizaje

# Definir el dispositivo: GPU si está disponible, de lo contrario CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Mover el modelo al dispositivo
model.to(device)

```


*Salida:*
```text
FasterRCNN(
  (transform): GeneralizedRCNNTransform(
      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      Resize(min_size=(800,), max_size=1333, mode='bilinear')
  )
  (backbone): BackboneWithFPN(
    (body): IntermediateLayerGetter(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): FrozenBatchNorm2d(64, eps=0.0)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(64, eps=0.0)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(64, eps=0.0)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(256, eps=0.0)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(256, eps=0.0)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(64, eps=0.0)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(64, eps=0.0)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(256, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(64, eps=0.0)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(64, eps=0.0)
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(256, eps=0.0)
          (relu): ReLU(inplace=True)
        )
      )
      (layer2): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(128, eps=0.0)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(128, eps=0.0)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(512, eps=0.0)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): FrozenBatchNorm2d(512, eps=0.0)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(128, eps=0.0)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(128, eps=0.0)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(512, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(128, eps=0.0)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(128, eps=0.0)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(512, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(128, eps=0.0)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(128, eps=0.0)
          (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(512, eps=0.0)
          (relu): ReLU(inplace=True)
        )
      )
      (layer3): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): FrozenBatchNorm2d(1024, eps=0.0)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (3): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (4): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (5): Bottleneck(
          (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(256, eps=0.0)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(256, eps=0.0)
          (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(1024, eps=0.0)
          (relu): ReLU(inplace=True)
        )
      )
      (layer4): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(512, eps=0.0)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(512, eps=0.0)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(2048, eps=0.0)
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): FrozenBatchNorm2d(2048, eps=0.0)
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(512, eps=0.0)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(512, eps=0.0)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(2048, eps=0.0)
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d(512, eps=0.0)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d(512, eps=0.0)
          (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d(2048, eps=0.0)
          (relu): ReLU(inplace=True)
        )
      )
    )
    (fpn): FeaturePyramidNetwork(
      (inner_blocks): ModuleList(
        (0): Conv2dNormActivation(
          (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (2): Conv2dNormActivation(
          (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        )
        (3): Conv2dNormActivation(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
        )
      )
      (layer_blocks): ModuleList(
        (0-3): 4 x Conv2dNormActivation(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
      )
      (extra_blocks): LastLevelMaxPool()
    )
  )
  (rpn): RegionProposalNetwork(
    (anchor_generator): AnchorGenerator()
    (head): RPNHead(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): ReLU(inplace=True)
        )
      )
      (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
      (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (roi_heads): RoIHeads(
    (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)
    (box_head): TwoMLPHead(
      (fc6): Linear(in_features=12544, out_features=1024, bias=True)
      (fc7): Linear(in_features=1024, out_features=1024, bias=True)
    )
    (box_predictor): FastRCNNPredictor(
      (cls_score): Linear(in_features=1024, out_features=2, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=8, bias=True)
    )
  )
)
```

Cargar datos de entrenamiento

Definición del dataset


**[Celda 9 - Código]**
```python
def __getitem__(self, idx):
    # Cargar la imagen
    img_path = self.image_paths[idx]
    img = Image.open(img_path).convert("RGB")

    # Obtener las anotaciones correspondientes
    target_data = self.annotations[idx]

    # Inicializar listas para cajas y etiquetas
    boxes = []
    labels = []

    # Procesar las anotaciones para extraer cajas y etiquetas
    for annotation in target_data:  # target_data es una lista de diccionarios
        bbox = annotation['bbox']  # Obtener la caja [xmin, ymin, width, height]
        # Convertir a [xmin, ymin, xmax, ymax]
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        boxes.append([xmin, ymin, xmax, ymax])

        # Obtener la etiqueta de la categoría
        labels.append(annotation['category_id'])  # Aquí asumimos que category_id es tu etiqueta

    # Convertir listas a tensores
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)

    # Construir el diccionario de anotaciones
    target = {
        "boxes": boxes,
        "labels": labels
    }

    # Aplicar transformaciones (si las hay)
    if self.transforms:
        img = self.transforms(img)

    return img, target

```

# **Entrenamiento del modelo**




DataLoader para manejar el dataset


**[Celda 12 - Código]**
```python
from torchvision.datasets import CocoDetection
from torchvision import transforms as T
from torch.utils.data import DataLoader

# Transformaciones (similar a antes)
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# Ruta a las imágenes y anotaciones
coco_root = '/content/drive/MyDrive/GIBD/E.C de imágenes a color/Faster RCNN/badgesbbfromroboflow/train'
ann_file = '/content/drive/MyDrive/GIBD/E.C de imágenes a color/Faster RCNN/badgesbbfromroboflow/train/_annotations.coco.json'

# Cargar el dataset en formato COCO
train_dataset = CocoDetection(root=coco_root, annFile=ann_file, transform=get_transform(train=True))

# Crear el DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

```


*Salida:*
```text
loading annotations into memory...
Done (t=0.01s)
creating index...
index created!
```

Definición de optimizador y un ciclo de entrenamiento


**[Celda 14 - Código]**
```python
# Mover el modelo a la GPU si es necesario
model.to(device)

num_epochs = 10  # Cambia esto al número de épocas que deseas
for epoch in range(num_epochs):
    model.train()  # Asegúrate de que el modelo esté en modo de entrenamiento
    i = 0  # Reinicia el contador de iteraciones por época

    for images, targets in train_loader:
        # Filtrar imágenes y targets donde no hay anotaciones
        valid_indices = [i for i, t in enumerate(targets[0]) if len(t['bbox']) > 0]
        if not valid_indices:
            continue

        # Filtrar imágenes y targets válidos
        images = [images[i] for i in valid_indices]
        targets = [targets[0][i] for i in valid_indices]

        # Enviar imágenes a la GPU si es necesario
        images = [img.to(device) for img in images]

        # Convertir las anotaciones a tensores y ajustar las cajas a 'x_min, y_min, x_max, y_max'
        converted_targets = []
        for target in targets:
            target_dict = {}
            for k, v in target.items():
                if k == 'bbox':
                    # Convertir 'bbox' de [x, y, width, height] a [x_min, y_min, x_max, y_max]
                    x, y, width, height = v
                    x_min = x
                    y_min = y
                    x_max = x + width
                    y_max = y + height

                    # Solo agregar si el width y height son positivos
                    if width > 0 and height > 0:
                        target_dict['boxes'] = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32).to(device)
                    else:
                        print(f"Found invalid box {v}, skipping this target.")
                        continue
                elif k == 'category_id':
                    # Usar 'category_id' como 'labels'
                    target_dict['labels'] = torch.tensor([v], dtype=torch.int64).to(device)
                elif isinstance(v, list):
                    target_dict[k] = torch.tensor(v, dtype=torch.float32).to(device)
                else:
                    target_dict[k] = torch.tensor([v], dtype=torch.int64).to(device)

            # Solo agregar si contiene una caja válida y una etiqueta
            if 'boxes' in target_dict and 'labels' in target_dict:
                converted_targets.append(target_dict)

        # Si no hay anotaciones válidas, continuar con el siguiente lote
        if not converted_targets:
            continue

        # Adelante y cálculo de la pérdida
        loss_dict = model(images, converted_targets)
        losses = sum(loss for loss in loss_dict.values())

        # Optimización
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Información de progreso
        print(f"Epoch {epoch + 1}, Iteration {i}, Loss: {losses.item()}")
        print(f"Predicted Boxes: {boxes}, Scores: {scores}, Labels: {labels}")
        i += 1

```


*Salida:*
```text
[1;30;43mStreaming output truncated to the last 5000 lines.[0m
Epoch 1, Iteration 250, Loss: 0.5912294983863831
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 251, Loss: 0.16647467017173767
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 252, Loss: 0.33123791217803955
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 253, Loss: 1.181389570236206
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 254, Loss: 0.2908584475517273
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 255, Loss: 0.24952015280723572
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 256, Loss: 0.20154806971549988
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 257, Loss: 0.09582771360874176
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 258, Loss: 0.880028486251831
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 259, Loss: 0.2780608534812927
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 260, Loss: 0.523391604423523
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 261, Loss: 0.1922742873430252
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 262, Loss: 0.2600278854370117
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 263, Loss: 0.1217227503657341
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 264, Loss: 0.27966436743736267
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 265, Loss: 0.40752774477005005
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 266, Loss: 0.13320329785346985
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 267, Loss: 0.24736028909683228
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 268, Loss: 1.2557748556137085
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 269, Loss: 0.1954537183046341
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 270, Loss: 0.12143155932426453
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 271, Loss: 0.2838824391365051
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 272, Loss: 0.09634596109390259
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 273, Loss: 0.3370599150657654
Predicted Boxes: [], Scores: [], Labels: []
Epoch 1, Iteration 274, Loss: 0.16934557259082794
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 0, Loss: 0.09003044664859772
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 1, Loss: 0.07681094110012054
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 2, Loss: 0.38118138909339905
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 3, Loss: 0.3157118558883667
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 4, Loss: 0.21031025052070618
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 5, Loss: 0.21942350268363953
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 6, Loss: 0.19547033309936523
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 7, Loss: 0.21394309401512146
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 8, Loss: 0.07136733084917068
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 9, Loss: 0.13822172582149506
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 10, Loss: 0.20647940039634705
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 11, Loss: 0.1036960557103157
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 12, Loss: 0.11767860502004623
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 13, Loss: 0.5407520532608032
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 14, Loss: 0.14831340312957764
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 15, Loss: 0.1665625423192978
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 16, Loss: 0.12751176953315735
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 17, Loss: 0.10287278145551682
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 18, Loss: 0.13851313292980194
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 19, Loss: 0.6792580485343933
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 20, Loss: 0.22112466394901276
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 21, Loss: 0.4146702289581299
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 22, Loss: 0.6896397471427917
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 23, Loss: 0.12366317212581635
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 24, Loss: 0.2005460113286972
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 25, Loss: 0.1716400682926178
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 26, Loss: 0.1037018820643425
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 27, Loss: 0.1167321428656578
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 28, Loss: 0.07539281994104385
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 29, Loss: 0.14800073206424713
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 30, Loss: 0.12786483764648438
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 31, Loss: 0.2068808227777481
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 32, Loss: 0.10382531583309174
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 33, Loss: 0.19570617377758026
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 34, Loss: 0.309696763753891
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 35, Loss: 0.0838174894452095
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 36, Loss: 0.1998886615037918
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 37, Loss: 0.07843415439128876
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 38, Loss: 0.14765290915966034
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 39, Loss: 0.13264745473861694
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 40, Loss: 0.23054558038711548
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 41, Loss: 0.10929201543331146
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 42, Loss: 0.16781195998191833
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 43, Loss: 0.44275009632110596
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 44, Loss: 0.11354011297225952
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 45, Loss: 0.2834570109844208
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 46, Loss: 0.1214873343706131
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 47, Loss: 0.20131777226924896
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 48, Loss: 0.200797438621521
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 49, Loss: 0.13072113692760468
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 50, Loss: 0.13948959112167358
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 51, Loss: 0.10429373383522034
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 52, Loss: 0.1653185486793518
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 53, Loss: 0.13937389850616455
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 54, Loss: 0.24679596722126007
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 55, Loss: 0.2674618363380432
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 56, Loss: 0.20041730999946594
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 57, Loss: 0.19999542832374573
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 58, Loss: 0.12240880727767944
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 59, Loss: 0.189367413520813
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 60, Loss: 0.13763831555843353
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 61, Loss: 0.18902401626110077
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 62, Loss: 0.2791147530078888
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 63, Loss: 0.1661219596862793
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 64, Loss: 0.3782444894313812
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 65, Loss: 0.1090388149023056
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 66, Loss: 0.1754712164402008
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 67, Loss: 0.5822730660438538
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 68, Loss: 0.2943859398365021
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 69, Loss: 0.2988874614238739
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 70, Loss: 0.12844762206077576
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 71, Loss: 0.2512928545475006
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 72, Loss: 0.17102795839309692
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 73, Loss: 0.13356156647205353
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 74, Loss: 0.193386048078537
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 75, Loss: 0.06816964596509933
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 76, Loss: 0.08193416148424149
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 77, Loss: 0.17939528822898865
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 78, Loss: 0.18291209638118744
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 79, Loss: 0.13866232335567474
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 80, Loss: 0.2664664387702942
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 81, Loss: 0.300713449716568
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 82, Loss: 0.2643831670284271
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 83, Loss: 0.1660410612821579
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 84, Loss: 0.21500927209854126
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 85, Loss: 0.16380088031291962
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 86, Loss: 0.131173238158226
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 87, Loss: 0.3041216731071472
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 88, Loss: 0.27034807205200195
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 89, Loss: 0.23343788087368011
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 90, Loss: 0.4850154221057892
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 91, Loss: 0.19330355525016785
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 92, Loss: 0.22357462346553802
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 93, Loss: 0.4697123169898987
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 94, Loss: 0.18709899485111237
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 95, Loss: 0.14524585008621216
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 96, Loss: 0.39090496301651
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 97, Loss: 0.12128221243619919
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 98, Loss: 0.09023363143205643
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 99, Loss: 0.1347673237323761
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 100, Loss: 0.08332698792219162
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 101, Loss: 0.11582410335540771
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 102, Loss: 0.0930802971124649
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 103, Loss: 0.14644597470760345
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 104, Loss: 0.12156480550765991
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 105, Loss: 0.10806793719530106
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 106, Loss: 0.12341121584177017
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 107, Loss: 0.059781309217214584
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 108, Loss: 0.1919063925743103
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 109, Loss: 0.13141971826553345
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 110, Loss: 0.1158628761768341
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 111, Loss: 0.10250157862901688
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 112, Loss: 0.2579650580883026
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 113, Loss: 0.11656050384044647
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 114, Loss: 0.16226783394813538
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 115, Loss: 0.0958397388458252
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 116, Loss: 0.11223824322223663
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 117, Loss: 0.11221165210008621
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 118, Loss: 0.11163291335105896
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 119, Loss: 0.24013358354568481
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 120, Loss: 0.09922141581773758
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 121, Loss: 0.2451971471309662
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 122, Loss: 0.35766711831092834
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 123, Loss: 0.08599758893251419
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 124, Loss: 0.10648743063211441
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 125, Loss: 0.19223414361476898
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 126, Loss: 0.08149092644453049
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 127, Loss: 0.18530665338039398
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 128, Loss: 0.12142468988895416
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 129, Loss: 0.12163008004426956
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 130, Loss: 0.14788831770420074
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 131, Loss: 0.2143402248620987
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 132, Loss: 0.3162080645561218
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 133, Loss: 0.0978502631187439
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 134, Loss: 0.1792590320110321
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 135, Loss: 0.17700029909610748
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 136, Loss: 0.09713779389858246
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 137, Loss: 0.23222166299819946
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 138, Loss: 0.15040189027786255
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 139, Loss: 0.20990607142448425
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 140, Loss: 0.11549320071935654
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 141, Loss: 0.1837005615234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 142, Loss: 0.16058313846588135
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 143, Loss: 0.10331792384386063
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 144, Loss: 0.18630743026733398
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 145, Loss: 0.13404443860054016
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 146, Loss: 0.150015190243721
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 147, Loss: 0.2185690551996231
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 148, Loss: 0.14010019600391388
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 149, Loss: 0.18029405176639557
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 150, Loss: 0.16552282869815826
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 151, Loss: 0.2179844081401825
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 152, Loss: 0.1837017983198166
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 153, Loss: 0.6197332143783569
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 154, Loss: 0.20490671694278717
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 155, Loss: 0.3511676788330078
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 156, Loss: 0.26664939522743225
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 157, Loss: 0.20535659790039062
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 158, Loss: 0.5129477381706238
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 159, Loss: 0.21198002994060516
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 160, Loss: 0.4190743565559387
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 161, Loss: 1.1770133972167969
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 162, Loss: 0.255778044462204
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 163, Loss: 0.15882933139801025
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 164, Loss: 0.26990216970443726
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 165, Loss: 0.22215451300144196
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 166, Loss: 0.18968524038791656
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 167, Loss: 0.11085379123687744
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 168, Loss: 0.12312103807926178
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 169, Loss: 0.18377450108528137
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 170, Loss: 0.1005585640668869
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 171, Loss: 0.07157283276319504
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 172, Loss: 1.3628991842269897
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 173, Loss: 0.153034508228302
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 174, Loss: 0.14763572812080383
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 175, Loss: 0.1933422088623047
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 176, Loss: 0.15377849340438843
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 177, Loss: 0.27639666199684143
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 178, Loss: 0.05510135740041733
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 179, Loss: 0.21018466353416443
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 180, Loss: 0.23321813344955444
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 181, Loss: 0.08095212280750275
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 182, Loss: 0.12259317189455032
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 183, Loss: 0.1716054081916809
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 184, Loss: 0.2255733758211136
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 185, Loss: 0.1502504050731659
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 186, Loss: 0.3151721656322479
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 187, Loss: 0.24120870232582092
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 188, Loss: 0.35396984219551086
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 189, Loss: 0.24131903052330017
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 190, Loss: 0.15245823562145233
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 191, Loss: 0.42221421003341675
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 192, Loss: 0.186086043715477
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 193, Loss: 0.308569073677063
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 194, Loss: 0.1768559068441391
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 195, Loss: 0.18971413373947144
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 196, Loss: 0.2237974852323532
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 197, Loss: 0.24083392322063446
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 198, Loss: 0.14104777574539185
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 199, Loss: 0.13418331742286682
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 200, Loss: 0.13824297487735748
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 201, Loss: 0.18489909172058105
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 202, Loss: 0.11087046563625336
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 203, Loss: 0.0789063349366188
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 204, Loss: 0.3860799968242645
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 205, Loss: 0.12553894519805908
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 206, Loss: 0.13121111690998077
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 207, Loss: 0.09894271194934845
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 208, Loss: 0.12270195782184601
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 209, Loss: 0.3107964098453522
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 210, Loss: 0.1757889688014984
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 211, Loss: 0.11275999248027802
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 212, Loss: 0.16073007881641388
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 213, Loss: 0.0963858962059021
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 214, Loss: 0.05667092651128769
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 215, Loss: 0.1821247935295105
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 216, Loss: 0.14354385435581207
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 217, Loss: 0.34598982334136963
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 218, Loss: 0.18615193665027618
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 219, Loss: 0.25558772683143616
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 220, Loss: 0.18262796103954315
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 221, Loss: 0.15497204661369324
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 222, Loss: 0.15447287261486053
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 223, Loss: 0.19206884503364563
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 224, Loss: 0.17813454568386078
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 225, Loss: 0.4290584623813629
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 226, Loss: 0.2673443555831909
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 227, Loss: 0.26552367210388184
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 228, Loss: 0.1506718546152115
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 229, Loss: 0.09526693820953369
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 230, Loss: 0.17896036803722382
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 231, Loss: 0.14591988921165466
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 232, Loss: 0.21940664947032928
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 233, Loss: 0.08187632262706757
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 234, Loss: 0.2057371288537979
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 235, Loss: 0.17749148607254028
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 236, Loss: 0.0882621705532074
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 237, Loss: 0.2535629868507385
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 238, Loss: 0.23482434451580048
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 239, Loss: 0.15810321271419525
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 240, Loss: 0.19936010241508484
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 241, Loss: 0.11090242862701416
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 242, Loss: 0.09738980233669281
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 243, Loss: 0.24427594244480133
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 244, Loss: 0.2836363613605499
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 245, Loss: 0.17927154898643494
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 246, Loss: 0.20629696547985077
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 247, Loss: 0.15401621162891388
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 248, Loss: 0.07155660539865494
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 249, Loss: 0.1909233182668686
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 250, Loss: 0.2210106998682022
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 251, Loss: 0.21603788435459137
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 252, Loss: 0.16589567065238953
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 253, Loss: 0.19414730370044708
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 254, Loss: 0.08806444704532623
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 255, Loss: 0.207907572388649
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 256, Loss: 0.2292029708623886
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 257, Loss: 0.11539973318576813
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 258, Loss: 0.2920462489128113
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 259, Loss: 0.09793657064437866
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 260, Loss: 0.16307146847248077
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 261, Loss: 0.20382338762283325
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 262, Loss: 0.19806146621704102
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 263, Loss: 0.15614645183086395
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 264, Loss: 0.1888977289199829
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 265, Loss: 0.13648496568202972
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 266, Loss: 0.15255697071552277
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 267, Loss: 0.12028080970048904
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 268, Loss: 0.13738158345222473
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 269, Loss: 0.20778432488441467
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 270, Loss: 0.17176398634910583
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 271, Loss: 0.32545989751815796
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 272, Loss: 0.07378134876489639
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 273, Loss: 0.16326498985290527
Predicted Boxes: [], Scores: [], Labels: []
Epoch 2, Iteration 274, Loss: 0.2379404604434967
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 0, Loss: 0.08129706233739853
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 1, Loss: 0.13455995917320251
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 2, Loss: 0.13819554448127747
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 3, Loss: 0.20892897248268127
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 4, Loss: 0.12847621738910675
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 5, Loss: 0.2595106065273285
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 6, Loss: 0.21743351221084595
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 7, Loss: 0.16281695663928986
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 8, Loss: 0.19093607366085052
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 9, Loss: 0.23144087195396423
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 10, Loss: 0.1360597312450409
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 11, Loss: 0.26819732785224915
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 12, Loss: 0.18463751673698425
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 13, Loss: 0.22934038937091827
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 14, Loss: 0.15601062774658203
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 15, Loss: 0.3237271010875702
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 16, Loss: 0.2744249403476715
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 17, Loss: 0.1686842441558838
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 18, Loss: 0.17097175121307373
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 19, Loss: 0.17604976892471313
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 20, Loss: 0.2165835201740265
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 21, Loss: 0.22816547751426697
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 22, Loss: 0.1100146472454071
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 23, Loss: 0.27996522188186646
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 24, Loss: 0.19324630498886108
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 25, Loss: 0.2539442777633667
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 26, Loss: 0.15383903682231903
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 27, Loss: 0.1743122935295105
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 28, Loss: 0.23761358857154846
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 29, Loss: 0.16689977049827576
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 30, Loss: 0.1838196963071823
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 31, Loss: 0.32006385922431946
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 32, Loss: 0.16049692034721375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 33, Loss: 0.21128351986408234
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 34, Loss: 0.3607594072818756
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 35, Loss: 0.17520400881767273
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 36, Loss: 0.12370742112398148
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 37, Loss: 0.22748661041259766
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 38, Loss: 0.09597526490688324
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 39, Loss: 0.29715457558631897
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 40, Loss: 0.15800841152668
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 41, Loss: 0.07516586780548096
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 42, Loss: 0.08073962479829788
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 43, Loss: 0.28391867876052856
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 44, Loss: 0.25365883111953735
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 45, Loss: 0.4065263271331787
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 46, Loss: 0.18716096878051758
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 47, Loss: 0.452924907207489
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 48, Loss: 0.08946844935417175
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 49, Loss: 0.1835838109254837
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 50, Loss: 0.13786989450454712
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 51, Loss: 0.18355253338813782
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 52, Loss: 0.16388899087905884
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 53, Loss: 0.134588360786438
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 54, Loss: 0.20566461980342865
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 55, Loss: 0.5636809468269348
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 56, Loss: 0.24903132021427155
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 57, Loss: 0.19111427664756775
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 58, Loss: 0.14503195881843567
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 59, Loss: 0.12796537578105927
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 60, Loss: 0.14094245433807373
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 61, Loss: 0.18876776099205017
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 62, Loss: 0.26816314458847046
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 63, Loss: 0.12541569769382477
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 64, Loss: 0.09801747649908066
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 65, Loss: 0.24775545299053192
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 66, Loss: 0.07025802135467529
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 67, Loss: 0.27317580580711365
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 68, Loss: 0.25492095947265625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 69, Loss: 0.17406824231147766
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 70, Loss: 0.23919151723384857
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 71, Loss: 0.13068252801895142
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 72, Loss: 0.2264876663684845
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 73, Loss: 0.16675350069999695
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 74, Loss: 0.09322848170995712
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 75, Loss: 0.1420537382364273
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 76, Loss: 0.11702579259872437
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 77, Loss: 0.1068924143910408
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 78, Loss: 0.19934995472431183
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 79, Loss: 0.12735030055046082
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 80, Loss: 0.24392342567443848
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 81, Loss: 0.3451439440250397
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 82, Loss: 0.14683565497398376
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 83, Loss: 0.18678446114063263
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 84, Loss: 0.08506662398576736
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 85, Loss: 0.14217375218868256
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 86, Loss: 0.14014728367328644
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 87, Loss: 0.19880089163780212
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 88, Loss: 0.18985722959041595
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 89, Loss: 0.14383913576602936
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 90, Loss: 0.16101711988449097
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 91, Loss: 0.1801629364490509
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 92, Loss: 0.16040754318237305
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 93, Loss: 0.2069212645292282
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 94, Loss: 0.16264665126800537
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 95, Loss: 0.13176003098487854
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 96, Loss: 0.10542900860309601
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 97, Loss: 0.12684732675552368
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 98, Loss: 0.10356474667787552
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 99, Loss: 0.2472730129957199
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 100, Loss: 0.09602239727973938
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 101, Loss: 0.21113695204257965
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 102, Loss: 0.3158017098903656
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 103, Loss: 0.1942102462053299
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 104, Loss: 0.19194376468658447
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 105, Loss: 0.09225451201200485
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 106, Loss: 0.2531476616859436
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 107, Loss: 0.13476872444152832
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 108, Loss: 0.2663048803806305
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 109, Loss: 0.17156822979450226
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 110, Loss: 0.3125159442424774
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 111, Loss: 0.2547474503517151
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 112, Loss: 0.15731260180473328
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 113, Loss: 0.30861449241638184
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 114, Loss: 0.16106577217578888
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 115, Loss: 0.2789629101753235
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 116, Loss: 0.22259144484996796
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 117, Loss: 0.1839749813079834
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 118, Loss: 0.2078407257795334
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 119, Loss: 0.2112233191728592
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 120, Loss: 0.16865578293800354
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 121, Loss: 0.25590547919273376
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 122, Loss: 0.39548543095588684
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 123, Loss: 0.2666182816028595
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 124, Loss: 0.2981121838092804
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 125, Loss: 0.18060822784900665
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 126, Loss: 0.20026642084121704
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 127, Loss: 0.17847108840942383
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 128, Loss: 0.18395406007766724
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 129, Loss: 0.1046600416302681
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 130, Loss: 0.40444180369377136
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 131, Loss: 0.06609829515218735
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 132, Loss: 0.4432705342769623
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 133, Loss: 0.3301166594028473
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 134, Loss: 0.22644951939582825
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 135, Loss: 0.21406111121177673
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 136, Loss: 0.2635217607021332
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 137, Loss: 0.17217427492141724
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 138, Loss: 0.33579638600349426
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 139, Loss: 0.18231964111328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 140, Loss: 0.08328081667423248
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 141, Loss: 0.22025300562381744
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 142, Loss: 0.43646371364593506
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 143, Loss: 0.1885661482810974
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 144, Loss: 0.11380618065595627
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 145, Loss: 0.19724029302597046
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 146, Loss: 0.1647217869758606
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 147, Loss: 0.35085296630859375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 148, Loss: 0.1459520310163498
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 149, Loss: 0.22109779715538025
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 150, Loss: 0.1225055679678917
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 151, Loss: 0.11595281958580017
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 152, Loss: 0.23635223507881165
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 153, Loss: 0.27286776900291443
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 154, Loss: 0.07648824155330658
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 155, Loss: 0.16016770899295807
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 156, Loss: 0.09536626189947128
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 157, Loss: 0.12255091220140457
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 158, Loss: 0.2133634239435196
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 159, Loss: 0.40420007705688477
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 160, Loss: 0.17558971047401428
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 161, Loss: 0.23573756217956543
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 162, Loss: 0.2942153811454773
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 163, Loss: 0.131160169839859
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 164, Loss: 0.16797415912151337
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 165, Loss: 0.18453259766101837
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 166, Loss: 0.22786469757556915
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 167, Loss: 0.07262016832828522
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 168, Loss: 0.23201262950897217
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 169, Loss: 0.23029792308807373
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 170, Loss: 0.21153303980827332
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 171, Loss: 0.10310438275337219
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 172, Loss: 0.10154096782207489
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 173, Loss: 0.15909287333488464
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 174, Loss: 0.06857915967702866
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 175, Loss: 0.18845096230506897
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 176, Loss: 0.11727911978960037
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 177, Loss: 0.1278335601091385
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 178, Loss: 0.31300118565559387
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 179, Loss: 0.10151290893554688
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 180, Loss: 0.12194506824016571
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 181, Loss: 0.13746237754821777
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 182, Loss: 0.15663741528987885
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 183, Loss: 0.30737966299057007
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 184, Loss: 0.1971684694290161
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 185, Loss: 0.1848553866147995
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 186, Loss: 0.14766453206539154
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 187, Loss: 0.11953345686197281
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 188, Loss: 0.1743263453245163
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 189, Loss: 0.03675287961959839
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 190, Loss: 0.22694221138954163
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 191, Loss: 0.27183791995048523
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 192, Loss: 0.12995383143424988
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 193, Loss: 0.11917103081941605
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 194, Loss: 0.20224297046661377
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 195, Loss: 0.24829533696174622
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 196, Loss: 0.18132194876670837
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 197, Loss: 0.12251542508602142
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 198, Loss: 0.17266163229942322
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 199, Loss: 0.2282451093196869
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 200, Loss: 0.1815204620361328
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 201, Loss: 0.23919811844825745
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 202, Loss: 0.28680455684661865
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 203, Loss: 0.33574914932250977
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 204, Loss: 0.512514591217041
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 205, Loss: 0.19841919839382172
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 206, Loss: 0.226253941655159
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 207, Loss: 0.2019294947385788
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 208, Loss: 0.09689725190401077
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 209, Loss: 0.28373587131500244
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 210, Loss: 0.22420866787433624
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 211, Loss: 0.20246738195419312
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 212, Loss: 0.2805050015449524
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 213, Loss: 0.19889616966247559
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 214, Loss: 0.3526366651058197
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 215, Loss: 0.1883569061756134
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 216, Loss: 0.2119266539812088
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 217, Loss: 0.27907395362854004
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 218, Loss: 0.1717146337032318
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 219, Loss: 0.21237321197986603
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 220, Loss: 0.13086847960948944
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 221, Loss: 0.28492647409439087
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 222, Loss: 0.11264491826295853
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 223, Loss: 0.16245166957378387
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 224, Loss: 0.13384348154067993
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 225, Loss: 0.19940827786922455
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 226, Loss: 0.1813875287771225
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 227, Loss: 0.08264709264039993
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 228, Loss: 0.141615092754364
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 229, Loss: 0.34415972232818604
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 230, Loss: 0.14307832717895508
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 231, Loss: 0.2888716161251068
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 232, Loss: 0.19205857813358307
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 233, Loss: 0.16593223810195923
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 234, Loss: 0.2284441590309143
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 235, Loss: 0.1499321609735489
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 236, Loss: 0.2603832185268402
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 237, Loss: 0.3591800332069397
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 238, Loss: 0.08004249632358551
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 239, Loss: 1.2347180843353271
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 240, Loss: 0.32689836621284485
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 241, Loss: 0.17196954786777496
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 242, Loss: 0.15534360706806183
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 243, Loss: 0.1564735472202301
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 244, Loss: 0.2585289180278778
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 245, Loss: 0.14693057537078857
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 246, Loss: 0.1080814003944397
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 247, Loss: 0.21335570514202118
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 248, Loss: 0.20977146923542023
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 249, Loss: 0.19425953924655914
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 250, Loss: 0.37305834889411926
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 251, Loss: 0.1938984990119934
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 252, Loss: 0.25540629029273987
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 253, Loss: 0.2751501202583313
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 254, Loss: 0.19694997370243073
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 255, Loss: 0.3429907560348511
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 256, Loss: 0.15015481412410736
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 257, Loss: 0.05355893820524216
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 258, Loss: 0.07637064903974533
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 259, Loss: 0.19685660302639008
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 260, Loss: 0.16380415856838226
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 261, Loss: 0.20088863372802734
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 262, Loss: 0.25868117809295654
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 263, Loss: 0.21851761639118195
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 264, Loss: 0.2331957072019577
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 265, Loss: 0.23029784858226776
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 266, Loss: 0.3455890417098999
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 267, Loss: 0.2246023714542389
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 268, Loss: 0.2714760899543762
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 269, Loss: 0.20106543600559235
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 270, Loss: 0.17756474018096924
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 271, Loss: 0.1599191427230835
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 272, Loss: 0.26402971148490906
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 273, Loss: 0.17776009440422058
Predicted Boxes: [], Scores: [], Labels: []
Epoch 3, Iteration 274, Loss: 0.23894862830638885
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 0, Loss: 0.10580145567655563
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 1, Loss: 0.1960185170173645
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 2, Loss: 0.16850583255290985
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 3, Loss: 0.15474900603294373
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 4, Loss: 0.1816716194152832
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 5, Loss: 0.12670056521892548
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 6, Loss: 0.14749838411808014
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 7, Loss: 0.1710231453180313
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 8, Loss: 0.1995408982038498
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 9, Loss: 0.19645120203495026
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 10, Loss: 0.2191539704799652
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 11, Loss: 0.1329846978187561
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 12, Loss: 0.20342713594436646
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 13, Loss: 0.20929615199565887
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 14, Loss: 0.18598739802837372
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 15, Loss: 0.23310674726963043
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 16, Loss: 0.21212175488471985
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 17, Loss: 0.12381558120250702
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 18, Loss: 0.35136210918426514
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 19, Loss: 0.22820338606834412
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 20, Loss: 0.23001767694950104
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 21, Loss: 0.22484798729419708
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 22, Loss: 0.17381927371025085
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 23, Loss: 0.08839242905378342
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 24, Loss: 0.12085407972335815
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 25, Loss: 0.1341174989938736
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 26, Loss: 0.3045566976070404
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 27, Loss: 0.25600314140319824
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 28, Loss: 0.07757624238729477
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 29, Loss: 0.16727164387702942
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 30, Loss: 0.21032695472240448
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 31, Loss: 0.16389508545398712
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 32, Loss: 0.22018395364284515
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 33, Loss: 0.2776917517185211
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 34, Loss: 0.28484657406806946
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 35, Loss: 0.23965482413768768
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 36, Loss: 0.25143730640411377
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 37, Loss: 0.34629449248313904
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 38, Loss: 0.2549391984939575
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 39, Loss: 0.2445080578327179
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 40, Loss: 0.2920982241630554
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 41, Loss: 0.1774122416973114
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 42, Loss: 0.2145063877105713
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 43, Loss: 0.2752154469490051
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 44, Loss: 0.2498275190591812
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 45, Loss: 0.21263936161994934
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 46, Loss: 0.38688552379608154
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 47, Loss: 0.14664359390735626
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 48, Loss: 0.17893916368484497
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 49, Loss: 0.17046859860420227
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 50, Loss: 0.24646130204200745
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 51, Loss: 0.18666039407253265
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 52, Loss: 0.10485473275184631
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 53, Loss: 0.27071988582611084
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 54, Loss: 0.3148640990257263
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 55, Loss: 0.20425917208194733
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 56, Loss: 0.15470118820667267
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 57, Loss: 0.21657289564609528
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 58, Loss: 0.2218964397907257
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 59, Loss: 0.13321392238140106
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 60, Loss: 0.19884099066257477
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 61, Loss: 0.2810921370983124
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 62, Loss: 0.07462595403194427
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 63, Loss: 0.09311290830373764
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 64, Loss: 0.061085958033800125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 65, Loss: 0.10610891133546829
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 66, Loss: 0.10590794682502747
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 67, Loss: 0.158109650015831
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 68, Loss: 0.2557467818260193
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 69, Loss: 0.33856526017189026
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 70, Loss: 0.15716663002967834
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 71, Loss: 0.20044977962970734
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 72, Loss: 0.13684989511966705
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 73, Loss: 0.08693544566631317
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 74, Loss: 0.31466877460479736
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 75, Loss: 0.25166457891464233
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 76, Loss: 0.29257819056510925
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 77, Loss: 0.255258709192276
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 78, Loss: 0.12750060856342316
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 79, Loss: 0.21959730982780457
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 80, Loss: 0.16336895525455475
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 81, Loss: 0.17217595875263214
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 82, Loss: 0.24401405453681946
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 83, Loss: 0.1485358327627182
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 84, Loss: 0.30304452776908875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 85, Loss: 0.35599565505981445
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 86, Loss: 0.1345629245042801
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 87, Loss: 0.23662036657333374
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 88, Loss: 0.16555704176425934
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 89, Loss: 0.2587931752204895
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 90, Loss: 0.2918024957180023
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 91, Loss: 0.08052840083837509
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 92, Loss: 0.21877649426460266
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 93, Loss: 0.09413374215364456
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 94, Loss: 0.17500032484531403
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 95, Loss: 0.10237927734851837
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 96, Loss: 0.12498923391103745
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 97, Loss: 0.2587570548057556
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 98, Loss: 0.2622935175895691
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 99, Loss: 0.17808954417705536
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 100, Loss: 0.26532235741615295
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 101, Loss: 0.15493199229240417
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 102, Loss: 0.2152481973171234
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 103, Loss: 0.098390594124794
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 104, Loss: 0.26346778869628906
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 105, Loss: 0.14175468683242798
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 106, Loss: 0.21207547187805176
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 107, Loss: 0.23143118619918823
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 108, Loss: 0.12153097987174988
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 109, Loss: 0.21164637804031372
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 110, Loss: 0.24650715291500092
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 111, Loss: 0.17758163809776306
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 112, Loss: 0.17165657877922058
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 113, Loss: 0.15998496115207672
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 114, Loss: 0.144492045044899
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 115, Loss: 0.2776159346103668
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 116, Loss: 0.24925009906291962
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 117, Loss: 0.09506765753030777
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 118, Loss: 0.31479591131210327
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 119, Loss: 0.1944267302751541
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 120, Loss: 0.24637404084205627
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 121, Loss: 0.2664654850959778
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 122, Loss: 0.43053361773490906
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 123, Loss: 0.1764543354511261
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 124, Loss: 0.26885199546813965
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 125, Loss: 0.2591004967689514
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 126, Loss: 0.3829076588153839
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 127, Loss: 0.13599079847335815
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 128, Loss: 0.31027859449386597
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 129, Loss: 0.17902185022830963
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 130, Loss: 0.12324162572622299
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 131, Loss: 0.173544242978096
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 132, Loss: 0.27600711584091187
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 133, Loss: 0.1429596096277237
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 134, Loss: 0.14914369583129883
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 135, Loss: 0.19629326462745667
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 136, Loss: 0.20328161120414734
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 137, Loss: 0.2589090168476105
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 138, Loss: 0.17631852626800537
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 139, Loss: 0.21561013162136078
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 140, Loss: 0.08597651869058609
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 141, Loss: 0.2796359956264496
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 142, Loss: 0.19901107251644135
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 143, Loss: 0.16432222723960876
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 144, Loss: 0.15112000703811646
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 145, Loss: 0.28987783193588257
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 146, Loss: 0.131726935505867
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 147, Loss: 0.10267013311386108
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 148, Loss: 0.23435170948505402
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 149, Loss: 0.30217933654785156
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 150, Loss: 0.20837371051311493
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 151, Loss: 0.20060208439826965
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 152, Loss: 0.21433734893798828
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 153, Loss: 0.21993738412857056
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 154, Loss: 0.21668049693107605
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 155, Loss: 0.17225967347621918
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 156, Loss: 0.15215958654880524
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 157, Loss: 0.1817464381456375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 158, Loss: 0.4341338574886322
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 159, Loss: 0.2301798164844513
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 160, Loss: 0.10141885280609131
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 161, Loss: 0.23421132564544678
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 162, Loss: 0.18495208024978638
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 163, Loss: 0.31598496437072754
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 164, Loss: 0.20112727582454681
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 165, Loss: 0.18770156800746918
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 166, Loss: 0.12454865127801895
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 167, Loss: 0.17078596353530884
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 168, Loss: 0.17689964175224304
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 169, Loss: 0.29818543791770935
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 170, Loss: 0.2916128933429718
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 171, Loss: 0.2633873522281647
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 172, Loss: 0.2550690472126007
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 173, Loss: 0.173359215259552
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 174, Loss: 0.25602900981903076
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 175, Loss: 0.3177083730697632
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 176, Loss: 0.2291429042816162
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 177, Loss: 0.1629294604063034
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 178, Loss: 0.42079803347587585
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 179, Loss: 0.13734860718250275
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 180, Loss: 0.2669214904308319
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 181, Loss: 0.1501660794019699
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 182, Loss: 0.16589222848415375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 183, Loss: 0.24164314568042755
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 184, Loss: 0.1551923304796219
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 185, Loss: 0.1368432641029358
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 186, Loss: 0.09901130199432373
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 187, Loss: 0.15386469662189484
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 188, Loss: 0.12787434458732605
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 189, Loss: 0.18145877122879028
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 190, Loss: 0.2492409497499466
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 191, Loss: 0.12753410637378693
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 192, Loss: 0.21123863756656647
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 193, Loss: 0.3194179832935333
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 194, Loss: 0.13719934225082397
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 195, Loss: 0.31450793147087097
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 196, Loss: 0.188667431473732
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 197, Loss: 0.39415547251701355
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 198, Loss: 0.24396951496601105
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 199, Loss: 0.23874539136886597
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 200, Loss: 0.30372172594070435
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 201, Loss: 0.149383544921875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 202, Loss: 0.15393158793449402
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 203, Loss: 0.1836075484752655
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 204, Loss: 0.26756247878074646
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 205, Loss: 0.23806649446487427
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 206, Loss: 0.24055904150009155
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 207, Loss: 0.13063225150108337
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 208, Loss: 0.3041047751903534
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 209, Loss: 0.2247193306684494
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 210, Loss: 0.1621367186307907
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 211, Loss: 0.27651816606521606
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 212, Loss: 0.226270392537117
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 213, Loss: 0.12329836189746857
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 214, Loss: 0.18368679285049438
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 215, Loss: 0.16927281022071838
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 216, Loss: 0.1746884137392044
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 217, Loss: 0.10077209025621414
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 218, Loss: 0.1677185744047165
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 219, Loss: 0.1574753075838089
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 220, Loss: 0.19331622123718262
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 221, Loss: 0.08909078687429428
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 222, Loss: 0.19154228270053864
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 223, Loss: 0.10481325536966324
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 224, Loss: 0.18638548254966736
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 225, Loss: 0.13193538784980774
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 226, Loss: 0.1404198259115219
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 227, Loss: 0.17443671822547913
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 228, Loss: 0.35189908742904663
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 229, Loss: 0.09329643845558167
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 230, Loss: 0.2626732587814331
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 231, Loss: 0.14140187203884125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 232, Loss: 0.17813417315483093
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 233, Loss: 0.2830847501754761
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 234, Loss: 0.11079110205173492
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 235, Loss: 0.22073908150196075
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 236, Loss: 0.09271930903196335
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 237, Loss: 0.1728164255619049
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 238, Loss: 0.2235514521598816
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 239, Loss: 0.21440808475017548
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 240, Loss: 0.15162882208824158
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 241, Loss: 0.27596330642700195
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 242, Loss: 0.11099208891391754
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 243, Loss: 0.15948782861232758
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 244, Loss: 0.17781667411327362
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 245, Loss: 0.21070712804794312
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 246, Loss: 0.093659907579422
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 247, Loss: 0.16493654251098633
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 248, Loss: 0.07587854564189911
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 249, Loss: 0.24307996034622192
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 250, Loss: 0.13572999835014343
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 251, Loss: 0.15170161426067352
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 252, Loss: 0.2032965123653412
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 253, Loss: 0.2654682397842407
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 254, Loss: 0.26375818252563477
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 255, Loss: 0.12030167877674103
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 256, Loss: 0.13547568023204803
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 257, Loss: 0.2410232275724411
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 258, Loss: 0.15144625306129456
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 259, Loss: 0.12361188977956772
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 260, Loss: 0.19349882006645203
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 261, Loss: 0.16823245584964752
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 262, Loss: 0.19504386186599731
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 263, Loss: 0.19407014548778534
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 264, Loss: 0.1510901153087616
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 265, Loss: 0.15353654325008392
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 266, Loss: 0.16073673963546753
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 267, Loss: 0.38276565074920654
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 268, Loss: 0.19025231897830963
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 269, Loss: 0.06800482422113419
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 270, Loss: 0.33117592334747314
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 271, Loss: 0.1512453854084015
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 272, Loss: 0.18001796305179596
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 273, Loss: 0.2340307980775833
Predicted Boxes: [], Scores: [], Labels: []
Epoch 4, Iteration 274, Loss: 0.12862072885036469
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 0, Loss: 0.17472746968269348
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 1, Loss: 0.18318405747413635
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 2, Loss: 0.14153079688549042
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 3, Loss: 0.22813192009925842
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 4, Loss: 0.22974975407123566
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 5, Loss: 0.3084150552749634
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 6, Loss: 0.16961795091629028
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 7, Loss: 0.1132114976644516
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 8, Loss: 0.2635606825351715
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 9, Loss: 0.18119001388549805
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 10, Loss: 0.20308195054531097
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 11, Loss: 0.2961651682853699
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 12, Loss: 0.0871291533112526
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 13, Loss: 0.10506998747587204
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 14, Loss: 0.12226108461618423
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 15, Loss: 0.19883139431476593
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 16, Loss: 0.23579002916812897
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 17, Loss: 0.18227700889110565
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 18, Loss: 0.10597681999206543
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 19, Loss: 0.163978710770607
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 20, Loss: 0.15722478926181793
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 21, Loss: 0.2450033575296402
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 22, Loss: 0.3330157995223999
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 23, Loss: 0.1634739488363266
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 24, Loss: 0.2486392855644226
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 25, Loss: 0.17219501733779907
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 26, Loss: 0.1486690491437912
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 27, Loss: 0.20807142555713654
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 28, Loss: 0.24389156699180603
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 29, Loss: 0.08006871491670609
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 30, Loss: 0.19582484662532806
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 31, Loss: 0.22273431718349457
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 32, Loss: 0.3632083833217621
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 33, Loss: 0.12195689976215363
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 34, Loss: 0.11217208951711655
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 35, Loss: 0.1476626694202423
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 36, Loss: 0.1687525063753128
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 37, Loss: 0.22074803709983826
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 38, Loss: 0.2702460289001465
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 39, Loss: 0.22450274229049683
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 40, Loss: 0.06825767457485199
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 41, Loss: 0.36987173557281494
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 42, Loss: 0.14685606956481934
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 43, Loss: 0.14513446390628815
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 44, Loss: 0.18078088760375977
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 45, Loss: 0.16497372090816498
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 46, Loss: 0.21857839822769165
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 47, Loss: 0.16245585680007935
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 48, Loss: 0.10287411510944366
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 49, Loss: 0.21065795421600342
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 50, Loss: 0.2696193754673004
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 51, Loss: 0.145252525806427
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 52, Loss: 0.15521371364593506
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 53, Loss: 0.17793026566505432
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 54, Loss: 0.05946500599384308
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 55, Loss: 0.19941648840904236
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 56, Loss: 0.13970588147640228
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 57, Loss: 0.07731879502534866
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 58, Loss: 0.41316843032836914
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 59, Loss: 0.2543833255767822
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 60, Loss: 0.1754414439201355
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 61, Loss: 0.07380692660808563
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 62, Loss: 0.20516397058963776
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 63, Loss: 0.16712582111358643
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 64, Loss: 0.2696062922477722
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 65, Loss: 0.2704631984233856
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 66, Loss: 0.29057472944259644
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 67, Loss: 0.28108352422714233
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 68, Loss: 0.3463492691516876
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 69, Loss: 0.15752659738063812
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 70, Loss: 0.19557829201221466
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 71, Loss: 0.16338329017162323
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 72, Loss: 0.08023639768362045
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 73, Loss: 0.15756744146347046
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 74, Loss: 0.16393251717090607
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 75, Loss: 0.09901604801416397
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 76, Loss: 0.13236504793167114
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 77, Loss: 0.20670737326145172
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 78, Loss: 0.276774525642395
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 79, Loss: 0.17009030282497406
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 80, Loss: 0.12433215230703354
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 81, Loss: 0.19344457983970642
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 82, Loss: 0.13428369164466858
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 83, Loss: 0.29530349373817444
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 84, Loss: 0.1856512427330017
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 85, Loss: 0.18191608786582947
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 86, Loss: 0.13250766694545746
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 87, Loss: 0.2822524905204773
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 88, Loss: 0.1750999242067337
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 89, Loss: 0.16732454299926758
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 90, Loss: 0.13121947646141052
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 91, Loss: 0.17238661646842957
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 92, Loss: 0.1950950026512146
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 93, Loss: 0.1938636600971222
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 94, Loss: 0.1744634211063385
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 95, Loss: 0.22141173481941223
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 96, Loss: 0.06786498427391052
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 97, Loss: 0.11993327736854553
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 98, Loss: 0.2206147462129593
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 99, Loss: 0.10529468953609467
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 100, Loss: 0.22520139813423157
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 101, Loss: 0.1765338033437729
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 102, Loss: 0.3523644804954529
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 103, Loss: 0.19478803873062134
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 104, Loss: 0.2607533037662506
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 105, Loss: 0.13276809453964233
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 106, Loss: 0.17941366136074066
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 107, Loss: 0.13560138642787933
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 108, Loss: 0.07946188002824783
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 109, Loss: 0.3092428147792816
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 110, Loss: 0.22151172161102295
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 111, Loss: 0.0860556960105896
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 112, Loss: 0.2785157263278961
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 113, Loss: 0.2510823607444763
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 114, Loss: 0.20390239357948303
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 115, Loss: 0.46727198362350464
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 116, Loss: 0.2152845859527588
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 117, Loss: 0.08869360387325287
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 118, Loss: 0.12092769145965576
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 119, Loss: 0.17122511565685272
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 120, Loss: 0.3467998802661896
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 121, Loss: 0.12602366507053375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 122, Loss: 0.2131182998418808
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 123, Loss: 0.18765386939048767
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 124, Loss: 0.11417078971862793
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 125, Loss: 0.31820788979530334
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 126, Loss: 0.190069779753685
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 127, Loss: 0.151798814535141
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 128, Loss: 0.1654897928237915
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 129, Loss: 0.08656341582536697
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 130, Loss: 0.09033596515655518
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 131, Loss: 0.30286839604377747
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 132, Loss: 0.2754156291484833
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 133, Loss: 0.24049781262874603
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 134, Loss: 0.16080746054649353
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 135, Loss: 0.2870978116989136
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 136, Loss: 0.24333584308624268
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 137, Loss: 0.1778055876493454
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 138, Loss: 0.2965088188648224
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 139, Loss: 0.17653247714042664
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 140, Loss: 0.19998252391815186
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 141, Loss: 0.11677370965480804
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 142, Loss: 0.06585252285003662
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 143, Loss: 0.24649742245674133
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 144, Loss: 0.4191093146800995
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 145, Loss: 0.10449402779340744
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 146, Loss: 0.23879478871822357
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 147, Loss: 0.11181221902370453
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 148, Loss: 0.3529731035232544
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 149, Loss: 0.4203750491142273
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 150, Loss: 0.19745323061943054
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 151, Loss: 0.28387802839279175
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 152, Loss: 0.30711066722869873
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 153, Loss: 0.2687206566333771
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 154, Loss: 0.2542670667171478
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 155, Loss: 0.2326488047838211
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 156, Loss: 0.24187512695789337
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 157, Loss: 0.1320016086101532
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 158, Loss: 0.15992657840251923
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 159, Loss: 0.16129720211029053
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 160, Loss: 0.1826714128255844
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 161, Loss: 0.14243417978286743
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 162, Loss: 0.4079696536064148
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 163, Loss: 0.2541336119174957
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 164, Loss: 0.23880082368850708
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 165, Loss: 0.2026081532239914
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 166, Loss: 0.18642671406269073
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 167, Loss: 0.33298465609550476
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 168, Loss: 0.22017349302768707
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 169, Loss: 0.23369213938713074
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 170, Loss: 0.20090970396995544
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 171, Loss: 0.21287290751934052
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 172, Loss: 0.18778324127197266
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 173, Loss: 0.1757170706987381
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 174, Loss: 0.14874985814094543
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 175, Loss: 0.1841457337141037
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 176, Loss: 0.11733352392911911
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 177, Loss: 0.19043156504631042
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 178, Loss: 0.21822603046894073
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 179, Loss: 0.07417035847902298
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 180, Loss: 0.18979871273040771
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 181, Loss: 0.09954964369535446
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 182, Loss: 0.11497775465250015
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 183, Loss: 0.28801053762435913
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 184, Loss: 0.1405368447303772
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 185, Loss: 0.1366749107837677
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 186, Loss: 0.28787362575531006
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 187, Loss: 0.11001316457986832
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 188, Loss: 0.1236843392252922
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 189, Loss: 0.20834797620773315
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 190, Loss: 0.397551029920578
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 191, Loss: 0.1825026124715805
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 192, Loss: 0.14656247198581696
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 193, Loss: 0.18012556433677673
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 194, Loss: 0.33267882466316223
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 195, Loss: 0.21000467240810394
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 196, Loss: 0.15127980709075928
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 197, Loss: 0.24177512526512146
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 198, Loss: 0.14383772015571594
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 199, Loss: 0.277635782957077
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 200, Loss: 0.17797045409679413
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 201, Loss: 0.10169930011034012
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 202, Loss: 0.21258671581745148
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 203, Loss: 0.32844874262809753
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 204, Loss: 0.17111514508724213
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 205, Loss: 0.1383124142885208
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 206, Loss: 0.30903053283691406
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 207, Loss: 0.08584664762020111
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 208, Loss: 0.15842697024345398
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 209, Loss: 0.23839664459228516
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 210, Loss: 0.4603795111179352
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 211, Loss: 0.3891923129558563
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 212, Loss: 0.25368979573249817
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 213, Loss: 0.20309977233409882
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 214, Loss: 0.15908502042293549
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 215, Loss: 0.12957216799259186
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 216, Loss: 0.1604146808385849
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 217, Loss: 0.30577218532562256
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 218, Loss: 0.18025381863117218
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 219, Loss: 0.4859583377838135
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 220, Loss: 0.18541811406612396
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 221, Loss: 0.21406260132789612
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 222, Loss: 0.25241196155548096
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 223, Loss: 0.09685549885034561
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 224, Loss: 0.10764104872941971
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 225, Loss: 0.20468953251838684
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 226, Loss: 0.11114795506000519
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 227, Loss: 0.1475113034248352
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 228, Loss: 0.1440209299325943
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 229, Loss: 0.3259369134902954
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 230, Loss: 0.11691437661647797
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 231, Loss: 0.1645020693540573
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 232, Loss: 0.22176049649715424
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 233, Loss: 0.15398305654525757
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 234, Loss: 0.23885802924633026
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 235, Loss: 0.2792951464653015
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 236, Loss: 0.18793095648288727
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 237, Loss: 0.0929563045501709
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 238, Loss: 0.2293011099100113
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 239, Loss: 0.22697649896144867
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 240, Loss: 0.2627718448638916
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 241, Loss: 0.23088839650154114
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 242, Loss: 0.2550617754459381
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 243, Loss: 0.10514678061008453
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 244, Loss: 0.1296742856502533
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 245, Loss: 0.20800118148326874
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 246, Loss: 0.16397131979465485
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 247, Loss: 0.24731813371181488
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 248, Loss: 0.10755384713411331
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 249, Loss: 0.18592186272144318
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 250, Loss: 0.1877356916666031
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 251, Loss: 0.2103576958179474
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 252, Loss: 0.22379587590694427
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 253, Loss: 0.2570769488811493
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 254, Loss: 0.22110895812511444
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 255, Loss: 0.13066641986370087
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 256, Loss: 0.16366077959537506
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 257, Loss: 0.20262360572814941
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 258, Loss: 0.17304928600788116
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 259, Loss: 0.14306774735450745
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 260, Loss: 0.2129576951265335
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 261, Loss: 0.3819468021392822
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 262, Loss: 0.2083219736814499
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 263, Loss: 0.2759685814380646
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 264, Loss: 0.10525497794151306
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 265, Loss: 0.17464548349380493
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 266, Loss: 0.4341045618057251
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 267, Loss: 0.18211804330348969
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 268, Loss: 0.18438802659511566
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 269, Loss: 0.4011146128177643
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 270, Loss: 0.14386163651943207
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 271, Loss: 0.23268401622772217
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 272, Loss: 0.11306194216012955
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 273, Loss: 0.10779216140508652
Predicted Boxes: [], Scores: [], Labels: []
Epoch 5, Iteration 274, Loss: 0.3259665071964264
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 0, Loss: 0.1749221831560135
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 1, Loss: 0.20261730253696442
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 2, Loss: 0.17785578966140747
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 3, Loss: 0.33825820684432983
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 4, Loss: 0.24942973256111145
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 5, Loss: 0.2869020104408264
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 6, Loss: 0.372880220413208
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 7, Loss: 0.18987014889717102
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 8, Loss: 0.23903712630271912
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 9, Loss: 0.17961639165878296
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 10, Loss: 0.11989734321832657
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 11, Loss: 0.14434382319450378
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 12, Loss: 0.2678120732307434
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 13, Loss: 0.17614924907684326
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 14, Loss: 0.4603314697742462
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 15, Loss: 0.17537356913089752
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 16, Loss: 0.13514664769172668
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 17, Loss: 0.2941923439502716
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 18, Loss: 0.19420784711837769
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 19, Loss: 0.14314667880535126
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 20, Loss: 0.21924380958080292
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 21, Loss: 0.22906626760959625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 22, Loss: 0.12551984190940857
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 23, Loss: 0.22196213901042938
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 24, Loss: 0.2744879722595215
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 25, Loss: 0.21981078386306763
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 26, Loss: 0.14942951500415802
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 27, Loss: 0.14621667563915253
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 28, Loss: 0.13354942202568054
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 29, Loss: 0.1472436487674713
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 30, Loss: 0.09642177820205688
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 31, Loss: 0.38858336210250854
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 32, Loss: 0.28175410628318787
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 33, Loss: 0.26590901613235474
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 34, Loss: 0.1521991640329361
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 35, Loss: 0.2857222855091095
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 36, Loss: 0.14963920414447784
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 37, Loss: 0.2634432911872864
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 38, Loss: 0.2675843834877014
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 39, Loss: 0.24566753208637238
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 40, Loss: 0.13681158423423767
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 41, Loss: 0.2906709909439087
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 42, Loss: 0.32734885811805725
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 43, Loss: 0.388773649930954
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 44, Loss: 0.2379225343465805
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 45, Loss: 0.19571147859096527
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 46, Loss: 0.13461947441101074
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 47, Loss: 0.17573915421962738
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 48, Loss: 0.15084779262542725
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 49, Loss: 0.20859672129154205
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 50, Loss: 0.2944933772087097
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 51, Loss: 0.13528186082839966
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 52, Loss: 0.19132386147975922
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 53, Loss: 0.20203785598278046
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 54, Loss: 0.1261904537677765
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 55, Loss: 0.28353050351142883
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 56, Loss: 0.17679449915885925
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 57, Loss: 0.20191262662410736
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 58, Loss: 0.2099226713180542
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 59, Loss: 0.2629794180393219
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 60, Loss: 0.2802848517894745
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 61, Loss: 0.10488305240869522
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 62, Loss: 0.31784501671791077
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 63, Loss: 0.25635185837745667
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 64, Loss: 0.18692553043365479
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 65, Loss: 0.5033628344535828
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 66, Loss: 0.2455861121416092
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 67, Loss: 0.16671235859394073
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 68, Loss: 0.1591525375843048
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 69, Loss: 0.1991698145866394
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 70, Loss: 0.16771309077739716
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 71, Loss: 0.30649182200431824
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 72, Loss: 0.5824247002601624
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 73, Loss: 0.3122555613517761
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 74, Loss: 0.3755917251110077
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 75, Loss: 0.13261370360851288
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 76, Loss: 0.1418744921684265
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 77, Loss: 0.23320141434669495
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 78, Loss: 0.20898737013339996
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 79, Loss: 0.18318404257297516
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 80, Loss: 0.14679548144340515
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 81, Loss: 0.36639562249183655
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 82, Loss: 0.1410352885723114
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 83, Loss: 0.14848442375659943
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 84, Loss: 0.160238578915596
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 85, Loss: 0.18094821274280548
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 86, Loss: 0.19267240166664124
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 87, Loss: 0.16260358691215515
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 88, Loss: 0.12860292196273804
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 89, Loss: 0.07572869211435318
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 90, Loss: 0.1273430734872818
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 91, Loss: 0.24027986824512482
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 92, Loss: 0.3170846700668335
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 93, Loss: 0.21386651694774628
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 94, Loss: 0.05751120671629906
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 95, Loss: 0.17967697978019714
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 96, Loss: 0.8319095969200134
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 97, Loss: 0.14554919302463531
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 98, Loss: 0.20754514634609222
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 99, Loss: 0.23078212141990662
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 100, Loss: 0.17832528054714203
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 101, Loss: 0.33307531476020813
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 102, Loss: 0.13050243258476257
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 103, Loss: 0.215598925948143
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 104, Loss: 1.6323235034942627
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 105, Loss: 0.16589343547821045
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 106, Loss: 0.5033285021781921
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 107, Loss: 0.7647499442100525
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 108, Loss: 0.20894019305706024
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 109, Loss: 0.2949874699115753
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 110, Loss: 0.6804661750793457
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 111, Loss: 0.4315943419933319
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 112, Loss: 0.1406799554824829
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 113, Loss: 0.19830921292304993
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 114, Loss: 0.5891600251197815
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 115, Loss: 0.21419279277324677
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 116, Loss: 0.22242681682109833
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 117, Loss: 0.4698964059352875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 118, Loss: 0.4497944414615631
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 119, Loss: 0.48830699920654297
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 120, Loss: 0.396045058965683
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 121, Loss: 0.3490443229675293
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 122, Loss: 0.32241055369377136
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 123, Loss: 0.5866146087646484
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 124, Loss: 0.9537909030914307
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 125, Loss: 0.7890599966049194
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 126, Loss: 0.5536063313484192
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 127, Loss: 0.7397179007530212
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 128, Loss: 0.18548819422721863
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 129, Loss: 0.7598240971565247
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 130, Loss: 1.0603042840957642
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 131, Loss: 0.1544855833053589
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 132, Loss: 0.6454839110374451
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 133, Loss: 0.5716687440872192
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 134, Loss: 1.373931646347046
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 135, Loss: 1.773115634918213
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 136, Loss: 2.1902408599853516
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 137, Loss: 1.783420443534851
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 138, Loss: 14.663437843322754
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 139, Loss: 0.9131879210472107
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 140, Loss: 2.1333584785461426
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 141, Loss: 11.265365600585938
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 142, Loss: 3.2748169898986816
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 143, Loss: 3.2171473503112793
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 144, Loss: 96.47817993164062
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 145, Loss: 0.8354545831680298
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 146, Loss: 76.52811431884766
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 147, Loss: 2646.563720703125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 148, Loss: 0.6477903723716736
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 149, Loss: 0.8979855179786682
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 150, Loss: 1.0315253734588623
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 151, Loss: 1.2159630060195923
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 152, Loss: 3348.8681640625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 153, Loss: 12131518.0
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 154, Loss: 3268.819580078125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 155, Loss: 4.573182106018066
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 156, Loss: 5.170116424560547
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 157, Loss: 1367.034912109375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 158, Loss: 2281.050537109375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 159, Loss: 1653.0052490234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 160, Loss: 66921.4140625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 161, Loss: 14.111167907714844
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 162, Loss: 93.29724884033203
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 163, Loss: 133.8766632080078
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 164, Loss: 45221.109375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 165, Loss: 2406.688720703125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 166, Loss: 8966.0693359375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 167, Loss: 11635.263671875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 168, Loss: 5119.1904296875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 169, Loss: 659.012939453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 170, Loss: 229687.734375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 171, Loss: 27952.236328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 172, Loss: 7176.794921875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 173, Loss: 521.2138061523438
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 174, Loss: 561.9922485351562
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 175, Loss: 300.32623291015625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 176, Loss: 219.32467651367188
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 177, Loss: 367.8133850097656
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 178, Loss: 347.18359375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 179, Loss: 65.13491821289062
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 180, Loss: 5031.47265625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 181, Loss: 85.34136199951172
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 182, Loss: 892.5702514648438
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 183, Loss: 341.0453796386719
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 184, Loss: 6045.03955078125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 185, Loss: 219.8708953857422
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 186, Loss: 10745.453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 187, Loss: 533.9769897460938
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 188, Loss: 21635.49609375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 189, Loss: 351.9065856933594
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 190, Loss: 619.4583740234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 191, Loss: 2774.549072265625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 192, Loss: 1983.848876953125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 193, Loss: 2463.056640625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 194, Loss: 5022366.5
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 195, Loss: 27639.39453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 196, Loss: 20141.505859375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 197, Loss: 866735.5625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 198, Loss: 4282.7939453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 199, Loss: 2266.218017578125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 200, Loss: 3219.614013671875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 201, Loss: 14670.8310546875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 202, Loss: 8723.04296875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 203, Loss: 7378.72119140625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 204, Loss: 5124.330078125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 205, Loss: 3482.281494140625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 206, Loss: 26857.857421875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 207, Loss: 122415.25
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 208, Loss: 31305.216796875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 209, Loss: 11136.142578125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 210, Loss: 12177.98828125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 211, Loss: 16262.1865234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 212, Loss: 7010.97021484375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 213, Loss: 80279.75
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 214, Loss: 97690.828125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 215, Loss: 8503.9033203125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 216, Loss: 12280.875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 217, Loss: 15682.81640625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 218, Loss: 11769.470703125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 219, Loss: 11563.216796875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 220, Loss: 11626.1123046875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 221, Loss: 6261.80859375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 222, Loss: 585386.5625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 223, Loss: 6306.115234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 224, Loss: 8647.6845703125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 225, Loss: 4968.92041015625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 226, Loss: 4732.32470703125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 227, Loss: 2269167.5
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 228, Loss: 13568.8017578125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 229, Loss: 8537.3779296875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 230, Loss: 10168.458984375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 231, Loss: 3944.221923828125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 232, Loss: 3112.76953125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 233, Loss: 6909.7138671875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 234, Loss: 16876.9453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 235, Loss: 4916.912109375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 236, Loss: 3211.46240234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 237, Loss: 47406.23046875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 238, Loss: 6671.490234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 239, Loss: 4936.72509765625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 240, Loss: 3056.403076171875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 241, Loss: 1577.0421142578125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 242, Loss: 4423.65380859375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 243, Loss: 32773.0546875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 244, Loss: 15097.001953125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 245, Loss: 7691.28759765625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 246, Loss: 3936.76123046875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 247, Loss: 2109.516357421875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 248, Loss: 2238.703369140625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 249, Loss: 2630.036865234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 250, Loss: 1224.9300537109375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 251, Loss: 2906.15576171875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 252, Loss: 1453.2432861328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 253, Loss: 2757.39013671875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 254, Loss: 1679.0989990234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 255, Loss: 3388.09912109375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 256, Loss: 1738.578125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 257, Loss: 985.2151489257812
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 258, Loss: 1143.5748291015625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 259, Loss: 7179.47607421875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 260, Loss: 1546.1495361328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 261, Loss: 2634.787109375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 262, Loss: 452867.25
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 263, Loss: 4323.81591796875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 264, Loss: 6988.2568359375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 265, Loss: 4460.05126953125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 266, Loss: 3108.44091796875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 267, Loss: 1549.0467529296875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 268, Loss: 1130.509521484375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 269, Loss: 253440.4375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 270, Loss: 931.979736328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 271, Loss: 2175.829345703125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 272, Loss: 2261.47412109375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 273, Loss: 2888.58349609375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 6, Iteration 274, Loss: 2496.078369140625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 0, Loss: 713.2063598632812
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 1, Loss: 302.468017578125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 2, Loss: 2772.140869140625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 3, Loss: 462.8231201171875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 4, Loss: 503.1226501464844
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 5, Loss: 4405.189453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 6, Loss: 795.395751953125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 7, Loss: 1258.5606689453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 8, Loss: 1212.23779296875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 9, Loss: 1289.2933349609375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 10, Loss: 1320.28173828125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 11, Loss: 284.6548767089844
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 12, Loss: 687.0478515625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 13, Loss: 1384.0947265625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 14, Loss: 4150.5185546875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 15, Loss: 1250.361328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 16, Loss: 1261.608154296875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 17, Loss: 1002.7198486328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 18, Loss: 331725.1875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 19, Loss: 1512.44189453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 20, Loss: 1394.95947265625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 21, Loss: 1264.2205810546875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 22, Loss: 398.4471130371094
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 23, Loss: 1057.3953857421875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 24, Loss: 672.1871948242188
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 25, Loss: 196.37091064453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 26, Loss: 267.73760986328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 27, Loss: 97.77171325683594
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 28, Loss: 1016.2484130859375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 29, Loss: 6350.91748046875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 30, Loss: 8000.93359375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 31, Loss: 5065.9560546875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 32, Loss: 5525.6943359375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 33, Loss: 6581.1767578125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 34, Loss: 5230.4365234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 35, Loss: 20434.15625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 36, Loss: 4946.19970703125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 37, Loss: 3866.01025390625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 38, Loss: 999.0162353515625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 39, Loss: 6976.59521484375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 40, Loss: 2340.93212890625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 41, Loss: 2387.553466796875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 42, Loss: 77082.484375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 43, Loss: 2339.04736328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 44, Loss: 35999.36328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 45, Loss: 2121.1025390625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 46, Loss: 20933.798828125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 47, Loss: 19845.8828125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 48, Loss: 23831.00390625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 49, Loss: 11704.173828125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 50, Loss: 4534.5830078125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 51, Loss: 5675.09912109375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 52, Loss: 3237.195068359375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 53, Loss: 4803.65478515625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 54, Loss: 4429.26904296875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 55, Loss: 5633.2158203125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 56, Loss: 1868.725830078125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 57, Loss: 24296.22265625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 58, Loss: 431249.8125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 59, Loss: 5219.92822265625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 60, Loss: 1474.469482421875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 61, Loss: 1736.2509765625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 62, Loss: 442.5808410644531
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 63, Loss: 3699.36279296875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 64, Loss: 3264.493896484375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 65, Loss: 531.0302124023438
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 66, Loss: 80.2430419921875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 67, Loss: 784.26708984375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 68, Loss: 3605.9248046875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 69, Loss: 1062.250244140625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 70, Loss: 2107.782470703125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 71, Loss: 189.8462677001953
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 72, Loss: 567.2410278320312
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 73, Loss: 165.33880615234375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 74, Loss: 159.70899963378906
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 75, Loss: 711.3294677734375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 76, Loss: 1073.120849609375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 77, Loss: 410.828369140625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 78, Loss: 16757.009765625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 79, Loss: 118.54474639892578
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 80, Loss: 1381.953369140625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 81, Loss: 827.1310424804688
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 82, Loss: 983.0331420898438
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 83, Loss: 212.9346466064453
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 84, Loss: 180.4431610107422
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 85, Loss: 826.8616333007812
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 86, Loss: 163170.34375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 87, Loss: 105.80167388916016
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 88, Loss: 79.95018005371094
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 89, Loss: 442.0644226074219
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 90, Loss: 37.974552154541016
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 91, Loss: 2661.234130859375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 92, Loss: 2256.573486328125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 93, Loss: 799.633056640625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 94, Loss: 916.99853515625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 95, Loss: 934.8072509765625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 96, Loss: 1264.4849853515625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 97, Loss: 727.4239501953125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 98, Loss: 1528.2637939453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 99, Loss: 1389.8336181640625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 100, Loss: 1097.7950439453125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 101, Loss: 1147.8756103515625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 102, Loss: 809.5645751953125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 103, Loss: 376.69647216796875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 104, Loss: 532.18408203125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 105, Loss: 141.95982360839844
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 106, Loss: 354.67706298828125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 107, Loss: 403.0216369628906
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 108, Loss: 37.71331024169922
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 109, Loss: 110.8746109008789
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 110, Loss: 47.37268829345703
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 111, Loss: 35.563289642333984
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 112, Loss: 9.956021308898926
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 113, Loss: 21.813411712646484
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 114, Loss: 27.823928833007812
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 115, Loss: 21.726377487182617
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 116, Loss: 32.569252014160156
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 117, Loss: 5.068048000335693
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 118, Loss: 34.75740051269531
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 119, Loss: 3053.07177734375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 120, Loss: 2122.315185546875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 121, Loss: 1229.2530517578125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 122, Loss: 218.23175048828125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 123, Loss: 129.9439697265625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 124, Loss: 173.59359741210938
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 125, Loss: 29.493282318115234
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 126, Loss: 126.41914367675781
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 127, Loss: 64.62705993652344
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 128, Loss: 127.34429168701172
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 129, Loss: 69.15574645996094
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 130, Loss: 488.1646728515625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 131, Loss: 126.06782531738281
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 132, Loss: 5562.32470703125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 133, Loss: 239.2513885498047
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 134, Loss: 66.58419036865234
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 135, Loss: 81.71398162841797
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 136, Loss: 92.47515869140625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 137, Loss: 174.97250366210938
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 138, Loss: 40.604400634765625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 139, Loss: 31.95096206665039
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 140, Loss: 52.55876541137695
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 141, Loss: 35.3726806640625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 142, Loss: 749.7528076171875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 143, Loss: 86.71488952636719
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 144, Loss: 17.196523666381836
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 145, Loss: 28.60283088684082
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 146, Loss: 551.63525390625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 147, Loss: 144.33059692382812
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 148, Loss: 24.994447708129883
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 149, Loss: 10.160910606384277
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 150, Loss: 30.73883056640625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 151, Loss: 43.142452239990234
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 152, Loss: 15.667588233947754
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 153, Loss: 30.30887222290039
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 154, Loss: 6.751097679138184
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 155, Loss: 4.7404985427856445
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 156, Loss: 9.647688865661621
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 157, Loss: 44.317718505859375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 158, Loss: 166.1627197265625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 159, Loss: 385.84381103515625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 160, Loss: 5.409590721130371
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 161, Loss: 13.64616870880127
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 162, Loss: 19.36711883544922
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 163, Loss: 8.989309310913086
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 164, Loss: 53.75381088256836
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 165, Loss: 19.264808654785156
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 166, Loss: 59.49858093261719
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 167, Loss: 48.721012115478516
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 168, Loss: 89.47708892822266
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 169, Loss: 7.978233337402344
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 170, Loss: 2.1798911094665527
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 171, Loss: 32.695037841796875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 172, Loss: 5.927618026733398
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 173, Loss: 2.8436551094055176
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 174, Loss: 16.128032684326172
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 175, Loss: 8.245745658874512
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 176, Loss: 9.829214096069336
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 177, Loss: 2.952423095703125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 178, Loss: 13.09868335723877
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 179, Loss: 11.536367416381836
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 180, Loss: 3.835890531539917
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 181, Loss: 73.06127166748047
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 182, Loss: 15.215091705322266
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 183, Loss: 14.291110038757324
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 184, Loss: 7.600564002990723
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 185, Loss: 6.535836219787598
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 186, Loss: 0.7646218538284302
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 187, Loss: 0.7941521406173706
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 188, Loss: 125.53826904296875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 189, Loss: 0.7629245519638062
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 190, Loss: 1.856613039970398
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 191, Loss: 301.1299133300781
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 192, Loss: 665.8533935546875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 193, Loss: 45.462127685546875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 194, Loss: 1.0004793405532837
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 195, Loss: 1.2998062372207642
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 196, Loss: 29.16067123413086
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 197, Loss: 18.655006408691406
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 198, Loss: 3.6851327419281006
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 199, Loss: 1587.8463134765625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 200, Loss: 14.174790382385254
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 201, Loss: 7.244946002960205
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 202, Loss: 26.133686065673828
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 203, Loss: 9.575162887573242
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 204, Loss: 0.6648195385932922
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 205, Loss: 0.6746385097503662
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 206, Loss: 9.491337776184082
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 207, Loss: 0.6504256725311279
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 208, Loss: 6.416690826416016
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 209, Loss: 86.33991241455078
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 210, Loss: 110.95458984375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 211, Loss: 15.608048439025879
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 212, Loss: 2.384877920150757
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 213, Loss: 0.6417297124862671
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 214, Loss: 0.5649406313896179
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 215, Loss: 0.5496351718902588
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 216, Loss: 0.5420182347297668
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 217, Loss: 9.959283828735352
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 218, Loss: 3.686004400253296
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 219, Loss: 9.718827247619629
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 220, Loss: 15.440134048461914
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 221, Loss: 0.5335919260978699
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 222, Loss: 0.46595099568367004
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 223, Loss: 12.123905181884766
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 224, Loss: 0.4858129322528839
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 225, Loss: 0.4331709146499634
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 226, Loss: 16.52223777770996
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 227, Loss: 0.41609376668930054
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 228, Loss: 0.7230269312858582
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 229, Loss: 1.481802225112915
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 230, Loss: 0.5269244909286499
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 231, Loss: 0.41856682300567627
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 232, Loss: 0.42993980646133423
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 233, Loss: 0.3909843862056732
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 234, Loss: 226.26657104492188
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 235, Loss: 0.4076356887817383
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 236, Loss: 0.3943640887737274
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 237, Loss: 0.9273787140846252
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 238, Loss: 0.33172857761383057
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 239, Loss: 67.76364135742188
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 240, Loss: 0.3493692874908447
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 241, Loss: 6.452171802520752
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 242, Loss: 0.31811416149139404
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 243, Loss: 0.29341086745262146
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 244, Loss: 0.30766430497169495
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 245, Loss: 0.3073263466358185
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 246, Loss: 0.2721043825149536
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 247, Loss: 20.01136016845703
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 248, Loss: 0.2671900987625122
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 249, Loss: 0.253711074590683
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 250, Loss: 4.591475486755371
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 251, Loss: 0.21011263132095337
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 252, Loss: 0.22187690436840057
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 253, Loss: 0.23103469610214233
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 254, Loss: 0.26939401030540466
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 255, Loss: 0.1827588677406311
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 256, Loss: 0.19107788801193237
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 257, Loss: 0.16579031944274902
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 258, Loss: 0.21845749020576477
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 259, Loss: 0.22201605141162872
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 260, Loss: 0.2022702693939209
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 261, Loss: 110.8741683959961
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 262, Loss: 0.1864251345396042
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 263, Loss: 0.23010841012001038
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 264, Loss: 0.21831773221492767
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 265, Loss: 26.019363403320312
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 266, Loss: 6.496160507202148
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 267, Loss: 0.21787726879119873
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 268, Loss: 0.17036867141723633
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 269, Loss: 0.8476024866104126
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 270, Loss: 0.21345964074134827
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 271, Loss: 105.17257690429688
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 272, Loss: 0.18045660853385925
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 273, Loss: 0.1784283071756363
Predicted Boxes: [], Scores: [], Labels: []
Epoch 7, Iteration 274, Loss: 0.18969549238681793
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 0, Loss: 0.19919568300247192
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 1, Loss: 0.1535569131374359
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 2, Loss: 5.445180416107178
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 3, Loss: 0.3155099153518677
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 4, Loss: 0.17092646658420563
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 5, Loss: 988.659423828125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 6, Loss: 32.97893142700195
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 7, Loss: 4.209911346435547
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 8, Loss: 127.10783386230469
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 9, Loss: 15.004308700561523
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 10, Loss: 74.22702026367188
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 11, Loss: 39.420589447021484
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 12, Loss: 21.54376983642578
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 13, Loss: 12.530061721801758
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 14, Loss: 0.6342204213142395
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 15, Loss: 0.25821182131767273
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 16, Loss: 0.16829223930835724
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 17, Loss: 0.21393314003944397
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 18, Loss: 16.539695739746094
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 19, Loss: 2.0682499408721924
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 20, Loss: 11.391018867492676
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 21, Loss: 0.1974138617515564
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 22, Loss: 0.20717430114746094
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 23, Loss: 0.16037850081920624
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 24, Loss: 0.17419245839118958
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 25, Loss: 0.16788429021835327
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 26, Loss: 0.17388850450515747
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 27, Loss: 2.508610486984253
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 28, Loss: 0.16019894182682037
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 29, Loss: 0.1809818297624588
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 30, Loss: 172.12179565429688
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 31, Loss: 0.15789148211479187
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 32, Loss: 0.15887781977653503
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 33, Loss: 0.680840790271759
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 34, Loss: 211.67825317382812
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 35, Loss: 47.45978546142578
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 36, Loss: 0.19278225302696228
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 37, Loss: 44.014766693115234
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 38, Loss: 3.274904727935791
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 39, Loss: 15.114410400390625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 40, Loss: 0.5789843797683716
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 41, Loss: 20.472606658935547
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 42, Loss: 26.150074005126953
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 43, Loss: 0.13285982608795166
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 44, Loss: 0.15498176217079163
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 45, Loss: 0.16732071340084076
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 46, Loss: 75.37345886230469
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 47, Loss: 0.16956481337547302
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 48, Loss: 0.12843576073646545
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 49, Loss: 47.67988586425781
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 50, Loss: 0.1481434553861618
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 51, Loss: 198.072998046875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 52, Loss: 0.1727602183818817
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 53, Loss: 4.2325053215026855
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 54, Loss: 0.17008210718631744
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 55, Loss: 0.36491256952285767
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 56, Loss: 0.1769370585680008
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 57, Loss: 0.19117018580436707
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 58, Loss: 0.18423809111118317
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 59, Loss: 0.1608818918466568
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 60, Loss: 0.16462549567222595
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 61, Loss: 0.1843559592962265
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 62, Loss: 0.17494575679302216
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 63, Loss: 1.5911622047424316
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 64, Loss: 0.1598479002714157
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 65, Loss: 0.1633184403181076
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 66, Loss: 0.14816085994243622
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 67, Loss: 0.18528372049331665
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 68, Loss: 0.15855418145656586
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 69, Loss: 0.21745844185352325
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 70, Loss: 0.2289099544286728
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 71, Loss: 0.15235331654548645
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 72, Loss: 0.18787738680839539
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 73, Loss: 0.13809332251548767
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 74, Loss: 0.1827993392944336
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 75, Loss: 0.16578182578086853
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 76, Loss: 0.21196217834949493
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 77, Loss: 0.18581219017505646
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 78, Loss: 0.16715869307518005
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 79, Loss: 0.1464202105998993
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 80, Loss: 0.1480763703584671
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 81, Loss: 377.0740051269531
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 82, Loss: 4.5671610832214355
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 83, Loss: 159.41944885253906
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 84, Loss: 0.14353641867637634
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 85, Loss: 0.1797066330909729
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 86, Loss: 0.14299902319908142
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 87, Loss: 0.16481679677963257
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 88, Loss: 0.14838118851184845
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 89, Loss: 0.11671656370162964
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 90, Loss: 0.18416652083396912
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 91, Loss: 29.913990020751953
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 92, Loss: 0.17358627915382385
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 93, Loss: 0.15971608459949493
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 94, Loss: 983.0350341796875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 95, Loss: 1.7869917154312134
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 96, Loss: 0.17397664487361908
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 97, Loss: 74.25337219238281
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 98, Loss: 0.7637839317321777
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 99, Loss: 0.16413633525371552
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 100, Loss: 0.20414999127388
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 101, Loss: 4.186915397644043
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 102, Loss: 14.268538475036621
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 103, Loss: 1.0360190868377686
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 104, Loss: 0.12879572808742523
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 105, Loss: 0.17958718538284302
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 106, Loss: 11.52200984954834
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 107, Loss: 0.1882622092962265
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 108, Loss: 20.47036361694336
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 109, Loss: 2.6068944931030273
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 110, Loss: 7.45745849609375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 111, Loss: 10.567100524902344
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 112, Loss: 6.3777971267700195
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 113, Loss: 184.77685546875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 114, Loss: 0.15949572622776031
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 115, Loss: 62.85374450683594
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 116, Loss: 9.290790557861328
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 117, Loss: 3.161578416824341
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 118, Loss: 0.1478641778230667
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 119, Loss: 0.17110484838485718
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 120, Loss: 0.18708902597427368
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 121, Loss: 0.16063667833805084
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 122, Loss: 0.1719830185174942
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 123, Loss: 0.1599922627210617
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 124, Loss: 0.14559736847877502
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 125, Loss: 0.2206849902868271
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 126, Loss: 0.1841191053390503
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 127, Loss: 0.18399043381214142
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 128, Loss: 0.16513802111148834
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 129, Loss: 0.15767443180084229
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 130, Loss: 0.1759207546710968
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 131, Loss: 0.1706472933292389
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 132, Loss: 0.1733924001455307
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 133, Loss: 0.15514473617076874
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 134, Loss: 0.15147432684898376
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 135, Loss: 0.14764650166034698
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 136, Loss: 0.1571420431137085
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 137, Loss: 0.16533468663692474
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 138, Loss: 0.15404579043388367
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 139, Loss: 0.6122385263442993
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 140, Loss: 0.14688259363174438
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 141, Loss: 0.15061654150485992
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 142, Loss: 0.14579670131206512
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 143, Loss: 0.16549812257289886
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 144, Loss: 0.1781083196401596
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 145, Loss: 0.15301816165447235
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 146, Loss: 0.14354915916919708
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 147, Loss: 0.16505934298038483
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 148, Loss: 0.18352866172790527
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 149, Loss: 0.1617605835199356
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 150, Loss: 0.16861361265182495
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 151, Loss: 0.19113469123840332
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 152, Loss: 0.18724879622459412
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 153, Loss: 0.15787215530872345
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 154, Loss: 0.18045499920845032
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 155, Loss: 0.15616288781166077
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 156, Loss: 0.18353213369846344
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 157, Loss: 0.17736880481243134
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 158, Loss: 0.21917632222175598
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 159, Loss: 0.17683464288711548
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 160, Loss: 0.14225153625011444
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 161, Loss: 0.13722679018974304
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 162, Loss: 0.14094612002372742
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 163, Loss: 9.162456512451172
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 164, Loss: 1.0505229234695435
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 165, Loss: 243.17051696777344
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 166, Loss: 0.1673034429550171
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 167, Loss: 0.16164752840995789
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 168, Loss: 0.17975549399852753
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 169, Loss: 0.164696604013443
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 170, Loss: 0.14214248955249786
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 171, Loss: 0.16896328330039978
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 172, Loss: 0.15830233693122864
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 173, Loss: 0.14338873326778412
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 174, Loss: 0.18530483543872833
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 175, Loss: 0.1398031860589981
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 176, Loss: 0.16800349950790405
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 177, Loss: 3.171525239944458
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 178, Loss: 0.1737261414527893
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 179, Loss: 0.17179562151432037
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 180, Loss: 0.13905459642410278
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 181, Loss: 0.17041130363941193
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 182, Loss: 0.17779426276683807
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 183, Loss: 0.14897091686725616
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 184, Loss: 0.20531493425369263
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 185, Loss: 0.15819284319877625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 186, Loss: 0.163142591714859
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 187, Loss: 0.15182633697986603
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 188, Loss: 0.16921722888946533
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 189, Loss: 482.59051513671875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 190, Loss: 0.15335895121097565
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 191, Loss: 0.1720827966928482
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 192, Loss: 1.3956961631774902
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 193, Loss: 0.14585526287555695
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 194, Loss: 0.15644173324108124
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 195, Loss: 0.15493722259998322
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 196, Loss: 0.1415797472000122
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 197, Loss: 0.15522293746471405
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 198, Loss: 0.14364390075206757
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 199, Loss: 0.3992213010787964
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 200, Loss: 0.16579777002334595
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 201, Loss: 0.14043641090393066
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 202, Loss: 0.11915028840303421
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 203, Loss: 0.14289194345474243
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 204, Loss: 0.17921775579452515
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 205, Loss: 0.13385240733623505
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 206, Loss: 0.1754409670829773
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 207, Loss: 0.18055404722690582
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 208, Loss: 0.18914395570755005
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 209, Loss: 0.154335156083107
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 210, Loss: 0.14162112772464752
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 211, Loss: 0.1473994106054306
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 212, Loss: 0.159038245677948
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 213, Loss: 0.15144497156143188
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 214, Loss: 0.13422057032585144
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 215, Loss: 0.17503592371940613
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 216, Loss: 12.20763874053955
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 217, Loss: 0.18063735961914062
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 218, Loss: 0.16180631518363953
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 219, Loss: 0.15352150797843933
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 220, Loss: 0.17274847626686096
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 221, Loss: 0.13071459531784058
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 222, Loss: 0.15741750597953796
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 223, Loss: 0.15377187728881836
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 224, Loss: 0.15840451419353485
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 225, Loss: 0.15813586115837097
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 226, Loss: 0.140299454331398
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 227, Loss: 0.14787444472312927
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 228, Loss: 0.14631839096546173
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 229, Loss: 0.14855606853961945
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 230, Loss: 0.15233738720417023
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 231, Loss: 0.15017758309841156
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 232, Loss: 0.14664208889007568
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 233, Loss: 0.17606177926063538
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 234, Loss: 0.18034912645816803
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 235, Loss: 0.14059162139892578
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 236, Loss: 0.1628626137971878
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 237, Loss: 0.13781704008579254
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 238, Loss: 0.13250082731246948
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 239, Loss: 0.14192895591259003
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 240, Loss: 0.14598317444324493
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 241, Loss: 0.12331271171569824
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 242, Loss: 0.1656724214553833
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 243, Loss: 0.15861821174621582
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 244, Loss: 0.1425178498029709
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 245, Loss: 0.15915876626968384
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 246, Loss: 0.1458682119846344
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 247, Loss: 0.15665587782859802
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 248, Loss: 0.1787610501050949
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 249, Loss: 0.16041748225688934
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 250, Loss: 0.16916485130786896
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 251, Loss: 0.1632518321275711
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 252, Loss: 0.1207839846611023
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 253, Loss: 0.13761982321739197
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 254, Loss: 0.17296646535396576
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 255, Loss: 0.16772820055484772
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 256, Loss: 0.17954851686954498
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 257, Loss: 0.1430846005678177
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 258, Loss: 0.13713426887989044
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 259, Loss: 0.14122813940048218
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 260, Loss: 0.15228824317455292
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 261, Loss: 0.1323142796754837
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 262, Loss: 0.15497097373008728
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 263, Loss: 132.85104370117188
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 264, Loss: 0.1605849713087082
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 265, Loss: 0.15979471802711487
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 266, Loss: 0.1475156545639038
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 267, Loss: 0.13671459257602692
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 268, Loss: 0.1687391996383667
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 269, Loss: 0.15449903905391693
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 270, Loss: 0.14952388405799866
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 271, Loss: 0.15646198391914368
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 272, Loss: 0.15296076238155365
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 273, Loss: 0.1596354842185974
Predicted Boxes: [], Scores: [], Labels: []
Epoch 8, Iteration 274, Loss: 0.12975963950157166
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 0, Loss: 0.13059110939502716
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 1, Loss: 0.1828620880842209
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 2, Loss: 56.01639175415039
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 3, Loss: 459.8063659667969
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 4, Loss: 0.1817788928747177
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 5, Loss: 0.14806854724884033
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 6, Loss: 0.15241557359695435
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 7, Loss: 0.13208557665348053
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 8, Loss: 0.15246501564979553
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 9, Loss: 0.14671167731285095
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 10, Loss: 0.15134911239147186
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 11, Loss: 0.1686442792415619
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 12, Loss: 0.13832570612430573
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 13, Loss: 0.15145763754844666
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 14, Loss: 0.1191769391298294
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 15, Loss: 0.1522836536169052
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 16, Loss: 0.1450089067220688
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 17, Loss: 0.1326896697282791
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 18, Loss: 0.13525202870368958
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 19, Loss: 0.12807609140872955
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 20, Loss: 0.14429791271686554
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 21, Loss: 0.15098819136619568
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 22, Loss: 0.1446188986301422
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 23, Loss: 0.1543317288160324
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 24, Loss: 0.1428714096546173
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 25, Loss: 0.14153245091438293
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 26, Loss: 0.11445501446723938
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 27, Loss: 0.15614458918571472
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 28, Loss: 0.13786862790584564
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 29, Loss: 0.14484837651252747
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 30, Loss: 0.13300441205501556
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 31, Loss: 0.16141058504581451
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 32, Loss: 0.13909058272838593
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 33, Loss: 0.14025601744651794
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 34, Loss: 0.17480003833770752
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 35, Loss: 0.15630170702934265
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 36, Loss: 0.1338767409324646
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 37, Loss: 0.18179583549499512
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 38, Loss: 0.15310132503509521
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 39, Loss: 0.1433800756931305
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 40, Loss: 0.1318456083536148
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 41, Loss: 0.12673895061016083
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 42, Loss: 0.1479286551475525
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 43, Loss: 0.12877142429351807
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 44, Loss: 0.22951436042785645
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 45, Loss: 0.14284056425094604
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 46, Loss: 0.2504230737686157
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 47, Loss: 0.20911149680614471
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 48, Loss: 0.13601909577846527
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 49, Loss: 0.16798831522464752
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 50, Loss: 0.2820550501346588
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 51, Loss: 0.2845282256603241
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 52, Loss: 0.27782541513442993
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 53, Loss: 0.13695602118968964
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 54, Loss: 0.36818408966064453
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 55, Loss: 0.0725606232881546
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 56, Loss: 515.094482421875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 57, Loss: 0.12381940335035324
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 58, Loss: 0.5256339907646179
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 59, Loss: 0.15290656685829163
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 60, Loss: 0.12501901388168335
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 61, Loss: 0.14390455186367035
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 62, Loss: 0.14172375202178955
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 63, Loss: 0.1355859786272049
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 64, Loss: 0.14985373616218567
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 65, Loss: 0.14593255519866943
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 66, Loss: 0.1291680485010147
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 67, Loss: 0.15023484826087952
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 68, Loss: 0.24180395901203156
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 69, Loss: 0.15316276252269745
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 70, Loss: 0.15771876275539398
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 71, Loss: 0.13681545853614807
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 72, Loss: 0.13225406408309937
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 73, Loss: 0.16364075243473053
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 74, Loss: 0.15622487664222717
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 75, Loss: 0.1520949900150299
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 76, Loss: 0.1525370180606842
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 77, Loss: 0.1399342566728592
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 78, Loss: 0.13091957569122314
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 79, Loss: 0.10636916756629944
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 80, Loss: 0.15009434521198273
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 81, Loss: 0.1605011224746704
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 82, Loss: 144.3787841796875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 83, Loss: 0.15237700939178467
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 84, Loss: 0.1223413422703743
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 85, Loss: 0.13414406776428223
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 86, Loss: 0.1489095240831375
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 87, Loss: 0.14579486846923828
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 88, Loss: 0.1425393521785736
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 89, Loss: 0.14991702139377594
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 90, Loss: 0.13508738577365875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 91, Loss: 0.15876446664333344
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 92, Loss: 0.1466112732887268
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 93, Loss: 0.11307405680418015
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 94, Loss: 0.16522733867168427
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 95, Loss: 0.15214844048023224
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 96, Loss: 0.17113900184631348
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 97, Loss: 0.14011767506599426
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 98, Loss: 0.17325177788734436
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 99, Loss: 0.11496558040380478
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 100, Loss: 0.14150361716747284
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 101, Loss: 0.14000843465328217
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 102, Loss: 0.14324603974819183
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 103, Loss: 0.14165669679641724
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 104, Loss: 0.11587381362915039
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 105, Loss: 0.1411546915769577
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 106, Loss: 0.15395598113536835
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 107, Loss: 0.13706274330615997
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 108, Loss: 0.16622380912303925
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 109, Loss: 0.15485352277755737
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 110, Loss: 0.16639696061611176
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 111, Loss: 0.12175466865301132
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 112, Loss: 0.14371636509895325
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 113, Loss: 0.13960400223731995
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 114, Loss: 0.13719432055950165
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 115, Loss: 0.14996741712093353
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 116, Loss: 0.13942794501781464
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 117, Loss: 0.13325856626033783
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 118, Loss: 0.17181654274463654
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 119, Loss: 0.1226082518696785
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 120, Loss: 0.1593869924545288
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 121, Loss: 0.10797971487045288
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 122, Loss: 0.16917403042316437
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 123, Loss: 0.14143423736095428
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 124, Loss: 0.11902283877134323
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 125, Loss: 0.16363078355789185
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 126, Loss: 0.16362901031970978
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 127, Loss: 0.11851409822702408
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 128, Loss: 0.13773392140865326
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 129, Loss: 0.13645261526107788
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 130, Loss: 0.14280463755130768
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 131, Loss: 0.14065828919410706
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 132, Loss: 0.12983468174934387
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 133, Loss: 0.15395185351371765
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 134, Loss: 0.16198886930942535
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 135, Loss: 0.16509051620960236
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 136, Loss: 0.14227063953876495
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 137, Loss: 0.13983570039272308
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 138, Loss: 0.12312282621860504
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 139, Loss: 0.13110245764255524
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 140, Loss: 0.12231560051441193
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 141, Loss: 0.1533764749765396
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 142, Loss: 0.10605792701244354
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 143, Loss: 0.1638299971818924
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 144, Loss: 0.15137752890586853
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 145, Loss: 0.14874115586280823
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 146, Loss: 0.2652941346168518
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 147, Loss: 0.12956273555755615
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 148, Loss: 0.14363865554332733
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 149, Loss: 0.15029333531856537
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 150, Loss: 0.15038353204727173
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 151, Loss: 0.13474833965301514
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 152, Loss: 0.15018261969089508
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 153, Loss: 0.14005155861377716
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 154, Loss: 0.1583944857120514
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 155, Loss: 0.11823558807373047
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 156, Loss: 0.13695722818374634
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 157, Loss: 0.14551451802253723
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 158, Loss: 0.10490801185369492
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 159, Loss: 0.13896551728248596
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 160, Loss: 0.15343891084194183
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 161, Loss: 0.1442994624376297
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 162, Loss: 0.10980550944805145
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 163, Loss: 0.11072390526533127
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 164, Loss: 0.13176973164081573
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 165, Loss: 0.22966060042381287
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 166, Loss: 0.13685742020606995
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 167, Loss: 0.1240089014172554
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 168, Loss: 0.24139592051506042
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 169, Loss: 0.11770157516002655
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 170, Loss: 0.11615625023841858
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 171, Loss: 0.21392127871513367
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 172, Loss: 0.11698035150766373
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 173, Loss: 0.141929030418396
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 174, Loss: 0.12223180383443832
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 175, Loss: 0.11246342211961746
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 176, Loss: 0.14057759940624237
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 177, Loss: 0.1380946934223175
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 178, Loss: 0.12642188370227814
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 179, Loss: 0.12552320957183838
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 180, Loss: 0.13593614101409912
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 181, Loss: 0.15073099732398987
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 182, Loss: 0.12729015946388245
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 183, Loss: 0.1144237369298935
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 184, Loss: 0.12694774568080902
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 185, Loss: 0.13161973655223846
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 186, Loss: 0.1327483355998993
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 187, Loss: 0.2470763921737671
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 188, Loss: 0.1547519415616989
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 189, Loss: 0.11471209675073624
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 190, Loss: 0.14725741744041443
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 191, Loss: 0.26280784606933594
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 192, Loss: 0.13165809214115143
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 193, Loss: 0.12086006999015808
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 194, Loss: 0.14188016951084137
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 195, Loss: 0.12366227060556412
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 196, Loss: 0.1541568487882614
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 197, Loss: 0.11859222501516342
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 198, Loss: 0.11603959649801254
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 199, Loss: 0.13034868240356445
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 200, Loss: 0.14077012240886688
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 201, Loss: 0.1327538639307022
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 202, Loss: 0.13338309526443481
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 203, Loss: 0.13955232501029968
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 204, Loss: 0.17988379299640656
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 205, Loss: 0.15048547089099884
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 206, Loss: 0.12503361701965332
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 207, Loss: 0.13159680366516113
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 208, Loss: 0.12636658549308777
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 209, Loss: 0.1428237408399582
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 210, Loss: 0.17148104310035706
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 211, Loss: 0.1337239146232605
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 212, Loss: 0.1161479726433754
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 213, Loss: 0.158378466963768
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 214, Loss: 0.12840138375759125
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 215, Loss: 0.25633788108825684
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 216, Loss: 0.11455930024385452
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 217, Loss: 0.1757635474205017
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 218, Loss: 0.12667056918144226
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 219, Loss: 0.13752657175064087
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 220, Loss: 0.11652717739343643
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 221, Loss: 0.11506909877061844
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 222, Loss: 0.1105758547782898
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 223, Loss: 0.12031634896993637
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 224, Loss: 0.15043029189109802
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 225, Loss: 0.12820085883140564
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 226, Loss: 0.11100029945373535
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 227, Loss: 0.16538503766059875
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 228, Loss: 0.12696783244609833
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 229, Loss: 0.24313709139823914
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 230, Loss: 0.13383479416370392
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 231, Loss: 0.10287316143512726
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 232, Loss: 0.12910163402557373
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 233, Loss: 0.1373755782842636
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 234, Loss: 0.13644908368587494
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 235, Loss: 0.17471472918987274
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 236, Loss: 0.16622421145439148
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 237, Loss: 0.1376349776983261
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 238, Loss: 0.19371500611305237
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 239, Loss: 0.14199846982955933
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 240, Loss: 0.12679316103458405
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 241, Loss: 0.11982841044664383
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 242, Loss: 0.13508985936641693
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 243, Loss: 0.13055606186389923
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 244, Loss: 0.12445469945669174
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 245, Loss: 0.13853059709072113
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 246, Loss: 0.11837276071310043
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 247, Loss: 0.11933859437704086
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 248, Loss: 0.16304254531860352
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 249, Loss: 0.12578052282333374
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 250, Loss: 0.12286198139190674
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 251, Loss: 0.1386105865240097
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 252, Loss: 0.12934547662734985
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 253, Loss: 0.13924022018909454
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 254, Loss: 0.17704297602176666
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 255, Loss: 0.12009812891483307
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 256, Loss: 0.1298040896654129
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 257, Loss: 0.13391786813735962
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 258, Loss: 0.12350146472454071
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 259, Loss: 0.12221011519432068
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 260, Loss: 0.11576781421899796
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 261, Loss: 0.16334445774555206
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 262, Loss: 0.10319732874631882
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 263, Loss: 0.17420542240142822
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 264, Loss: 0.12865763902664185
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 265, Loss: 0.12500526010990143
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 266, Loss: 0.12412699311971664
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 267, Loss: 0.12996990978717804
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 268, Loss: 0.11903174221515656
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 269, Loss: 0.15465368330478668
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 270, Loss: 0.1241096630692482
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 271, Loss: 0.16386987268924713
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 272, Loss: 0.11966235935688019
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 273, Loss: 0.15759524703025818
Predicted Boxes: [], Scores: [], Labels: []
Epoch 9, Iteration 274, Loss: 0.14819449186325073
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 0, Loss: 0.12942402064800262
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 1, Loss: 0.11933284252882004
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 2, Loss: 0.1199902594089508
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 3, Loss: 0.13089138269424438
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 4, Loss: 0.11716543138027191
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 5, Loss: 0.12204486131668091
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 6, Loss: 0.22535601258277893
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 7, Loss: 0.13302694261074066
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 8, Loss: 0.1256004273891449
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 9, Loss: 0.12123773247003555
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 10, Loss: 0.12694673240184784
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 11, Loss: 0.1315261572599411
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 12, Loss: 0.10571669787168503
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 13, Loss: 0.185899555683136
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 14, Loss: 0.1438845694065094
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 15, Loss: 0.126299649477005
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 16, Loss: 0.10797397792339325
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 17, Loss: 0.13482671976089478
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 18, Loss: 0.15539248287677765
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 19, Loss: 0.13811084628105164
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 20, Loss: 0.12750767171382904
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 21, Loss: 0.1482664793729782
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 22, Loss: 0.1284390538930893
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 23, Loss: 0.15707971155643463
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 24, Loss: 0.11442558467388153
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 25, Loss: 0.10697545111179352
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 26, Loss: 0.13254567980766296
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 27, Loss: 0.1238754391670227
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 28, Loss: 0.1093500405550003
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 29, Loss: 0.11934205889701843
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 30, Loss: 0.12585116922855377
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 31, Loss: 0.10599472373723984
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 32, Loss: 0.12822555005550385
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 33, Loss: 0.12396740913391113
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 34, Loss: 0.11308646947145462
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 35, Loss: 0.13048569858074188
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 36, Loss: 0.12544485926628113
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 37, Loss: 0.14353300631046295
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 38, Loss: 0.12149360775947571
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 39, Loss: 0.14681370556354523
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 40, Loss: 0.1233721524477005
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 41, Loss: 0.11752695590257645
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 42, Loss: 0.10398206114768982
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 43, Loss: 0.14211846888065338
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 44, Loss: 0.11956676840782166
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 45, Loss: 0.10333990305662155
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 46, Loss: 0.11375357955694199
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 47, Loss: 0.11396338045597076
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 48, Loss: 0.1322111189365387
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 49, Loss: 0.11732449382543564
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 50, Loss: 0.1244647353887558
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 51, Loss: 0.12802143394947052
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 52, Loss: 0.21377503871917725
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 53, Loss: 0.13469938933849335
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 54, Loss: 0.2031339406967163
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 55, Loss: 0.1696569174528122
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 56, Loss: 0.13408343493938446
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 57, Loss: 0.1336798518896103
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 58, Loss: 0.12878099083900452
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 59, Loss: 0.16216352581977844
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 60, Loss: 0.12355178594589233
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 61, Loss: 0.14145079255104065
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 62, Loss: 0.11065901815891266
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 63, Loss: 0.12096304446458817
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 64, Loss: 0.09980310499668121
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 65, Loss: 0.1514844447374344
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 66, Loss: 0.14302746951580048
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 67, Loss: 0.1336396038532257
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 68, Loss: 0.1272188276052475
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 69, Loss: 0.13587592542171478
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 70, Loss: 0.12387146055698395
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 71, Loss: 0.13786932826042175
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 72, Loss: 0.12358871102333069
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 73, Loss: 0.13184437155723572
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 74, Loss: 0.13553307950496674
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 75, Loss: 0.0991806760430336
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 76, Loss: 0.1217794343829155
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 77, Loss: 0.12466590106487274
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 78, Loss: 0.12019268423318863
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 79, Loss: 0.14578965306282043
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 80, Loss: 0.1385544389486313
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 81, Loss: 0.1362282633781433
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 82, Loss: 0.12850674986839294
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 83, Loss: 0.13194526731967926
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 84, Loss: 0.20880404114723206
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 85, Loss: 0.1426568478345871
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 86, Loss: 0.10000994801521301
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 87, Loss: 0.12483114004135132
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 88, Loss: 0.14969271421432495
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 89, Loss: 0.1269487738609314
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 90, Loss: 0.12414748966693878
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 91, Loss: 0.14724546670913696
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 92, Loss: 0.12058999389410019
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 93, Loss: 0.2879297435283661
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 94, Loss: 0.1361948698759079
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 95, Loss: 0.10115455090999603
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 96, Loss: 0.1271805316209793
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 97, Loss: 0.12182365357875824
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 98, Loss: 0.11713189631700516
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 99, Loss: 0.11617748439311981
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 100, Loss: 0.11466976255178452
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 101, Loss: 0.11430466920137405
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 102, Loss: 0.12393089383840561
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 103, Loss: 0.12466653436422348
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 104, Loss: 0.12626709043979645
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 105, Loss: 0.13360761106014252
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 106, Loss: 0.12672896683216095
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 107, Loss: 0.12441441416740417
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 108, Loss: 0.1162857860326767
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 109, Loss: 0.11661133170127869
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 110, Loss: 0.1037500724196434
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 111, Loss: 0.11525101214647293
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 112, Loss: 0.12501409649848938
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 113, Loss: 0.23439541459083557
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 114, Loss: 0.113425113260746
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 115, Loss: 0.09067924320697784
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 116, Loss: 0.12094391137361526
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 117, Loss: 0.1209687888622284
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 118, Loss: 0.14205363392829895
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 119, Loss: 0.14474233984947205
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 120, Loss: 0.12909387052059174
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 121, Loss: 0.1184125617146492
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 122, Loss: 0.11320522427558899
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 123, Loss: 0.09334848821163177
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 124, Loss: 0.1168689876794815
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 125, Loss: 0.1298942118883133
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 126, Loss: 0.20753593742847443
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 127, Loss: 0.12170273810625076
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 128, Loss: 0.12329739332199097
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 129, Loss: 0.09334862232208252
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 130, Loss: 0.11093045026063919
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 131, Loss: 0.12520621716976166
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 132, Loss: 0.1282138228416443
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 133, Loss: 0.12330852448940277
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 134, Loss: 0.10427607595920563
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 135, Loss: 0.09633567929267883
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 136, Loss: 0.1317504346370697
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 137, Loss: 0.1366870254278183
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 138, Loss: 0.0989699438214302
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 139, Loss: 0.11537989974021912
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 140, Loss: 0.10697697848081589
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 141, Loss: 0.12510085105895996
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 142, Loss: 0.12126214057207108
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 143, Loss: 0.11893545836210251
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 144, Loss: 0.11840295791625977
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 145, Loss: 0.10084579139947891
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 146, Loss: 0.12624582648277283
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 147, Loss: 0.12562869489192963
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 148, Loss: 0.11902577430009842
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 149, Loss: 0.14656317234039307
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 150, Loss: 0.11906171590089798
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 151, Loss: 0.1355123072862625
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 152, Loss: 0.10859942436218262
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 153, Loss: 0.1353149265050888
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 154, Loss: 0.17225292325019836
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 155, Loss: 0.12514953315258026
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 156, Loss: 0.12361606955528259
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 157, Loss: 0.1520400047302246
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 158, Loss: 0.11306150257587433
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 159, Loss: 0.12277505546808243
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 160, Loss: 0.22801314294338226
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 161, Loss: 0.13193680346012115
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 162, Loss: 0.14096017181873322
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 163, Loss: 0.10769569873809814
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 164, Loss: 0.13180553913116455
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 165, Loss: 0.12342927604913712
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 166, Loss: 0.11470100283622742
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 167, Loss: 0.10351119190454483
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 168, Loss: 0.13808798789978027
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 169, Loss: 0.10154616832733154
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 170, Loss: 0.14013268053531647
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 171, Loss: 0.1175086721777916
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 172, Loss: 0.12574075162410736
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 173, Loss: 0.10727676749229431
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 174, Loss: 0.0860932469367981
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 175, Loss: 0.11485128849744797
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 176, Loss: 0.11503064632415771
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 177, Loss: 0.1070484071969986
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 178, Loss: 0.23985251784324646
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 179, Loss: 0.11374051123857498
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 180, Loss: 0.11432529985904694
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 181, Loss: 0.1084439754486084
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 182, Loss: 0.1125497967004776
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 183, Loss: 0.1208122968673706
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 184, Loss: 0.1403663456439972
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 185, Loss: 0.21809206902980804
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 186, Loss: 0.11020013689994812
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 187, Loss: 0.10766613483428955
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 188, Loss: 0.09788738936185837
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 189, Loss: 0.1045013964176178
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 190, Loss: 0.2051878124475479
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 191, Loss: 0.11850177496671677
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 192, Loss: 0.20592352747917175
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 193, Loss: 0.14168404042720795
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 194, Loss: 0.1099390983581543
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 195, Loss: 0.1019463986158371
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 196, Loss: 0.1366541087627411
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 197, Loss: 0.11472991108894348
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 198, Loss: 0.11913798749446869
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 199, Loss: 0.10395874828100204
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 200, Loss: 0.12093032896518707
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 201, Loss: 0.09844031184911728
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 202, Loss: 0.1343296617269516
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 203, Loss: 0.11338851600885391
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 204, Loss: 0.1390155851840973
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 205, Loss: 0.11760234087705612
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 206, Loss: 0.10861271619796753
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 207, Loss: 0.1204744428396225
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 208, Loss: 0.13631799817085266
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 209, Loss: 0.10337840765714645
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 210, Loss: 0.12407676875591278
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 211, Loss: 0.06819150596857071
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 212, Loss: 0.11760452389717102
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 213, Loss: 0.13811060786247253
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 214, Loss: 0.13770176470279694
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 215, Loss: 0.12670336663722992
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 216, Loss: 0.11342830210924149
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 217, Loss: 0.14449232816696167
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 218, Loss: 0.15078064799308777
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 219, Loss: 0.10937603563070297
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 220, Loss: 0.13729554414749146
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 221, Loss: 0.1209673061966896
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 222, Loss: 0.14938782155513763
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 223, Loss: 0.11294686049222946
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 224, Loss: 0.09693527221679688
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 225, Loss: 0.10603727400302887
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 226, Loss: 0.10869500786066055
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 227, Loss: 0.12980610132217407
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 228, Loss: 0.11293310672044754
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 229, Loss: 0.11439170688390732
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 230, Loss: 0.11084238439798355
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 231, Loss: 0.11003546416759491
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 232, Loss: 0.16308164596557617
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 233, Loss: 0.11077092587947845
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 234, Loss: 0.1039242371916771
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 235, Loss: 0.11548404395580292
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 236, Loss: 0.10155156254768372
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 237, Loss: 0.11748582124710083
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 238, Loss: 0.11992771178483963
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 239, Loss: 0.14466434717178345
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 240, Loss: 0.09237019717693329
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 241, Loss: 0.08691675215959549
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 242, Loss: 0.1194939985871315
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 243, Loss: 0.10741004347801208
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 244, Loss: 0.1051822155714035
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 245, Loss: 0.1252480298280716
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 246, Loss: 0.12003850936889648
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 247, Loss: 0.11994576454162598
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 248, Loss: 0.11638423800468445
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 249, Loss: 0.12356972694396973
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 250, Loss: 0.13027386367321014
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 251, Loss: 0.10840654373168945
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 252, Loss: 0.1217290610074997
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 253, Loss: 0.11311982572078705
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 254, Loss: 0.11559180170297623
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 255, Loss: 0.09995808452367783
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 256, Loss: 0.12639908492565155
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 257, Loss: 0.12707148492336273
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 258, Loss: 0.12017299234867096
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 259, Loss: 0.12383848428726196
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 260, Loss: 0.13544175028800964
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 261, Loss: 0.1287888139486313
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 262, Loss: 0.1235911101102829
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 263, Loss: 0.10142280906438828
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 264, Loss: 0.11815540492534637
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 265, Loss: 0.10587125271558762
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 266, Loss: 0.19808311760425568
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 267, Loss: 0.09352793544530869
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 268, Loss: 0.1852453649044037
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 269, Loss: 0.13804186880588531
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 270, Loss: 0.11931824684143066
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 271, Loss: 0.11953979730606079
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 272, Loss: 0.11999689042568207
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 273, Loss: 0.1096494123339653
Predicted Boxes: [], Scores: [], Labels: []
Epoch 10, Iteration 274, Loss: 0.2039462924003601
Predicted Boxes: [], Scores: [], Labels: []
```

# **Evaluación y visualización del modelo**

Evaluar el modelo


**[Celda 17 - Código]**
```python
# Modo de evaluación
model.eval()

# Cargar y transformar la imagen de prueba
image_path = "/content/drive/MyDrive/GIBD/E.C de imágenes a color/Faster RCNN/badgesbbfromroboflow/train/camiseta_100_jpg.rf.9349201981071bb61f4ef82b5de83048.jpg"
image = Image.open(image_path).convert("RGB")
image = get_transform(train=False)(image)
image = image.unsqueeze(0)  # Añadir dimensión de batch

# Realizar la predicción
with torch.no_grad():
    prediction = model(image.to(device))

# Visualizar las cajas y las etiquetas predichas
boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()
print(f"Scores: {scores}")  # Revisa las puntuaciones predichas

# Filtrar por un umbral de puntuación (por ejemplo, 0.5) y la clase de interés
threshold = 0.1
#target_label = 1  # Asume que la clase "badge" tiene id 1
filtered_indices = (scores > threshold) #& (labels == target_label)
boxes = boxes[filtered_indices]
labels = labels[filtered_indices]
scores = scores[filtered_indices]

# Mostrar la imagen con las cajas y puntuaciones
plt.figure(figsize=(12, 8))
plt.imshow(image.squeeze(0).permute(1, 2, 0))
for box, score in zip(boxes, scores):
    plt.gca().add_patch(
        plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            edgecolor='red',
            fill=False,
            linewidth=2
        )
    )
    plt.text(
        box[0],
        box[1] - 10,  # Posición del texto por encima de la caja
        f'{score:.2f}',
        color='white',
        fontsize=12,
        bbox=dict(facecolor='red', alpha=0.5)
    )

plt.axis('off')
plt.show()

```


*Salida:*
```text
Scores: []
<Figure size 1200x800 with 1 Axes>
```
