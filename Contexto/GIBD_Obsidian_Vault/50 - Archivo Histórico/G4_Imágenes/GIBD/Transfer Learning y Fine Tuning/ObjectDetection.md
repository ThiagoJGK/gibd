---
aliases: [ObjectDetection]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/notebook
formato_original: ipynb
fecha_modificacion: 2024-08-15
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Transfer Learning y Fine Tuning/Tattoos/Notebooks/ObjectDetection.ipynb"
tamanio_bytes: 64493
---

# Notebook: ObjectDetection.ipynb

Ruta interna: `GIBD/Transfer Learning y Fine Tuning/Tattoos/Notebooks/ObjectDetection.ipynb`

---

***Instalacion de librerias***

!git clone https://github.com/tensorflow/models.git
%cd models/research
!pip install .



**[Celda 3 - Código]**
```python
!pip install tensorflow tensorflow-hub tensorflow-datasets
```


*Salida:*
```text
Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (2.17.0)
Requirement already satisfied: tensorflow-hub in /usr/local/lib/python3.10/dist-packages (0.16.1)
Requirement already satisfied: tensorflow-datasets in /usr/local/lib/python3.10/dist-packages (4.9.6)
Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.3.25)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.6.0)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)
Requirement already satisfied: h5py>=3.10.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.11.0)
Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (18.1.1)
Requirement already satisfied: ml-dtypes<0.5.0,>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.4.0)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.3.0)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.1)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.20.3)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.32.3)
Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow) (71.0.4)
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (4.12.2)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.64.1)
Requirement already satisfied: tensorboard<2.18,>=2.17 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.17.0)
Requirement already satisfied: keras>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.4.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.37.1)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.26.4)
Requirement already satisfied: tf-keras>=2.14.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow-hub) (2.17.0)
Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (8.1.7)
Requirement already satisfied: dm-tree in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (0.1.8)
Requirement already satisfied: immutabledict in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (4.2.0)
Requirement already satisfied: promise in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (2.3)
Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (5.9.5)
Requirement already satisfied: pyarrow in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (14.0.2)
Requirement already satisfied: simple-parsing in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (0.1.5)
Requirement already satisfied: tensorflow-metadata in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (1.15.0)
Requirement already satisfied: toml in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (0.10.2)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (4.66.5)
Requirement already satisfied: array-record>=0.5.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (0.5.1)
Requirement already satisfied: etils>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from etils[enp,epath,epy,etree]>=1.6.0; python_version < "3.11"->tensorflow-datasets) (1.7.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow) (0.44.0)
Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from etils[enp,epath,epy,etree]>=1.6.0; python_version < "3.11"->tensorflow-datasets) (2024.6.1)
Requirement already satisfied: importlib_resources in /usr/local/lib/python3.10/dist-packages (from etils[enp,epath,epy,etree]>=1.6.0; python_version < "3.11"->tensorflow-datasets) (6.4.0)
Requirement already satisfied: zipp in /usr/local/lib/python3.10/dist-packages (from etils[enp,epath,epy,etree]>=1.6.0; python_version < "3.11"->tensorflow-datasets) (3.19.2)
Requirement already satisfied: rich in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (13.7.1)
Requirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.0.8)
Requirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.12.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2024.7.4)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.6)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.0.3)
Requirement already satisfied: docstring-parser~=0.15 in /usr/local/lib/python3.10/dist-packages (from simple-parsing->tensorflow-datasets) (0.16)
Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.18,>=2.17->tensorflow) (2.1.5)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (2.16.1)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.2.0->tensorflow) (0.1.2)
```


**[Celda 4 - Código]**
```python
!pip install opencv-python-headless

```


*Salida:*
```text
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.10/dist-packages (4.10.0.84)
Requirement already satisfied: numpy>=1.21.2 in /usr/local/lib/python3.10/dist-packages (from opencv-python-headless) (1.26.4)
```


**[Celda 5 - Código]**
```python
!pip install tensorflow-object-detection-api
```


*Salida:*
```text
Requirement already satisfied: tensorflow-object-detection-api in /usr/local/lib/python3.10/dist-packages (0.1.1)
Requirement already satisfied: Pillow>=1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (9.4.0)
Requirement already satisfied: Matplotlib>=2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (3.7.1)
Requirement already satisfied: Cython>=0.28.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (3.0.11)
Requirement already satisfied: Protobuf in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (3.20.3)
Requirement already satisfied: lxml in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (4.9.4)
Requirement already satisfied: jupyter in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (1.0.0)
Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (2.17.0)
Requirement already satisfied: contextlib2 in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (21.6.0)
Requirement already satisfied: wheel in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (0.44.0)
Requirement already satisfied: twine in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (5.1.1)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (1.2.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (4.53.1)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (1.4.5)
Requirement already satisfied: numpy>=1.20 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (1.26.4)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (24.1)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (3.1.2)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (2.8.2)
Requirement already satisfied: notebook in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (6.5.5)
Requirement already satisfied: qtconsole in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (5.5.2)
Requirement already satisfied: jupyter-console in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (6.1.0)
Requirement already satisfied: nbconvert in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (6.5.4)
Requirement already satisfied: ipykernel in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (5.5.6)
Requirement already satisfied: ipywidgets in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (7.7.1)
Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (1.6.3)
Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (24.3.25)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (0.6.0)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (0.2.0)
Requirement already satisfied: h5py>=3.10.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (3.11.0)
Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (18.1.1)
Requirement already satisfied: ml-dtypes<0.5.0,>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (0.4.0)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (3.3.0)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (2.32.3)
Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (71.0.4)
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (4.12.2)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (1.64.1)
Requirement already satisfied: tensorboard<2.18,>=2.17 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (2.17.0)
Requirement already satisfied: keras>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (3.4.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (0.37.1)
Requirement already satisfied: pkginfo>=1.8.1 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (1.10.0)
Requirement already satisfied: readme-renderer>=35.0 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (44.0)
Requirement already satisfied: requests-toolbelt!=0.9.0,>=0.8.0 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (1.0.0)
Requirement already satisfied: urllib3>=1.26.0 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (2.0.7)
Requirement already satisfied: importlib-metadata>=3.6 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (8.2.0)
Requirement already satisfied: keyring>=15.1 in /usr/lib/python3/dist-packages (from twine->tensorflow-object-detection-api) (23.5.0)
Requirement already satisfied: rfc3986>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (2.0.0)
Requirement already satisfied: rich>=12.0.0 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (13.7.1)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata>=3.6->twine->tensorflow-object-detection-api) (3.19.2)
Requirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow->tensorflow-object-detection-api) (0.0.8)
Requirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow->tensorflow-object-detection-api) (0.12.1)
Requirement already satisfied: nh3>=0.2.14 in /usr/local/lib/python3.10/dist-packages (from readme-renderer>=35.0->twine->tensorflow-object-detection-api) (0.2.18)
Collecting docutils>=0.21.2 (from readme-renderer>=35.0->twine->tensorflow-object-detection-api)
  Using cached docutils-0.21.2-py3-none-any.whl.metadata (2.8 kB)
Requirement already satisfied: Pygments>=2.5.1 in /usr/local/lib/python3.10/dist-packages (from readme-renderer>=35.0->twine->tensorflow-object-detection-api) (2.16.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow->tensorflow-object-detection-api) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow->tensorflow-object-detection-api) (3.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow->tensorflow-object-detection-api) (2024.7.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=12.0.0->twine->tensorflow-object-detection-api) (3.0.0)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow->tensorflow-object-detection-api) (3.6)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow->tensorflow-object-detection-api) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow->tensorflow-object-detection-api) (3.0.3)
Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.10/dist-packages (from ipykernel->jupyter->tensorflow-object-detection-api) (0.2.0)
Requirement already satisfied: ipython>=5.0.0 in /usr/local/lib/python3.10/dist-packages (from ipykernel->jupyter->tensorflow-object-detection-api) (7.34.0)
Requirement already satisfied: traitlets>=4.1.0 in /usr/local/lib/python3.10/dist-packages (from ipykernel->jupyter->tensorflow-object-detection-api) (5.7.1)
Requirement already satisfied: jupyter-client in /usr/local/lib/python3.10/dist-packages (from ipykernel->jupyter->tensorflow-object-detection-api) (6.1.12)
Requirement already satisfied: tornado>=4.2 in /usr/local/lib/python3.10/dist-packages (from ipykernel->jupyter->tensorflow-object-detection-api) (6.3.3)
Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets->jupyter->tensorflow-object-detection-api) (3.6.8)
Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets->jupyter->tensorflow-object-detection-api) (3.0.11)
Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from jupyter-console->jupyter->tensorflow-object-detection-api) (3.0.47)
Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (4.12.3)
Requirement already satisfied: bleach in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (6.1.0)
Requirement already satisfied: defusedxml in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (0.7.1)
Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (0.4)
Requirement already satisfied: jinja2>=3.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (3.1.4)
Requirement already satisfied: jupyter-core>=4.7 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (5.7.2)
Requirement already satisfied: jupyterlab-pygments in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (0.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (2.1.5)
Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (0.8.4)
Requirement already satisfied: nbclient>=0.5.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (0.10.0)
Requirement already satisfied: nbformat>=5.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (5.10.4)
Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (1.5.1)
Requirement already satisfied: tinycss2 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (1.3.0)
Requirement already satisfied: pyzmq<25,>=17 in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (24.0.1)
Requirement already satisfied: argon2-cffi in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (23.1.0)
Requirement already satisfied: nest-asyncio>=1.5 in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (1.6.0)
Requirement already satisfied: Send2Trash>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (1.8.3)
Requirement already satisfied: terminado>=0.8.3 in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (0.18.1)
Requirement already satisfied: prometheus-client in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (0.20.0)
Requirement already satisfied: nbclassic>=0.4.7 in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (1.1.0)
Requirement already satisfied: qtpy>=2.4.0 in /usr/local/lib/python3.10/dist-packages (from qtconsole->jupyter->tensorflow-object-detection-api) (2.4.1)
Requirement already satisfied: jedi>=0.16 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (0.19.1)
Requirement already satisfied: decorator in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (4.4.2)
Requirement already satisfied: pickleshare in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (0.7.5)
Requirement already satisfied: backcall in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (0.2.0)
Requirement already satisfied: matplotlib-inline in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (0.1.7)
Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (4.9.0)
Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.10/dist-packages (from jupyter-core>=4.7->nbconvert->jupyter->tensorflow-object-detection-api) (4.2.2)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=12.0.0->twine->tensorflow-object-detection-api) (0.1.2)
Requirement already satisfied: notebook-shim>=0.2.3 in /usr/local/lib/python3.10/dist-packages (from nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (0.2.4)
Requirement already satisfied: fastjsonschema>=2.15 in /usr/local/lib/python3.10/dist-packages (from nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (2.20.0)
Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.10/dist-packages (from nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (4.23.0)
Requirement already satisfied: wcwidth in /usr/local/lib/python3.10/dist-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->jupyter-console->jupyter->tensorflow-object-detection-api) (0.2.13)
Requirement already satisfied: ptyprocess in /usr/local/lib/python3.10/dist-packages (from terminado>=0.8.3->notebook->jupyter->tensorflow-object-detection-api) (0.7.0)
Requirement already satisfied: argon2-cffi-bindings in /usr/local/lib/python3.10/dist-packages (from argon2-cffi->notebook->jupyter->tensorflow-object-detection-api) (21.2.0)
Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->nbconvert->jupyter->tensorflow-object-detection-api) (2.5)
Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach->nbconvert->jupyter->tensorflow-object-detection-api) (0.5.1)
Requirement already satisfied: parso<0.9.0,>=0.8.3 in /usr/local/lib/python3.10/dist-packages (from jedi>=0.16->ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (0.8.4)
Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (24.2.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (2023.12.1)
Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (0.35.1)
Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (0.20.0)
Requirement already satisfied: jupyter-server<3,>=1.8 in /usr/local/lib/python3.10/dist-packages (from notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (1.24.0)
Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from argon2-cffi-bindings->argon2-cffi->notebook->jupyter->tensorflow-object-detection-api) (1.17.0)
Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook->jupyter->tensorflow-object-detection-api) (2.22)
Requirement already satisfied: anyio<4,>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (3.7.1)
Requirement already satisfied: websocket-client in /usr/local/lib/python3.10/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (1.8.0)
Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio<4,>=3.1.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (1.3.1)
Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<4,>=3.1.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (1.2.2)
Using cached docutils-0.21.2-py3-none-any.whl (587 kB)
Installing collected packages: docutils
  Attempting uninstall: docutils
    Found existing installation: docutils 0.18.1
    Uninstalling docutils-0.18.1:
      Successfully uninstalled docutils-0.18.1
[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
sphinx 5.0.2 requires docutils<0.19,>=0.14, but you have docutils 0.21.2 which is incompatible.[0m[31m
[0mSuccessfully installed docutils-0.21.2
```


**[Celda 6 - Código]**
```python
!pip install tensorflow tensorflow-hub tensorflow-datasets opencv-python-headless

```


*Salida:*
```text
Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (2.17.0)
Requirement already satisfied: tensorflow-hub in /usr/local/lib/python3.10/dist-packages (0.16.1)
Requirement already satisfied: tensorflow-datasets in /usr/local/lib/python3.10/dist-packages (4.9.6)
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.10/dist-packages (4.10.0.84)
Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.3.25)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.6.0)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.2.0)
Requirement already satisfied: h5py>=3.10.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.11.0)
Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (18.1.1)
Requirement already satisfied: ml-dtypes<0.5.0,>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.4.0)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.3.0)
Requirement already satisfied: packaging in /usr/local/lib/python3.10/dist-packages (from tensorflow) (24.1)
Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.20.3)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.32.3)
Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow) (71.0.4)
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (4.12.2)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.64.1)
Requirement already satisfied: tensorboard<2.18,>=2.17 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (2.17.0)
Requirement already satisfied: keras>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (3.4.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (0.37.1)
Requirement already satisfied: numpy<2.0.0,>=1.23.5 in /usr/local/lib/python3.10/dist-packages (from tensorflow) (1.26.4)
Requirement already satisfied: tf-keras>=2.14.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow-hub) (2.17.0)
Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (8.1.7)
Requirement already satisfied: dm-tree in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (0.1.8)
Requirement already satisfied: immutabledict in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (4.2.0)
Requirement already satisfied: promise in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (2.3)
Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (5.9.5)
Requirement already satisfied: pyarrow in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (14.0.2)
Requirement already satisfied: simple-parsing in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (0.1.5)
Requirement already satisfied: tensorflow-metadata in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (1.15.0)
Requirement already satisfied: toml in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (0.10.2)
Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (4.66.5)
Requirement already satisfied: array-record>=0.5.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow-datasets) (0.5.1)
Requirement already satisfied: etils>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from etils[enp,epath,epy,etree]>=1.6.0; python_version < "3.11"->tensorflow-datasets) (1.7.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.10/dist-packages (from astunparse>=1.6.0->tensorflow) (0.44.0)
Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from etils[enp,epath,epy,etree]>=1.6.0; python_version < "3.11"->tensorflow-datasets) (2024.6.1)
Requirement already satisfied: importlib_resources in /usr/local/lib/python3.10/dist-packages (from etils[enp,epath,epy,etree]>=1.6.0; python_version < "3.11"->tensorflow-datasets) (6.4.0)
Requirement already satisfied: zipp in /usr/local/lib/python3.10/dist-packages (from etils[enp,epath,epy,etree]>=1.6.0; python_version < "3.11"->tensorflow-datasets) (3.19.2)
Requirement already satisfied: rich in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (13.7.1)
Requirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.0.8)
Requirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow) (0.12.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (3.7)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2.0.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow) (2024.7.4)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.6)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow) (3.0.3)
Requirement already satisfied: docstring-parser~=0.15 in /usr/local/lib/python3.10/dist-packages (from simple-parsing->tensorflow-datasets) (0.16)
Requirement already satisfied: MarkupSafe>=2.1.1 in /usr/local/lib/python3.10/dist-packages (from werkzeug>=1.0.1->tensorboard<2.18,>=2.17->tensorflow) (2.1.5)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (3.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich->keras>=3.2.0->tensorflow) (2.16.1)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich->keras>=3.2.0->tensorflow) (0.1.2)
```


**[Celda 7 - Código]**
```python
!pip install tensorflow-object-detection-api

```


*Salida:*
```text
Requirement already satisfied: tensorflow-object-detection-api in /usr/local/lib/python3.10/dist-packages (0.1.1)
Requirement already satisfied: Pillow>=1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (9.4.0)
Requirement already satisfied: Matplotlib>=2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (3.7.1)
Requirement already satisfied: Cython>=0.28.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (3.0.11)
Requirement already satisfied: Protobuf in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (3.20.3)
Requirement already satisfied: lxml in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (4.9.4)
Requirement already satisfied: jupyter in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (1.0.0)
Requirement already satisfied: tensorflow in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (2.17.0)
Requirement already satisfied: contextlib2 in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (21.6.0)
Requirement already satisfied: wheel in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (0.44.0)
Requirement already satisfied: twine in /usr/local/lib/python3.10/dist-packages (from tensorflow-object-detection-api) (5.1.1)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (1.2.1)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (4.53.1)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (1.4.5)
Requirement already satisfied: numpy>=1.20 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (1.26.4)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (24.1)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (3.1.2)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from Matplotlib>=2.1->tensorflow-object-detection-api) (2.8.2)
Requirement already satisfied: notebook in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (6.5.5)
Requirement already satisfied: qtconsole in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (5.5.2)
Requirement already satisfied: jupyter-console in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (6.1.0)
Requirement already satisfied: nbconvert in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (6.5.4)
Requirement already satisfied: ipykernel in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (5.5.6)
Requirement already satisfied: ipywidgets in /usr/local/lib/python3.10/dist-packages (from jupyter->tensorflow-object-detection-api) (7.7.1)
Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (1.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (1.6.3)
Requirement already satisfied: flatbuffers>=24.3.25 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (24.3.25)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (0.6.0)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (0.2.0)
Requirement already satisfied: h5py>=3.10.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (3.11.0)
Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (18.1.1)
Requirement already satisfied: ml-dtypes<0.5.0,>=0.3.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (0.4.0)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (3.3.0)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (2.32.3)
Requirement already satisfied: setuptools in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (71.0.4)
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (1.16.0)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (2.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (4.12.2)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (1.16.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (1.64.1)
Requirement already satisfied: tensorboard<2.18,>=2.17 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (2.17.0)
Requirement already satisfied: keras>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (3.4.1)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.10/dist-packages (from tensorflow->tensorflow-object-detection-api) (0.37.1)
Requirement already satisfied: pkginfo>=1.8.1 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (1.10.0)
Requirement already satisfied: readme-renderer>=35.0 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (44.0)
Requirement already satisfied: requests-toolbelt!=0.9.0,>=0.8.0 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (1.0.0)
Requirement already satisfied: urllib3>=1.26.0 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (2.0.7)
Requirement already satisfied: importlib-metadata>=3.6 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (8.2.0)
Requirement already satisfied: keyring>=15.1 in /usr/lib/python3/dist-packages (from twine->tensorflow-object-detection-api) (23.5.0)
Requirement already satisfied: rfc3986>=1.4.0 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (2.0.0)
Requirement already satisfied: rich>=12.0.0 in /usr/local/lib/python3.10/dist-packages (from twine->tensorflow-object-detection-api) (13.7.1)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata>=3.6->twine->tensorflow-object-detection-api) (3.19.2)
Requirement already satisfied: namex in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow->tensorflow-object-detection-api) (0.0.8)
Requirement already satisfied: optree in /usr/local/lib/python3.10/dist-packages (from keras>=3.2.0->tensorflow->tensorflow-object-detection-api) (0.12.1)
Requirement already satisfied: nh3>=0.2.14 in /usr/local/lib/python3.10/dist-packages (from readme-renderer>=35.0->twine->tensorflow-object-detection-api) (0.2.18)
Requirement already satisfied: docutils>=0.21.2 in /usr/local/lib/python3.10/dist-packages (from readme-renderer>=35.0->twine->tensorflow-object-detection-api) (0.21.2)
Requirement already satisfied: Pygments>=2.5.1 in /usr/local/lib/python3.10/dist-packages (from readme-renderer>=35.0->twine->tensorflow-object-detection-api) (2.16.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow->tensorflow-object-detection-api) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow->tensorflow-object-detection-api) (3.7)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests<3,>=2.21.0->tensorflow->tensorflow-object-detection-api) (2024.7.4)
Requirement already satisfied: markdown-it-py>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=12.0.0->twine->tensorflow-object-detection-api) (3.0.0)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow->tensorflow-object-detection-api) (3.6)
Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow->tensorflow-object-detection-api) (0.7.2)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from tensorboard<2.18,>=2.17->tensorflow->tensorflow-object-detection-api) (3.0.3)
Requirement already satisfied: ipython-genutils in /usr/local/lib/python3.10/dist-packages (from ipykernel->jupyter->tensorflow-object-detection-api) (0.2.0)
Requirement already satisfied: ipython>=5.0.0 in /usr/local/lib/python3.10/dist-packages (from ipykernel->jupyter->tensorflow-object-detection-api) (7.34.0)
Requirement already satisfied: traitlets>=4.1.0 in /usr/local/lib/python3.10/dist-packages (from ipykernel->jupyter->tensorflow-object-detection-api) (5.7.1)
Requirement already satisfied: jupyter-client in /usr/local/lib/python3.10/dist-packages (from ipykernel->jupyter->tensorflow-object-detection-api) (6.1.12)
Requirement already satisfied: tornado>=4.2 in /usr/local/lib/python3.10/dist-packages (from ipykernel->jupyter->tensorflow-object-detection-api) (6.3.3)
Requirement already satisfied: widgetsnbextension~=3.6.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets->jupyter->tensorflow-object-detection-api) (3.6.8)
Requirement already satisfied: jupyterlab-widgets>=1.0.0 in /usr/local/lib/python3.10/dist-packages (from ipywidgets->jupyter->tensorflow-object-detection-api) (3.0.11)
Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from jupyter-console->jupyter->tensorflow-object-detection-api) (3.0.47)
Requirement already satisfied: beautifulsoup4 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (4.12.3)
Requirement already satisfied: bleach in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (6.1.0)
Requirement already satisfied: defusedxml in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (0.7.1)
Requirement already satisfied: entrypoints>=0.2.2 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (0.4)
Requirement already satisfied: jinja2>=3.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (3.1.4)
Requirement already satisfied: jupyter-core>=4.7 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (5.7.2)
Requirement already satisfied: jupyterlab-pygments in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (0.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (2.1.5)
Requirement already satisfied: mistune<2,>=0.8.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (0.8.4)
Requirement already satisfied: nbclient>=0.5.0 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (0.10.0)
Requirement already satisfied: nbformat>=5.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (5.10.4)
Requirement already satisfied: pandocfilters>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (1.5.1)
Requirement already satisfied: tinycss2 in /usr/local/lib/python3.10/dist-packages (from nbconvert->jupyter->tensorflow-object-detection-api) (1.3.0)
Requirement already satisfied: pyzmq<25,>=17 in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (24.0.1)
Requirement already satisfied: argon2-cffi in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (23.1.0)
Requirement already satisfied: nest-asyncio>=1.5 in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (1.6.0)
Requirement already satisfied: Send2Trash>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (1.8.3)
Requirement already satisfied: terminado>=0.8.3 in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (0.18.1)
Requirement already satisfied: prometheus-client in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (0.20.0)
Requirement already satisfied: nbclassic>=0.4.7 in /usr/local/lib/python3.10/dist-packages (from notebook->jupyter->tensorflow-object-detection-api) (1.1.0)
Requirement already satisfied: qtpy>=2.4.0 in /usr/local/lib/python3.10/dist-packages (from qtconsole->jupyter->tensorflow-object-detection-api) (2.4.1)
Requirement already satisfied: jedi>=0.16 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (0.19.1)
Requirement already satisfied: decorator in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (4.4.2)
Requirement already satisfied: pickleshare in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (0.7.5)
Requirement already satisfied: backcall in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (0.2.0)
Requirement already satisfied: matplotlib-inline in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (0.1.7)
Requirement already satisfied: pexpect>4.3 in /usr/local/lib/python3.10/dist-packages (from ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (4.9.0)
Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.10/dist-packages (from jupyter-core>=4.7->nbconvert->jupyter->tensorflow-object-detection-api) (4.2.2)
Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py>=2.2.0->rich>=12.0.0->twine->tensorflow-object-detection-api) (0.1.2)
Requirement already satisfied: notebook-shim>=0.2.3 in /usr/local/lib/python3.10/dist-packages (from nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (0.2.4)
Requirement already satisfied: fastjsonschema>=2.15 in /usr/local/lib/python3.10/dist-packages (from nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (2.20.0)
Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.10/dist-packages (from nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (4.23.0)
Requirement already satisfied: wcwidth in /usr/local/lib/python3.10/dist-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->jupyter-console->jupyter->tensorflow-object-detection-api) (0.2.13)
Requirement already satisfied: ptyprocess in /usr/local/lib/python3.10/dist-packages (from terminado>=0.8.3->notebook->jupyter->tensorflow-object-detection-api) (0.7.0)
Requirement already satisfied: argon2-cffi-bindings in /usr/local/lib/python3.10/dist-packages (from argon2-cffi->notebook->jupyter->tensorflow-object-detection-api) (21.2.0)
Requirement already satisfied: soupsieve>1.2 in /usr/local/lib/python3.10/dist-packages (from beautifulsoup4->nbconvert->jupyter->tensorflow-object-detection-api) (2.5)
Requirement already satisfied: webencodings in /usr/local/lib/python3.10/dist-packages (from bleach->nbconvert->jupyter->tensorflow-object-detection-api) (0.5.1)
Requirement already satisfied: parso<0.9.0,>=0.8.3 in /usr/local/lib/python3.10/dist-packages (from jedi>=0.16->ipython>=5.0.0->ipykernel->jupyter->tensorflow-object-detection-api) (0.8.4)
Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (24.2.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (2023.12.1)
Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (0.35.1)
Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=2.6->nbformat>=5.1->nbconvert->jupyter->tensorflow-object-detection-api) (0.20.0)
Requirement already satisfied: jupyter-server<3,>=1.8 in /usr/local/lib/python3.10/dist-packages (from notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (1.24.0)
Requirement already satisfied: cffi>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from argon2-cffi-bindings->argon2-cffi->notebook->jupyter->tensorflow-object-detection-api) (1.17.0)
Requirement already satisfied: pycparser in /usr/local/lib/python3.10/dist-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi->notebook->jupyter->tensorflow-object-detection-api) (2.22)
Requirement already satisfied: anyio<4,>=3.1.0 in /usr/local/lib/python3.10/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (3.7.1)
Requirement already satisfied: websocket-client in /usr/local/lib/python3.10/dist-packages (from jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (1.8.0)
Requirement already satisfied: sniffio>=1.1 in /usr/local/lib/python3.10/dist-packages (from anyio<4,>=3.1.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (1.3.1)
Requirement already satisfied: exceptiongroup in /usr/local/lib/python3.10/dist-packages (from anyio<4,>=3.1.0->jupyter-server<3,>=1.8->notebook-shim>=0.2.3->nbclassic>=0.4.7->notebook->jupyter->tensorflow-object-detection-api) (1.2.2)
```


**[Celda 8 - Código]**
```python
!pip install docutils==0.18.1

```


*Salida:*
```text
Collecting docutils==0.18.1
  Using cached docutils-0.18.1-py2.py3-none-any.whl.metadata (2.9 kB)
Using cached docutils-0.18.1-py2.py3-none-any.whl (570 kB)
Installing collected packages: docutils
  Attempting uninstall: docutils
    Found existing installation: docutils 0.21.2
    Uninstalling docutils-0.21.2:
      Successfully uninstalled docutils-0.21.2
[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
readme-renderer 44.0 requires docutils>=0.21.2, but you have docutils 0.18.1 which is incompatible.[0m[31m
[0mSuccessfully installed docutils-0.18.1
```


**[Celda 9 - Código]**
```python
from google.colab import drive
drive.mount('/content/drive')

```


**[Celda 10 - Código]**
```python
!cp /content/drive/MyDrive/tfrecords/train.tfrecord /content/

```


**[Celda 11 - Código]**
```python
import tensorflow as tf

def parse_tfrecord(example_proto):
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        'image/object/class/text': tf.io.VarLenFeature(tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }

    parsed_features = tf.io.parse_single_example(example_proto, feature_description)

    image = tf.image.decode_jpeg(parsed_features['image/encoded'])
    image = tf.image.resize(image, [416, 416])

    bbox = tf.stack([
        parsed_features['image/object/bbox/xmin'].values,
        parsed_features['image/object/bbox/xmax'].values,
        parsed_features['image/object/bbox/ymin'].values,
        parsed_features['image/object/bbox/ymax'].values
    ], axis=1)

    labels = parsed_features['image/object/class/label'].values

    return image, (bbox, labels)

def load_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)
    return dataset

# Load the dataset
dataset = load_dataset('/content/train.tfrecord')

```


**[Celda 12 - Código]**
```python
!python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=/content/models/research/object_detection/samples/configs/ssd_mobilenet_v2_fpnlite_320x320.config \
    --model_dir=/content/training/ \
    --alsologtostderr

```


**[Celda 13 - Código]**
```python
!python models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=/content/models/research/object_detection/samples/configs/ssd_mobilenet_v2_fpnlite_320x320.config \
    --model_dir=/content/training/ \
    --checkpoint_dir=/content/training/ \
    --eval_dir=/content/evaluation/ \
    --alsologtostderr

```


**[Celda 14 - Código]**
```python
!python models/research/object_detection/exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path /content/models/research/object_detection/samples/configs/ssd_mobilenet_v2_fpnlite_320x320.config \
    --trained_checkpoint_dir /content/training/ \
    --output_directory /content/exported_model/

```


**[Celda 15 - Código]**
```python
import matplotlib.pyplot as plt

def display_image(image_np):
    plt.figure(figsize=(12, 8))
    plt.imshow(image_np)
    plt.show()

# Test the model
image_np = np.array(tf.image.decode_jpeg(tf.io.read_file('/content/test_image.jpg')))
output_dict = run_inference_for_single_image(image_np)
display_image(image_np)

```
