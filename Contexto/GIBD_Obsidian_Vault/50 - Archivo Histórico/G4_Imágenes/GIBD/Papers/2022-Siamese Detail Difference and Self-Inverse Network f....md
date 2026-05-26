---
aliases: [2022-Siamese Detail Difference and Self-Inverse Network f...]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2023-03-28
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Papers/2022-Siamese Detail Difference and Self-Inverse Network for Forest Cover Change Extraction Based on Landsat 8 OLI Satellite Images.pdf"
tamanio_bytes: 13195154
---

# 2022-Siamese Detail Difference and Self-Inverse Network for Forest Cover Change Extraction Based on Landsat 8 OLI Satellite Images

Ruta interna: `GIBD/Papers/2022-Siamese Detail Difference and Self-Inverse Network for Forest Cover Change Extraction Based on Landsat 8 OLI Satellite Images.pdf`

---

/gid00030/gid00035/gid00032/gid00030/gid00038/gid00001/gid00033/gid00042/gid00045/gid00001
/gid00048/gid00043/gid00031/gid00028/gid00047/gid00032/gid00046
Citation: Guo, Y.; Long, T.; Jiao, W.;
Zhang, X.; He, G.; Wang, W.; Peng, Y.;
Xiao, H. Siamese Detail Difference
and Self-Inverse Network for Forest
Cover Change Extraction Based on
Landsat 8 OLI Satellite Images.
Remote Sens. 2022, 14, 627. https://
doi.org/10.3390/rs14030627
Academic Editor: Gang Chen
Received: 29 November 2021
Accepted: 23 January 2022
Published: 28 January 2022
Publisher’s Note: MDPI stays neutral
with regard to jurisdictional claims in
published maps and institutional afﬁl-
iations.
Copyright: © 2022 by the authors.
Licensee MDPI, Basel, Switzerland.
This article is an open access article
distributed under the terms and
conditions of the Creative Commons
Attribution (CC BY) license (https://
creativecommons.org/licenses/by/
4.0/).
remote sensing  
Article
Siamese Detail Difference and Self-Inverse Network for Forest
Cover Change Extraction Based on Landsat 8 OLI
Satellite Images
Yantao Guo 1,2
 , Tengfei Long 1,3
 , Weili Jiao 1,3,*, Xiaomei Zhang 1,3, Guojin He 1,2,3, Wei Wang1,3, Yan Peng 1,2,3
and Han Xiao 1,2
1 Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing 100094, China;
guoyt@radi.ac.cn (Y.G.); longtf@radi.ac.cn (T.L.); zhangxm@radi.ac.cn (X.Z.); hegj@radi.ac.cn (G.H.);
wangwei@radi.ac.cn (W.W.); pengyan@radi.ac.cn (Y.P .); xiaohan@aircas.ac.cn (H.X.)
2 University of Chinese Academy of Sciences, Beijing 100049, China
3 Key Laboratory of Earth Observation of Hainan Province, Hainan Research Institute, Aerospace Information
Research Institute, Chinese Academy of Sciences, Sanya 572000, China
* Correspondence: jiaowl@aircas.ac.cn; Tel.: +86-010-8217-8191
Abstract: In the context of carbon neutrality, forest cover change detection has become a key topic of
global environmental monitoring. As a large-scale monitoring technique, remote sensing has received
obvious attention in various land cover observation applications. With the rapid development of
deep learning, remote sensing change detection combined with deep neural network has achieved
high accuracy. In this paper, the deep neural network is used to study forest cover change with
Landsat images. The main research ideas are as follows. (1) A Siamese detail difference neural
network is proposed, which uses a combination of concatenate weight sharing mode and subtract
weight sharing mode to improve the accuracy of forest cover change detection. (2) The self-inverse
network is introduced to detect the change of forest increase by using the sample data set of forest
decrease, which realizes the transfer learning of the sample data set and improves the utilization rate
of the sample data set. The experimental results on Landsat 8 images show that the proposed method
outperforms several Siamese neural network methods in forest cover change extraction.
Keywords: forest cover change extraction; deep convolutional neural network; Siamese difference
neural network; self-inverse network
1. Introduction
Changes in forest cover affect the delivery of important ecosystem services, including
biodiversity richness climate regulation, carbon storage, and water supplies [ 1,2]. As a
considerable form of forest information extraction, forest cover change means transition
between land with trees and land without trees [ 3]. Forest cover change mapping has
important research value and economic beneﬁts according to climate and carbon-cycle
modeling [4,5], hydrological studies [6], habitat analyses [7–9], biological conservation [10],
and land-use planning [11,12].
Remote sensing observation is a timely and accurate means to detect forest cover
change on a large scale [ 13,14]. Compared with traditional ﬁeld forestry investigation,
remote sensing has an advantage of larger observation range and longer detection time
span. So far, many researches on forest cover change mapping have been extensively
carried out [15,16]. In the ﬁeld of optical image, Kim et al. used the circa-1990 epoch of
the Global Land Survey collection of Landsat images to detect forest cover change from
1990 to 2000 [17], and the results obtained 93% accuracy for forest cover and 84% for forest
cover change. In the ﬁeld of hyperspectral image, Huang et al. used 500 m MODIS time
series images and a distance metric-based method to detect forest cover change in Paciﬁc
Remote Sens. 2022, 14, 627. https://doi.org/10.3390/rs14030627 https://www.mdpi.com/journal/remotesensing
Remote Sens. 2022, 14, 627 2 of 20
Northwest region of the United States and tropical forests of the Xingu River Basin in Mato
Grosso, Brazil [18]. Over 80 % pixels with 20% deforested area were correctly identiﬁed.
In the ﬁeld of synthetic aperture radar (SAR), Qin et al. [19] used the integration of the L-
band Advanced Land Observation Satellite (ALOS) PALSAR Fine Beam Dual Polarization
(FBD) mosaic dataset and Landsat images to map annual forests in sub-humid and semi-
arid regions. The overall accuracy and Kappa coefﬁcient of the PALSAR/Landsat forest
map were nearly 88.2% and 0.75, respectively, in 2010. The accuracy of forest mapping has
been improved as the development of earth observation technology and change information
extraction technology.
The methods of forest cover change extraction can be divided into four categories:
threshold segmentation, vegetation index segmentation, object-oriented segmentation and
machine learning segmentation methods. The threshold segmentation method generates
different change extraction levels using vegetable sensitive spectral bands [20]. The key
point of the threshold segmentation method is to select the boundary value of the forest
and other ground objects in the spectrum and index. The vegetation index segmentation
method is a traditional way for detecting change information by establishing a moni-
toring index, such as the Enhanced Vegetation Index (EVI) [21–23], Global Environment
Monitoring Index (GEMI) [24], Normalized Difference Fraction Index (NDFI) [25,26], Nor-
malized Difference Moisture Index (NDMI) [ 27–29], Normalized Difference Vegetation
Index (NDVI) [30,31], and Soil Adjusted Vegetation Index (SAVI) [ 32]. However, some
vegetation indexes are particularly sensitive to observation frequency when forest cover
change occurred gradually, and when change is abrupt the others are sensitive to obser-
vation frequency [33]. Different from pixel-based change detection, the object-oriented
methods are usually used to extract forest cover change by reducing small spurious changes
in scattered pixels [34]. The object-oriented methods require a lot of time for image segmen-
tation and expert experience for image analysis [35,36]. The traditional machine learning
segmentation method uses a wealth of training samples to train different types of classiﬁers
for forest cover change detection, which include maximum likelihood [37], support vector
machine [38,39], decision tree [40–42], random forest [43] and neural network (NN) [44,45].
These methods have the advantages of automatic, efﬁcient target classiﬁcation ability and
require less manual labor [46]. While the machine learning segmentation method facilitates
forest cover change extraction, there are still challenges in extracting complex change
targets and context information [47].
Compared with above approaches, the deep neural network has the advantage of rec-
ognizing spectral and spatial features in remote sensing image synchronously. In the ﬁelds
of various remote sensing image processing, deep neural network is a popular method
used in target detection and image classiﬁcation [ 48]. Along with the development of
deep neural networks, diversiﬁed techniques are proposed to improve and extend the
application of remote sensing [49–51]. As the popular architectures used in change detec-
tion, convolutional network and residual network have respective beneﬁts and limitations.
Fully convolutional network (FCN) is an architecture of convolutional neural network
(CNN) and one of the most powerful algorithms for change detection of optical images [52].
However, the popular CNN-based segmentation methods, such as FCN [53], U-Net [54]
and DeepLab [55], are usually developed from computer vision benchmarks composed of
small-scale, high-resolution images. Timilsina, S et al. [56] trained an object-based convolu-
tional neural network by object based image analysis thresholds and mapping tree cover
changes between 2005 and 2015/16 images. This research generated tree training samples
from RGB bands using only the threshold of canopy height model with manual editing.
Bragagnolo, L et al. [ 57] compared U-Net with other state-of-the-art FCN architectures
for Amazon forest cover change mapping. In this study, U-Nets achieved superior classi-
ﬁcation performance and could track forest cover changes from multi-temporal satellite
imagery. Pablo Pozzobon de Bem et al. [ 58] used SharpMusk, U-Net, ResUnet, random
forest and multilayer perception to classify the deforestation in the Brazilian Amazon with
Landsat data. This research generated 844 samples through 200 × 200 pixel windows with
Remote Sens. 2022, 14, 627 3 of 20
10-pixel overlap on image side. Compared among machine learning models, deep learn-
ing provided better classiﬁcations and have less process to remove noise in the research.
The FCN architecture is more suitable for single temporal image segmentation, rather than
bi-temporal image change detection. FCN architecture utilizes different weights to extract
features for bi-temporal images in change detection. In fact, feature extraction process for
same type bi-temporal images is identical. The feature extraction method for FCN causes
more parameters of architecture and more samples to train the network. Siamese network
was created for ﬁngerprint recognition initially and promoted to detect difference of two
images according to same weights. Since Siamese network is designed and developed
in theory and practice, this neural network contained two sub-networks has promising
potential in cover change detection of large-size bi-temporal images. Zhan et al. [59] pro-
posed a deep Siamese convolutional network with contrastive loss weight sharing mode
and achieved the better accuracy on optical aerial images, compared with no-weighted
loss weight sharing mode. Nevertheless, there are still difﬁculties to deal with forest cover
change extraction and sample generation [60,61]. In order to solve this problem, the pur-
pose of this paper is to introduce an application of Siamese neural network (SNN) to extract
forest cover change.
The rest of this paper is organized as follows. Section 2 introduces related studies that
were used for inspiration or comparison during the development of this work. Section 3 de-
scribes in detail the proposed module in Siamese neural network and self-inverse network
that will be tested in the proposed neural network. Section 4 contains the experiments of
proposed neural network architecture and other methods for forest cover change extraction
and result comparisons in the test dataset. Sections 5 and 6 contain this work’s discussion
and concluding remarks.
2. Related Work
Baldi and Chauvin [62] proposed a neural network algorithm for ﬁngerprint recogni-
tion. Presented with a pair of ﬁngerprint images, the algorithm calculated an estimation
of the probability that an image pair came from the same ﬁnger. Bromley et al. [63] ﬁrst
introduced an algorithm consisting of two identical sub-networks to verify the authenticity
of the signatures, which was named Siamese neural network. In the study by Brom-
ley et al. [ 63], Siamese neural network utilized the distance threshold between features
from two signatures to decide the authenticity.
The Siamese neural network consists of two identical neural networks that share
weights during the encoding process. Different from the typical convolutional network
classifying its input data, a Siamese neural network learns the similarities between the
two inputs to distinguish them. When a pair of pictures are input into the Siamese neural
network, the distance metric among them is calculated by the sub-network in the neural
network. This is called symmetry, and the twin network measures the distance metric to
measure similarity. In the experiment of change detection task, the Siamese neural network
can compare each image in the test set and the training set, and then select which one is
likely to be in the different category.
When the Siamese neural network was ﬁrst proposed, it was used for video recogni-
tion and image matching [64–67]. Then, Siamese neural networks are enriched in change
detection among various image datasets and improvement tricks [68–76]. There are differ-
ent image experiment datasets to train and test Siamese neural network. Zhang et al. [68]
proposed a spectral–spatial joint learning network for change detection task. This method
integrates spatial information and spectral information in the input process, and uses
the two kinds of information to explore the changing area of the multi-spectral image.
Zhang et al. [69] proposed a light-weighted pseudo-Siamese convolutional neural network
(PSI-CNN) for change detection between airborne laser scanning and photogrammetric
data. Applied on a large building changes dataset, the proposed PSI-CNN achieved better
accuracy compared with ﬁve different architectures of CNN. To address the class labels that
are not available from the input images, Hedjam, R et al. [70] proposed a Siamese neural
Remote Sens. 2022, 14, 627 4 of 20
network to detect the changes occurring in a geographical area after major damage. Trained
with genuine and imposter patch-pairs deﬁned semi-supervised way, the Siamese neural
network has a promising performance on four real datasets. For the change detection in
aerial images, Mesquita, DB et al. [71] presented a fully convolutional Siamese autoencoder
method. This method reduces the number of labeled samples and gets competitive results
on two different datasets. As very-high-resolution (VHR) images are increasingly available,
Chen et al. [72] proposed a deep Siamese convolutional multiple-layers recurrent neural
network (SiamCRNN) for change detection in multi-temporal VHR images. Integrating the
merits of both convolutional neural network and recurrent neural network, SiamCRNN
has an advantage in two homogeneous datasets and one challenging heterogeneous VHR
images dataset. Besides, there are a variety of adjusted skills used to improve the Siamese
neural network. Jiang et al. [73] proposed a pyramid-shaped feature-based Siamese neural
network to extract building change information. The research introduced a global co-
attention mechanism to evaluate the correlation among the input feature pairs and utilized
various attention mechanisms to improve feature dependency. Chen et al. [74] proposed
a Siamese-based spatial–temporal attention neural network to capture spatial–temporal
at various scales. This model uses a self-attention change detection method in the feature
extraction stage and generates more distinctive training features. In the ﬁeld of change
detection, Siamese neural network has less parameters to extract features than fully con-
volutional neural network. Compared with FCN, Siamese neural network architecture
utilizes same weights for bi-temporal images and saves half parameters for training process.
Lee et al. [75] proposed a local similarity Siamese network (LSS-Net) to detect urban land
change in remote sensing images. The research developed a change attention map-based
content loss function to use content information on two sequential images and introduced a
local similarity attention module to enhance the performance of the LSS-Net. Wu et al. [76]
proposed an attention mechanism Siamese neural network to localize and classify damaged
buildings simultaneously. This kind of network uses different backbones and attention
mechanisms to obtain effective classiﬁcation features and channels. R. Caye Daudt [ 77]
proposed two Siamese extensions of fully convolutional networks. The ﬁrst network con-
tains concatenation weight sharing mode, which is named fully convolutional Siamese
concatenate neural network (FCSC). The second network contains subtract weight sharing
mode, which is named fully convolutional Siamese difference neural network (FCSD).
The two Siamese neural networks achieved better performance than fully convolutional
network, using both RGB and multispectral images. Due to the single patch stretched
method, FCSD is more suited for openly change detection dataset than FCSC.
In previous papers, the focus of these studies is mainly on urban change and disaster
detection, while there are little attention on forest cover change and inverse use of samples
in change detection. In addition, the feature information before change and the information
difference in the change the process have not been paid attention to in the previous Siamese
neural network research. In view of the unique characteristics of the mutual transformation
of forest cover changes and the high cost of generating forest change samples, how to
improve the detection accuracy and sample utilization rate is a problem that needs to be
solved. In this study, two weight sharing Siamese neural networks, fully concatenation and
fully subtraction, are selected as the comparison network to compare the improvement of
the network by different levels of weight sharing. Figure 1 presents the structure of the
above Siamese neural network model.
Remote Sens. 2022, 14, 627 5 of 20
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Concat
Concat
Concat
Trans Conv
Concat
Trans Conv
Concat
Trans Conv
Concat
Trans Conv
Concat
Concat
Trans Conv
Downsampling Upsampling
(a) FCSC
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Subtract
Subtract
Subtract
Trans Conv
Concat
Trans Conv
Concat
Trans Conv
Concat
Trans Conv
Concat
Subtract
Trans Conv
Downsampling Upsampling (b) FCSD
Figure 1. The architectures of Siamese neural networks. Left architecture is FCSC, and right archi-
tecture is FCSD. Compared with concatenated weights transported to upsampling process in FCSC,
the differences of weights are imported to upsampling process in FCSD.
3. Methodology
3.1. Overview
The main process of this study is shown in Figure 2, which mainly includes the
following four parts.
(1) Suface reﬂectance calculation. In order to make the images of different time compara-
ble, the digital number (DN) value of the images used in the experiment are converted
into surface reﬂectance. The LEDAPS and LaSRC surface reﬂectance algorithms re-
leased by NASA/GSFC and the University of Maryland [78,79] is used to calculate
the surface reﬂectance in this study.
(2) Sample generation. The deforestation and degradation samples are generated and
clipped into image patches. In this process, the surface reﬂectance images are aligned
according to coordinates and manually labeled to create sample images. Thus, the re-
mote sensing images and sample images are clipped into patches.
(3) Forest decrease extraction. Two new versions of the Siamese neural networks are
proposed to predict forest decrease. They are Fully Convolutional Siamese Global
Difference network (FCGD) and Fully Convolutional Siamese Detail Difference net-
work (FCDD). See Section 3.2 for details. The surface reﬂectance variables of the six
bands (blue, green, red, near-infrared and two mid-infrared bands) in the labeled
pixel are input into these models. The accuracy comparison of different methods are
conducted using quantitative evaluation metrics. The extraction results in the various
deforestation and degradation region are analyzed visually.
(4) Forest increase extraction. Self-inverse networks are designed for forest increase
extraction. See Section 3.3 for details. The forest decrease sample dataset is reversed
in time phase and imported into the network used in the forest decrease prediction.
Without changing the architecture of the networks, the Siamese neural networks are
trained by the same sample and tested for forest increase extraction.
Remote Sens. 2022, 14, 627 6 of 20
Band selection and 
Image preprocessing
Convolutional neural 
networks
2015 year Landsat-8 
OLI image
2018 year Landsat-8 
OLI image
Band selection and 
Image preprocessing
Deforestation sample 
and image patch
Image clipping 
Classification image
Image 
preprocessing
Surface reflectance
Siamese neural 
network
2015 year Landsat-8 
OLI image
2018 year Landsat-8 
OLI image
Surface reflectance
Deforestation and 
degradation sample 
generation
Image patch 
Deforestation and 
degradation prediction
Forest decrease 
extraction and 
accuracy assessment
Sample time phase 
transformation
Afforestation and 
reforestation sample
Siamese neural 
network
Afforestation and 
reforestation 
prediction
Forest increase 
extraction and 
accuracy assessment
Validation set
Performance 
comparison
Accuracy 
metrics
Quantitative 
assessment
Validation set
Performance 
comparison
Accuracy 
metrics
Quantitative 
assessment
Sample generation
Figure 2. Overall ﬂowchart adopted in this study.
Remote Sens. 2022, 14, 627 7 of 20
3.2. The Structure of the Proposed Siamese Neural Network
In the general Siamese neural network, the image patches are encoded into vector
representations through the processing of convolutional layers and pooling layers. In order
to cope with the task of change detection, different techniques and improvements are
added to the Siamese neural networks. In previous studies, concatenation and subtraction
weight sharing modes are general choices for Siamese neural network. In the FCSC,
concatenation weight sharing mode lacks the focus of change information between the
former and latter phases, while subtraction weight sharing mode lacks the focus of the
former phase information in the FCSD. To address the deﬁciency of two above networks,
it’s feasible to combine the two weight sharing modes in one neural network. In this
research, two new weight sharing modes are proposed to integrate the advantages of the
previous conventional methods. This modiﬁcation can combine the advantages of different
weight sharing methods into one network. The ﬁrst type of network is to apply the
concatenation method to the top layer of the convolutional layer, and use the subtraction
method for the remaining layers. This network is named Fully Convolutional Detail
Difference network (FCDD). This modiﬁcation puts more weight on the change information
in different layers, while the detail image information in two phases is not discarded.
The second type of network is the opposite. It applies the subtraction method to the top
layer of the convolutional layer, and uses the concatenation method for the remaining
layers. This network is named Fully Convolutional Global Difference network (FCGD).
This modiﬁcation puts more weight on the information of two phases in different layers,
while the detailed change information in two phases is not discarded. Figure 3 presents the
structure of the FCGD and FCDD models.
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Subtract
Concat
Concat
Trans Conv
Concat
Trans Conv
Concat
Trans Conv
Concat
Trans Conv
Concat
Concat
Trans Conv
Downsampling Upsampling
(a) FCGD
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Conv+Pool Conv+Pool
Concat
Subtract
Subtract
Trans Conv
Concat
Trans Conv
Concat
Trans Conv
Concat
Trans Conv
Concat
Subtract
Trans Conv
Downsampling Upsampling (b) FCDD
Figure 3. The architectures of proposed Siamese neural networks. Left architecture is FCGD, and right
architecture is FCDD. FCGD contains subtract weight sharing mode in the top layer and concatenate
weight sharing modes in the other layers, while FCDD contains concatenate weight sharing mode in
the top layer and subtract weight sharing modes in the other layers.
3.3. Self-Inverse Network
Self-inverse network is a network that can achieve the possibility of using only one
network for bi-directional image translation [80]. The self-inverse network generates an
output given an input, and vice versa, with the same neural network. For example, we
Remote Sens. 2022, 14, 627 8 of 20
call the transforming from domain X to domain Y process A, and the transforming from
domain Y to domain X process B. Compared with assigning two neural networks to process
A and B, respectively, a self-inverse network can be used to detect these two processes with
the same network. In a self-inverse network, a function f is self-inverse, meaning
f = f−1 (1)
It guarantees a one-to-one mapping. The feature of self-inverse network is that it learns
one network to perform both forward (A→ B: from A to B) and backward (B→ A: from B
to A) translation tasks. Figure 4 shows the comparison of self-inverse network and general
neural network. The decrease and increase of forest can be regarded as a pair of reverse
processes to some extent. The self-reverse network is to train the same neural network
with the forest decrease samples, and to predict the forest increase on the same image.
With the only one Siamese neural network and the same train sample dataset, the two
inverse forest change detection tasks could be completed simultaneously. As an opposite
process of forest decrease, reforestation and afforestation have inverse change processes
from deforestation and degradation. Reforestation stands for the establishment of a forest
cover in a location where the forest has been cleared for the activities like agriculture or
mining in the recent past. Afforestation is deﬁned as an establishment of forests where
there wasn’t forest before, or where forest has been missing for a period. The processes
both represent a conversion of non-forest areas into new forest, which is opposite to the
transformation of deforestation and degradation. The approximate self-reverse process of
forest reduction and forest increase motivates us to use forest reduction samples to train
the Siamese neural networks and conduct self-reverse experiments for forest increase. It’s
worth noting that the varieties of forest decrease contain those not belonging to forest
increase, such as forest to large roads. The difference tests the self-inverse learning ability
of Siamese neural network in forest increase experiment.
Forest Non-
forest
Non-
forest Forest
f
f -1
DDE
DIN
Task: A
Task: B
(a) General neural network
Forest Non-
forest
f
DDEDIN
Task: ATask: B (b) Self-inverse network
Figure 4. A comparison of the general neural network and self-inverse network. We deﬁne forest
decrease as task A and forest increase as task B. The f and f−1 are the two generator networks for the
tasks A and B, respectively. The DDE and the DI N are the associated adversarial discriminators.
3.4. Data Augmentation
Aiming to enrich the varieties of training images, the data augmentation method is
used to enhance the accuracy and reduce overﬁtting results. Before the training dataset
is given to the network architecture, the images in sample are processed by two data
augmentation skills, i.e., ﬂipping the image samples horizontally and rotating the image
samples by 90 or 180 degrees. Figures 5 and 6 are examples of ﬂipped and rotated samples,
Remote Sens. 2022, 14, 627 9 of 20
respectively. Using these augmentation skills, the raw sample quantity is enlarged by
three times.
(a)
 (b)
Figure 5. Examples of image samples ﬂipped. ( a): Raw sample. (b): Horizontally ﬂipped sample.
(a)
 (b)
 (c)
Figure 6. Examples of image samples rotated. (a): Raw sample. (b): Sample rotated by 90 degrees.
(c): Sample rotated by 180 degrees.
4. Experiments and Results Analysis
In this section, we ﬁrst present the experiment implementation details. Then, we imple-
ment quantitative analysis and qualitative analysis on the proposed method (FCGD&FCDD)
and other state-of-the-art methods (FCSC&FCSD) on forest decrease extraction. Finally,
the self-inverse experiment performances of the above-mentioned methods on forest in-
crease are compared.
4.1. Implementation Details
4.1.1. Datasets
In the subtropical and tropical area, there are different types of forest cover change.
In this study, two representative regions are selected as the study area, which include
abundant types of features, such as cities, hills, mountains, plains, and oceans. The Landsat
8 OLI images covering the study area are downloaded from the U.S. Geological Survey,
whose metadata and cloud cover are shown in Table 1. In addition, major noise of cloud,
mountain shadows, grassland and cropland are also included in the collected dataset.
Figure 7 presents the selected images in 2015 and 2018 years.
Remote Sens. 2022, 14, 627 10 of 20
Table 1. The metadata and cloud cover of the selected Landsat 8 OLI images.
Landsat Scene ID Path/Row Data Acquisition Cloud Cover Site Center
LC81190422015270LGN01 119/42 27 September 2015 2.28% 25 ◦N, 118.9◦E
LC81190422018070LGN00 119/42 11 March 2018 0.03% 25 ◦N, 118.9◦E
LC81250442015104LGN01 125/44 14 April 2015 0.75% 23.19 ◦N, 108.9◦E
LC81250442018304LGN00 125/44 31 October 2018 0.02% 23.19 ◦N, 108.9◦E
Nanning
110°0'0"E
110°0'0"E
109°0'0"E
109°0'0"E
108°0'0"E
108°0'0"E
24°0'0"N
24°0'0"N
23°0'0"N
23°0'0"N
22°0'0"N
±
0 40 8020
km
(a)
Fuzhou
120°0'0"E
120°0'0"E
119°0'0"E
119°0'0"E
118°0'0"E
118°0'0"E
27°0'0"N26°0'0"N
26°0'0"N
25°0'0"N
±
0 40 8020
km (b)
Nanning
110°0'0"E
110°0'0"E
109°0'0"E
109°0'0"E
108°0'0"E
108°0'0"E
24°0'0"N
24°0'0"N
23°0'0"N
23°0'0"N
22°0'0"N
±
0 40 8020
km
(c)
Fuzhou
120°0'0"E
120°0'0"E
119°0'0"E
119°0'0"E
118°0'0"E
118°0'0"E
27°0'0"N26°0'0"N
26°0'0"N
25°0'0"N
±
0 40 8020
km (d)
Figure 7. (a) image of 125/44 in 2015 year. (b): image of 119/42 in 2015 year. (c): image of 125/44 in
2018 year. (d): image of 119/42 in 2018 year.
4.1.2. The Creation of Training Dataset
The training samples for each image are manually labeled. The forest cover change
samples contain all deforestation and degradation types. The labeling process of these
samples is based on expert experience, and we try our best to ensure the reliability and
accuracy of sample according to the high-resolution images in the Google Map software.
To ensure the accuracy of the samples, only identiﬁed deforestation and degradation are
selected as samples. The mixed and dubious pixels in farmland boundary, grassland and
shrubland area are considered as non-forest. The representative sample examples for
deforestation and degradation types are shown in Figure 8. The 480 sample patches are
randomly divided into training part and testing part. In the experiment, 75% of the total
samples are training samples, and 25% of the total samples are testing samples. In order to
Remote Sens. 2022, 14, 627 11 of 20
ensure the consistency conditions for algorithms comparison, the training samples of forest
decrease and non-decrease in FCSC, FCSD, FCGD and FCDD are the same.
(a) Former
 (b) Latter
 (c) Label
Figure 8. The samples of training datasets. Each row represents one sample, including the image
pair (column (A,B)), and the label (the last column, white denotes change, black means no change).
Row (A) shows the degradation example; Row (B) shows the deforestation example.
4.1.3. Evaluation Metrics
Based on the error matrix, the overall accuracy (OA), the kappa coefﬁcients (KCs), Dice
Index (DI), precision, recall, F-Measure and Intersection over Union (IoU) can be calculated
using the following equations:
OA = TP + TN
TP + TN + FP + FN× 100 (2)
KC = N ∑(TP + TN )− ∑(TN× FN )
N2− ∑(TN× FN ) (3)
precision = TP
TP + FP (4)
recall = TP
TP + FN (5)
F-Measure = 2× precision× recall
precision + recall (6)
IoU = TP
TP + FP + FN (7)
where TP, TN, FN and FP represent the numbers of pixels of true forest change, true
background, false background, false change, respectively.
4.2. Comparison of Forest Decrease
4.2.1. Quantitative Analysis
The overall accuracy, Kappa coefﬁcient, precision, recall, F-Measure and IoU of each
architecture are calculated to quantitatively evaluate the accuracy of forest decrease extrac-
tion, which are shown in Table 2. Comparing the quantitative indices of the four methods in
the test dataset, the kappa coefﬁcient of the FCDD is higher than those of the FCSC, FCSD
and FCGD. The kappa coefﬁcient of the FCDD is 0.8255, while those of FCSC, FCSD and
FCGD are 0.7774, 0.7636 and 0.5823, respectively. Furthermore, comparison of the kappa
coefﬁcients shows that FCDD presents the best forest decrease and non-decrease extraction
Remote Sens. 2022, 14, 627 12 of 20
result in the four methods. The precision of FCGD is highest in the four architectures,
while the precisions of FCSC, FCSD, FCGD and FCDD are 0.8779, 0.6474, 0.9311 and 0.8565,
respectively. The recall of FCSD is highest in the four architectures, while the recalls of
FCSC, FCSD, FCGD and FCDD are 0.7023, 0.9431, 0.4277, and 0.8013, respectively. As a
comprehensive evaluation standard of precision and recall, F-Measure shows an integral
evaluation for the models. The F-Measures of FCSC, FCSD, FCGD and FCDD are 0.7803,
0.7677, 0.5861, 0.8280, respectively. This result means that FCDD has a better ability to
detect the true forest decrease and non-decrease than the other architectures. Besides, IoUs
of FCSC, FCSD, FCGD and FCDD are 0.6398, 0.6230, 0.4145, and 0.7064, respectively. This
result means that considering forest decrease detection accuracy as the unique evaluation
standard, FCDD shows a good ability in forest decrease detection.
Table 2. Accuracy evaluation of the four methods in the test dataset.
Models OA (%) KC Precision Recall F-Measure IoU
FCSC 1 99.42 0.7774 0.8779 0.7023 0.7803 0.6398
FCSD 2 99.16 0.7636 0.6474 0.9431 0.7677 0.6230
FCGD 3 99.11 0.5823 0.9311 0.4277 0.5861 0.4145
FCDD 4 99.51 0.8255 0.8565 0.8013 0.8280 0.7064
1 Siamese concatenate neural network (FCSC). 2 Siamese difference neural network (FCSD). 3 Siamese global
difference neural network (FCGD). 4 Siamese detail difference neural network (FCDD).
4.2.2. Qualitative Analysis
To evaluate the universality of the Siamese Convolutional Networks, typical forest
decrease areas, including large roads and infrastructure projects, urban expansion, com-
mercial deforestation, are selected from the classiﬁcation results. The false and true forest
decrease areas of the four neural networks are contrasted by image inspection to analyze
the reasons for classiﬁcation difference. The classiﬁcation results of the deforestation and
degradation using the four models are shown in Figure 9, and the longitude and latitude
information of center sites in Figure 9 are shown in Table 3. In Figure 9, white represents
true decrease, green represents false decrease, purple represents false background and
black represents true background.
The performance comparisons among forest selected logging are shown in row A
and B of Figure 9. With regard to most forms of logging for timber harvesting, especially
selective logging, there are various status and shapes of bareland after logging, because the
forest recover growth and vegetation of diverse densities are different when the images
are observed. Therefore, distinct extract results of degradation are necessary, when the
four methods different phenological characters and vegetation restoration status. In row
A, FCSC and FCSD have some false classiﬁcation in the left upper logging area. The false
classiﬁcation area is the land remaining some trees, which are different from the adjacent
bareland. In row B, it can be clearly seen that FCSC and FCGD have more omission forest
decrease. For the extraction of sinerispeal narrow and less obvious forest decrease, FCSC
and FCGD miss some detailed information in row B. According to the pictures in row A
and B, FCDD and FCSD have better extraction ability in forest decrease in the selected
logging area.
The performances of large roads and infrastructure projects for logging are shown in
row C and D of Figure 9. Compared with selected logging, forest decrease of large roads
and infrastructure projects have better obvious features of shape and spectral. In row C,
there is a large road construction project at the foot of the mountain. In the prediction
results of FCSC and FCGD, there are some missed extraction in the road area. FCSD and
FCDD have better prediction results in the prediction of road construction and logging in
row C. In row D, there are large areas of infrastructure projects and path construction for
infrastructure projects. The four models extract the forest decrease of path construction
well, while FCSC and FCGD miss some forest decreases in the infrastructure projects.
Remote Sens. 2022, 14, 627 13 of 20
By analysis, FCDD shows an adequate performance in extracting various degradation
areas, which conﬁrms the compatibility and robustness of the proposed algorithms.
Former
 Latter
 FCSC
 FCSD
 FCGD
 FCDD
True positive
 False positive
 False negative
 True negative
Figure 9. The classiﬁcation comparison of the four algorithms in two regions. Rows (A,B) represent
selected logging. Rows ( C,D) large roads and infrastructure projects. White: true positive; Green:
false positive; Purple: false negative; Black: true negative.
Table 3. The longitude and latitude information of site center in Figure 9.
Row Site Center
A 22 ◦28′30′′N, 109◦35′30′′E
B 25 ◦45′00′′N, 119◦34′30′′E
C 23 ◦10′30′′N, 109◦15′00′′E
D 22 ◦16′00′′N, 109◦32′30′′E
4.3. Comparison of Forest Increase
4.3.1. Quantitative Analysis
The experiment results in Table 4 delineate that the Siamese neural networks trained
by the same forest decrease samples successfully extract the reforestation and afforestation
areas. In theory, forest decrease sample dataset includes the inverse forest increase sample
dataset. This provides an opportunity to train self-inverse networks using forest decrease
samples. The experiment results of the self-inverse network of forest increase verify the
feasibility of the above idea. Among the quantitative experiment results, FCDD attains
the best score compared with other three architectures. When considering the prediction
results of forest increase and non-forest increase at the same time, KC and F-Measure can be
used as evaluation criterias. FCDD achieves the best result in the two evaluation criterias,
followed by FCSD, FCSC and FCGD. When considering the forest increase prediction as
single focus point, IoU of the four architectures show that FCSC, FCSD and FCGD (IoU of
0.4557, 0.4757 and 0.4137) are not suitable to be directly applied to self-inverse prediction,
while FCDD (IoU of 0.6923) has much better performances in forest increase detection.
Remote Sens. 2022, 14, 627 14 of 20
Table 4. Accuracy evaluation of the FCSD method in the forest increase dataset.
Models OA( %) KC Precision Recall F-Measure IoU
FCSC 1 99.33 0.6229 0.5202 0.7862 0.6261 0.4557
FCSD 2 99.57 0.6426 0.7741 0.5524 0.6447 0.4757
FCGD 3 99.17 0.5814 0.4543 0.8223 0.5852 0.4137
FCDD 4 99.76 0.8169 0.8805 0.7640 0.8181 0.6923
1 Siamese concatenate neural network (FCSC). 2 Siamese difference neural network (FCSD). 3 Siamese global
difference neural network (FCGD). 4 Siamese detail difference neural network (FCDD).
4.3.2. Qualitative Analysis
The prediction results of FCSC, FCSD, FCGD and FCDD are shown in Figure 10,
and the longitude and latitude of site center are shown in Table 5. In the forest increase ex-
traction experiments, the former status is usually bareland or shrubland, which is opposite
to latter status of forest increase.
According to row A and B, FCDD learned the conversion principle and extracted
the forest increase in bareland and shrubland precisely. In the prediction results of FCSC
and FCGD, there are some false positive area near the forest increase regions, which are
mostly the transformation from shrubland to forest. Processing the forest increase in row
B, FCDD draws the outline of target area with little mislassiﬁcation in row B. The false
positive areas in the other three experiments show that FCSC, FCSD and FCGD lack the
ability of distinguishing grassland and shrubland growth from forest increase.
Former
 Latter
 FCSC
 FCSD
 FCGD
 FCDD
True positive
 False positive
 False negative
 True negative
Figure 10. The classiﬁcation comparison of the four algorithms in afforestation and reforestation.
Rows (A,B) represent the process of bareland to forest. Rows (C–E) represent the process of bareland
to shrubland and farmland. White: true positive; Green: false positive; Purple: false negative; Black:
true negative.
In the forest increase prediction results of row C, D and E, the former status is usually
bareland, while the latter status is shrubland or farmland. In row C and D, the change
Remote Sens. 2022, 14, 627 15 of 20
process is difﬁcult to classify for FCSC and FCGD. These two models classify the process
from bareland to farmland as forest increase, which is due to the concatenation weight
sharing modes of the two models. The subtraction weight sharing mode gives FCSD and
FCDD the ability of separating farmland increase from forest increase. In row E, FCSC and
FCGD have the same misclassiﬁcation in the increase of shrubland. On the contrary, FCDD
and FCSD have little misclassiﬁcation in such area.
Table 5. The longitude and latitude information of site center in Figure 10.
Row Site Center
A 23 ◦36′00′′N, 109◦10′30′′E
B 22 ◦56′30′′N, 108◦09′00′′E
C 24 ◦00′00′′N, 108◦37′15′′E
D 23 ◦25′15′′N, 109◦13′15′′E
E 23 ◦01′30′′N, 108◦12′45′′E
5. Discussion
The application of the Siamese neural network in the ﬁeld of remote sensing has
gradually matured, and good results have been obtained in the direction of target detection
and image matching. However, research on the use of Siamese neural networks to extract
forest cover changes is still lacking. In this study, there are ﬁve meaningful aspects worth
to discuss, which are listed in following.
(1) Improvement compared with traditional Siamese neural networks. In this study,
the classiﬁcation accuracy of forest decrease and increase in two regions is evaluated
through qualitative and quantitative analysis. The results demonstrate that the clas-
siﬁcation accuracy of subtract weight mode FCDD in various forest cover change
is higher than that of concatenate and subtract weight sharing mode in FCSC and
FCSD. Subsequently, the performance of subtract weight sharing mode in eliminating
noise of forest increase is better than that of concatenate weight sharing mode. This
phenomenon is due to the fact that the subtract weight sharing mode is more able
to use the different information for forest change extraction than the concatenate
weight sharing mode. Due to the fully concatenation weight sharing mode, FCSC
extracts two phase information and lacks focus of change information. Due to the
fully subtraction weight sharing mode, FCSD is designed to focus on the change
information between two phase and extracts some pseudo-change information in
the same time. Combining the above weight sharing mode, FCDD has the ability of
utilizing different information and pseudo-change information simultaneously.
(2) Differences between FCDD and FCGD. As two types of Siamese neural network,
FCDD and FCGD have obvious differences in the theoretical method and experiment
results. In the theoretical method, FCDD has a concatenation weight sharing mode in
the top layer of downsampling process, while the other layers are made by subtract
weight sharing mode. In the downsampling process of FCGD, the top layer is a
subtract weight sharing mode, and the other layers are a concatenate weight sharing
mode. In the experiment results, FCDD has better capability of noise eliminating than
FCGD, such as shrub wither and grass wilt. The subtract weight sharing mode in the
more subtle convolutional layers gives FCDD better forest cover change detection
ability. As the break-even point of precision and recall, F-measure shows the quantita-
tive performance measure of predict results. FCDD has the best F-measure score in
forest decrease and increase extraction experiments, following by FCSC and FCGD.
This difference approves that given the same forest change image, FCDD has better
ability in predicting forest cover change than FCGD.
(3) Self-inverse network. In the ﬁeld of remote sensing, it is the ﬁrst time that self-inverse
network is used for forest cover change detection. In such a bidirectional change ﬁeld,
the self-inverse network experiment not only reduces the generation cost of training
Remote Sens. 2022, 14, 627 16 of 20
data, but also extends universality of classiﬁcation architecture model. According to
the results in the paper, the Siamese neural network has proven its accurate detection
capacity and compatible universality in forest cover translation. The types of features
before and after forest decrease and increase are not completely equal, thus forest
decrease and increase are not completely reversible. The feature types after the forest
decrease include the feature types before the increase, which allows the sample of the
forest decrease to be used for training increase. However, lacking the combination of
the bi-temporal feature information and the difference information, the traditional
Siamese neural networks (FCSC and FCSD) perform poorly in self-inverse prediction.
On the other hand, the novel fused weight sharing strategies make the proposed
Siamese neural networks FCDD more robust to be applied to a self-inverse task.
(4) Factors affecting accuracy. In the whole process of forest cover change detection, there
are several factors affecting accuracy. The ﬁrst factor is image preprocessing. To ensure
data consistency, most change detection maps are based on the top of atmosphere
(TOA) reﬂectance or surface reﬂectance. However, due to various shooting time and
geographic variation, the surface reﬂectance of image collection exists difference,
which inﬂuences control variables of the change information extraction. Secondly,
the forest decrease training dataset includes some kinds of deforestation that doesn’t
exist in the reforestation. This situation leads to the problem that the training dataset of
deforestation and reforestation is incompletely self-inverse. The reforestation samples
are improper for self-inverse deforestation experiment. Otherwise, various mixtures
between the forest change and unchange areas exist.
(5) Training samples generation. In order to select accurate samples of forest cover
changes in the complex surface, this experiment combined Landsat medium-resolution
images and Google Earth high-resolution images to manually label the forest decrease
and increase samples. However, this process limits the automatic processing capacity
of the proposed algorithm. This problem shows that various types of typical forest
cover change samples will be the main demand for future work.
Compared with the studies of forest cover change extraction in fully convolutional
neural networks, the present study utilized a weight sharing mode composed of subtrac-
tion and concatenation means in the Siamese neural networks. This promotion reduced
the parameters to recognize the change features in the architecture, and decreased the
possibility of overﬁtting in the training process. In addition, the proposed self-inverse
network demonstrated that Siamese neural networks have the ability of extracting forest
cover increase and decrease using a series of forest cover decrease sample through the
self-inverse network. This adjustment saved the cost of sample generation in forest cover
change extraction and enlarged the feasibility of the Siamese neural network.
The two major contributions of this paper are listed as follows:
(1) A novel weight sharing mode of a Siamese neural network based on U-Net for forest
cover change detection is proposed, and this method obtains promising classiﬁca-
tion results.
(2) Self-inverse network of Siamese architecture is generated. According to the self-
inverse network, forest decrease sample dataset is used for change detection in forest
increase, which implements transfer learning of sample dataset and improves the
utilization rate of sample dataset.
On the premise of providing more numbers and types of training samples, the pro-
posed algorithm can reﬁne the types of changes and be used to predict large-scale forest
cover detection. In addition, the model can be further improved to suitable for other types
of land cover change extractions with self-inverse process.
6. Conclusions
Deep neural networks have demonstrated good capabilities in target recognition and
image segmentation in the ﬁeld of remote sensing. This study uses a Siamese neural
network after adjusting the weight sharing method to extract deforestation, degradation,
Remote Sens. 2022, 14, 627 17 of 20
afforestation and reforestation areas in Landsat 8 OLI images. Two images of typical
subtropical regions are selected, and two traditional algorithms (i.e., Siamese concatenate
neural network (FCSC), Siamese difference neural network (FCSD) are included to compare
with Siamese global difference neural network (FCGD) and Siamese detail difference neural
network (FCDD) in their performances when extracting forest change information. Then,
the performances of various forest cover change extractions and noise suppressions are
comprehensively compared. The conclusions are summarized as follows:
(1) Based on a visual comparison, the performance of the Siamese detail difference neural
network extracting forest cover change is better than those of Siamese concatenate
neural network, Siamese difference neural network and Siamese global difference
neural network. Moreover, quantitative evaluation shows that the overall accuracy
and kappa coefﬁcients of FCDD are higher than those of the other three classiﬁers.
The kappa coefﬁcients of FCDD in forest decrease and increase extraction experiments
are 82.55% and 81.69%, and the F-measures and IoUs of those are 0.8280 and 0.8181,
0.7064 and 0.6923.
(2) Compared with FCSC, FCSD and FCGD, the performance of FCDD demonstrates
that it can precisely extract three types of large forest decrease areas (i.e., large roads
and infrastructure projects, urban expand and logging), and detailed deforestation
can also be identiﬁed. Furthermore, FCDD can effectively eliminate noise, such as
grassland and shrub perishment.
(3) In the forest increase extraction, FCDD has the advantage of self-inverse function
learning the principle of forest transfer to non-forest. Trained by the existed forest
decrease dataset, FCDD has the capacity of detecting forest increase without the effort
of amending neural network parameters.
This paper introduces a Siamese neural network for extracting forest cover change,
and the results conﬁrm that the proposed method can achieve sufﬁcient performance.
For future research, the newly released forest cover change products can be used to further
enhance the automation and versatility of the proposed algorithm. Then, the proposed
method can be used to map large scale forest cover change, which will help us understand
the forest change information under a background of global change.
Author Contributions: Y.G. developed the methods, carried out the experiments and wrote the
manuscript. W.J., X.Z. and T.L. supervised the research. All the authors analyzed the results and im-
proved the manuscript. All authors have read and agreed to the published version of the manuscript.
Funding: This research was funded by the program of the National Natural Science Foundation
of China (61731022, 62101531); the National Key Research and Development Programs of China
(2016YFA0600302); the Strategic Priority Research Program of the Chinese Academy of Sciences
(XDA19090300).
Institutional Review Board Statement: Not applicable.
Informed Consent Statement: Not applicable.
Data Availability Statement: Not applicable.
Conﬂicts of Interest: The authors declare no conﬂict of interest.
References
1. Hansen, M.C.; Potapov, P .V .; Moore, R.; Hancher, M.; Turubanova, S.A.; Tyukavina, A.; Townshend, J. High-resolution global
maps of 21st-century forest cover change. Science 2013, 342, 850–853. [CrossRef] [PubMed]
2. Feng, M.; Sexton, J.O.; Huang, C.; Anand, A.; Channan, S.; Song, X.P .; Townshend, J.R. Earth science data records of global forest
cover and change: Assessment of accuracy in 1990, 2000, and 2005 epochs. Remote Sens. Environ. 2016, 184, 73–85. [CrossRef]
3. Curtis, P .G.; Slay, C.M.; Harris, N.L.; Tyukavina, A.; Hansen, M.C. Classifying drivers of global forest loss. Science 2018, 361,
1108–1111. [CrossRef]
4. Houghton, R.A. Historic role of forests in the global carbon cycle. In Carbon Dioxide Mitigation in Forestry and Wood Industry ;
Springer: Berlin/Heidelberg, Germany, 1998; pp. 1–24.
Remote Sens. 2022, 14, 627 18 of 20
5. Song, X.P .; Huang, C.; Saatchi, S.S.; Hansen, M.C.; Townshend, J.R. Annual carbon emissions from deforestation in the Amazon
Basin between 2000 and 2010. PLoS ONE 2015, 10, e0126754. [CrossRef] [PubMed]
6. Woodward, C.; Shulmeister, J.; Larsen, J.; Jacobsen, G.E.; Zawadzki, A. The hydrological legacy of deforestation on global
wetlands. Science 2014, 346, 844–847. [CrossRef] [PubMed]
7. Haddad, N.M.; Brudvig, L.A.; Clobert, J.; Davies, K.F.; Gonzalez, A.; Holt, R.D.; Townshend, J.R. Habitat fragmentation and its
lasting impact on Earth’s ecosystems. Sci. Adv. 2015, 1, e1500052. [CrossRef] [PubMed]
8. Smart, L.S.; Swenson, J.J.; Christensen, N.L.; Sexton, J.O. Three-dimensional characterization of pine forest type and red-cockaded
woodpecker habitat by small-footprint, discrete-return lidar. For. Ecol. Manag. 2012, 281, 100–110. [CrossRef]
9. Zuluaga, G.J. C.; Rodewald, A.D. Response of mixed-species ﬂocks to habitat alteration and deforestation in the Andes. Biol.
Conserv. 2015, 188, 72–81. [CrossRef]
10. Barber, C.P .; Cochrane, M.A.; Souza Jr, C.M.; Laurance, W.F. Roads, deforestation, and the mitigating effect of protected areas in
the Amazon. Biol. Conserv. 2014, 177, 203–209. [CrossRef]
11. Laumonier, Y.; Uryu, Y.; Stüwe, M.; Budiman, A.; Setiabudi, B.; Hadian, O. Eco-ﬂoristic sectors and deforestation threats in
Sumatra: Identifying new conservation area network priorities for ecosystem-based land use planning. Biodivers. Conserv. 2010,
19, 1153–1174. [CrossRef]
12. Etter, A.; McAlpine, C.; Wilson, K.; Phinn, S.; Possingham, H. Regional patterns of agricultural land use and deforestation in
Colombia. Agric. Ecosyst. Environ. 2006, 114, 369–386. [CrossRef]
13. Pahari, K.; Murai, S. Modelling for prediction of global deforestation based on the growth of human population. ISPRS J.
Photogramm. Remote Sens. 1999, 54, 317–324. [CrossRef]
14. Hansen, M.C.; DeFries, R.S. Detecting long-term global forest change using continuous ﬁelds of tree-cover maps from 8-km
advanced very high resolution radiometer (AVHRR) data for the years 1982–1999.Ecosystems 2004, 7, 695–716. [CrossRef]
15. Renó, V .F.; Novo, E.M.; Suemitsu, C.; Rennó, C.D.; Silva, T.S. Assessment of deforestation in the Lower Amazon ﬂoodplain using
historical Landsat MSS/TM imagery. Remote Sens. Environ. 2011, 115, 3446–3456. [CrossRef]
16. Thiel, C.; Thiel, C.; Riedel, T.; Schmullius, C. Object based classiﬁcation of SAR data for the delineation of forest cover maps and
the detection of deforestation—A viable procedure and its application in GSE Forest Monitoring. In Object-Based Image Analysis;
Springer: Berlin/Heidelberg, Germany, 2008; pp. 327–343.
17. Kim, D.H.; Sexton, J.O.; Noojipady, P .; Huang, C.Q.; An, A.; Channan, S.; Feng, M.; Townshend, J.R. Global, Landsat-based
forest-cover change from 1990 to 2000. Remote Sens. Environ. 2014, 155, 178–193. [CrossRef]
18. Huang, X.M.; Fied, M.A. Distance metric-based forest cover change detection using MODIS time series. Int. J. Appl. Earth Obs.
Geoinf. 2014, 29, 78–92. [CrossRef]
19. Qin, Y.W.; Xiao, X.M.; Wang, J.; Dong, J.W.; Ewing, K.T.; Hoagl, B.; Hough, D.J.; Fagin, T.D.; Zou, Z.H.; Geissler, G.L.; et al.
Mapping Annual Forest Cover in Sub-Humid and Semi-Arid Regions through Analysis of Landsat and PALSAR Imagery. Remote
Sens. 2016, 8, 933. [CrossRef]
20. De Filho, F.J. B.O.; Metzger, J.P . Thresholds in landscape structure for three common deforestation patterns in the Brazilian
Amazon. Landsc. Ecol. 2006, 21, 1061–1073. [CrossRef]
21. Huete, A.; Didan, K.; Miura, T.; Rodriguez, E.P .; Gao, X.; Ferreira, L.G. Overview of the radiometric and biophysical performance
of the MODIS vegetation indices. Remote Sens. Environ. 2002, 83, 195–213. [CrossRef]
22. Huete, A.; Justice, C.; Liu, H. Development of vegetation and soil indices for MODIS-EOS. Remote Sens. Environ. 2002, 49, 224–234.
[CrossRef]
23. Huete, A.R.; Liu, H.Q.; Batchily, K.; Van Leeuwen, W. A comparison of vegetation indices over a global set of TM images for
EOS-MODIS. Remote Sens. Environ. 1997, 59, 440–451. [CrossRef]
24. Pinty, B.; Verstraete, M.M. GEMI: A non-linear index to monitor global vegetation from satellites. Vegetation 1992, 101, 15–20.
[CrossRef]
25. Souza, C.M., Jr.; Siqueira, J.V .; Sales, M.H.; Fonseca, A.V .; Ribeiro, J.G.; Numata, I.; Cochrane, M.A.; Barber, C.P .; Roberts, D.A.;
Barlow, J. Ten-year landsat classiﬁcation of deforestation and forest degradation in the brazilian amazon.Remote Sens. 2013, 5,
5493–5513. [CrossRef]
26. Carlos, S.M., Jr.; Roberts, D.A.; Cochrane, M.A. Combining spectral and spatial information to map canopy damage from selective
logging and forest ﬁres. Remote Sens. Environ. 2005, 98, 329–343. [CrossRef]
27. Jin, S.; Sader, S.A. Comparison of time series tasseled cap wetness and the normalized difference moisture index in detecting
forest disturbances. Remote Sens. Environ. 2005, 94, 364–372. [CrossRef]
28. Goodwin, N.R.; Coops, N.C.; Wulder, M.A.; Gillanders, S.; Schroeder, T.A.; Nelson, T. Estimation of insect infestation dynamics
using a temporal sequence of Landsat data. Remote Sens. Environ. 2008, 112, 3680–3689. [CrossRef]
29. Hayes, D.J.; Cohen, W.B.; Sader, S.A.; Irwin, D.E. Estimating proportional change in forest cover as a continuous variable from
multi-year MODIS data. Remote Sens. Environ. 2008, 112, 735–749. [CrossRef]
30. Tucker C.J. Red and photographic infrared linear combinations for monitoring vegetation. Remote Sens. Environ. 1979, 8, 127–150.
[CrossRef]
31. DeVries, B.; Verbesselt, J.; Kooistra, L.; Herold, M. Robust monitoring of small-scale forest disturbances in a tropical montane
forest using Landsat time series. Remote Sens. Environ. 2015, 161, 107–121. [CrossRef]
Remote Sens. 2022, 14, 627 19 of 20
32. Miura, T.; Huete, A.R.; Van Leeuwen, W.J. D.; Didan, K. Vegetation detection through smoke-ﬁlled AVIRIS images: An assessment
using MODIS band passes. J. Geophys. Res. Atmos. 1998, 103, 32001–32011. [CrossRef]
33. Schultz, M.; Clevers, J.G.; Carter, S.; Verbesselt, J.; Avitabile, V .; Quang, H.V .; Herold, M. Performance of vegetation indices from
Landsat time series in deforestation monitoring. Int. J. Appl. Earth Obs. Geoinf. 2016, 52, 318–327. [CrossRef]
34. Chen, G.; Hay, G.J.; Carvalho, L.M.; Wulder, M.A. Object-based change detection. Int. J. Remote Sens. 2012, 33, 4434–4457.
[CrossRef]
35. Heurich, M.; Ochs, T.; Resen, T.; Schneider, T. Object-orientated image analysis for the semi-automatic detection of dead trees
following a spruce bark beetle (Ips typographus) outbreak. Eur. J. For. Res. 2010, 129, 313–324. [CrossRef]
36. Hese, S.; Schmullius, C. Approaches to KYOTO Afforestation, Reforestation and Deforestation Mapping in Siberia using Object
Oriented Change Detection Methods. In Hazard Ecology: Approaches and T echniques; Mittal Publications: Delhi, India, 2010; p. 77.
37. Guild, L.S.; Cohen, W.B.; Kauffman, J.B. Detection of deforestation and land conversion in Rondônia, Brazil using change
detection techniques. Int. J. Remote Sens. 2004, 25, 731–750. [CrossRef]
38. Longépé, N.; Rakwatin, P .; Isoguchi, O.; Shimada, M.; Uryu, Y.; Yulianto, K. Assessment of ALOS PALSAR 50 m Orthorectiﬁed
FBD Data for Regional Land Cover Classiﬁcation by Support Vector Machines. IEEE T rans. Geosci. Remote Sens. 2011, 49,
2135–2150. [CrossRef]
39. Huang, C.; Song, K.; Kim, S.; Townshend, J.R.; Davis, P .; Masek, J.G.; Goward, S.N. Use of a dark object concept and support
vector machines to automate forest cover change analysis. Remote Sens. Environ. 2008, 112, 970–985. [CrossRef]
40. DeFries, R.S.; Chan, J.C. W. Multiple Criteria for Evaluating Machine Learning Algorithms for Land Cover Classiﬁcation from
Satellite Data. Remote Sens. Environ. 2000, 74, 503–515. [CrossRef]
41. Hosonuma, N.; Herold, M.; De Sy, V .; De Fries, R.S.; Brockhaus, M.; Verchot, L.; Romijn, E. An assessment of deforestation and
forest degradation drivers in developing countries. Environ. Res. Lett. 2012, 7, 044009. [CrossRef]
42. Jin, Y.; Sung, S.; Lee, D.K.; Biging, G.S.; Jeong, S. Mapping Deforestation in North Korea Using Phenology-Based Multi-Index and
Random Forest. Remote Sens. 2016, 8, 997. [CrossRef]
43. Bueno, I.T.; Acerbi Júnior, F.W.; Silveira, E.M.O.; Mello, J.M.; Carvalho, L.M.T.; Gomide, L.R.; Withey, K.; Scolforo, J.R.S.
Object-Based Change Detection in the Cerrado Biome Using Landsat Time Series. Remote Sens. 2019, 11, 570. [CrossRef]
44. Mas, J.F.; Puig, H.; Palacio, J.L.; Sosa-López, A. Modelling deforestation using GIS and artiﬁcial neural networks. Environ. Model.
Softw. 2004, 19, 461–471. [CrossRef]
45. Nunes Kehl, T.; Todt, V .; Veronez, M.R.; Cazella, S.C. Amazon Rainforest Deforestation Daily Detection Tool Using Artiﬁcial
Neural Networks and Satellite Images. Sustainability 2012, 4, 2566–2573. [CrossRef]
46. Maretto, R.V .; Fonseca, L.M.; Jacobs, N.; Körting, T.S.; Bendini, H.N.; Parente, L.L. Spatio-Temporal Deep Learning Approach to Map
Deforestation in Amazon Rainforest.IEEE Geosci. Remote Sens. Lett. 2020, 18, 771–775. [CrossRef]
47. Mayﬁeld, H.J.; Smith, C.; Gallagher, M.; Hockings, M. Considerations for selecting a machine learning technique for predicting
deforestation. Environ. Model. Softw. 2020, 104741. [CrossRef]
48. Zhang, L.; Zhang, L.; Du, B. Deep Learning for Remote Sensing Data: A Technical Tutorial on the State of the Art. IEEE Geosci.
Remote Sens. Mag. 2016, 4, 22–40. [CrossRef]
49. Hughes, L.H.; Schmitt, M.; Zhu, X.X. Mining Hard Negative Samples for SAR-Optical Image Matching Using Generative
Adversarial Networks. Remote Sens. 2018, 10, 1552. [CrossRef]
50. Ma, W.; Zhang, J.; Wu, Y.; Jiao, L.; Zhu, H.; Zhao, W. A Novel Two-Step Registration Method for Remote Sensing Images Based on
Deep and Local Features. IEEE T rans. Geosci. Remote Sens. 2019, 57, 4834–4843. [CrossRef]
51. Merkle, N.; Auer, S.; Müller, R.; Reinartz, P . Exploring the Potential of Conditional Adversarial Networks for Optical and SAR
Image Matching. IEEE J. Sel. T op. Appl. Earth Obs. Remote Sens. 2018, 11, 1811–1820. [CrossRef]
52. Kemker, R.; Salvaggio, C.; Kanan, C. Algorithms for semantic segmentation of multispectral remote sensing imagery using deep
learning. ISPRS J. Photogramm. Remote Sens. 2018, 145, 60–77. [CrossRef]
53. Long, J.; Shelhamer, E.; Darrell, T. Fully Convolutional Networks for Semantic Segmentation. In Proceedings of the IEEE
Conference on Computer Vision and Pattern Recognition, Boston, MA, USA, 7–12 June 2015; pp. 3431–3440.
54. Ronneberger, O.; Fischer, P .; Brox, T. U-Net: Convolutional Networks for Biomedical Image Segmentation. In Medical Image
Computing and Computer-Assisted Intervention—MICCAI 2015 ; Navab, N., Hornegger, J., Wells, W., Frangi, A., Eds.; Springer:
Cham, Switzerland, 2015.
55. Chen, L.C.; Papandreou, G.; Kokkinos, I.; Murphy, K.; Yuille, A.L. DeepLab: Semantic Image Segmentation with Deep Con-
volutional Nets, Atrous Convolution, and Fully Connected CRFs. IEEE T rans. Pattern Anal. Mach. Intell. 2018, 40, 834–848.
[CrossRef]
56. Timilsina, S.; Aryal, J.; Kirkpatrick, J.B. Mapping Urban Tree Cover Changes Using Object-Based Convolution Neural Network
(OB-CNN). Remote Sens. 2020, 12, 3017. [CrossRef]
57. Bragagnolo, L.; da Silva, R.V .; Grzybowski, J.M.V . Amazon forest cover change mapping based on semantic segmentation by
U-Nets. Ecol. Inform. 2021, 62, 101279. [CrossRef]
58. de Bem, P .P .; de Carvalho Junior, O.A.; Fontes Guimarães, R.; Trancoso Gomes, R.A. Change Detection of Deforestation in the
Brazilian Amazon Using Landsat Data and Convolutional Neural Networks. Remote Sens. 2020, 12, 901. [CrossRef]
59. Zhan, Y.; Fu, K.; Yan, M.; Sun, X.; Wang, H.; Qiu, X. Change Detection Based on Deep Siamese Convolutional Network for Optical
Aerial Images. IEEE Geosci. Remote Sens. Lett. 2017, 14, 1845–1849. [CrossRef]
Remote Sens. 2022, 14, 627 20 of 20
60. Rendenieks, Z.; Nita, M.D.; Nikodemus, O.; Radeloff, V .C. Half a century of forest cover change along the Latvian-Russian border
captured by object-based image analysis of Corona and Landsat TM/OLI data.Remote Sens. Environ. 2020, 249, 112010. [CrossRef]
61. Huang, C.; Kim, S.; Song, K.; Townshend, J.R.; Davis, P .; Altstatt, A.; Musinsky, J. Assessment of Paraguay’s forest cover change
using Landsat observations. Glob. Planet. Chang. 2009, 67, 1–12. [CrossRef]
62. Baldi, P .; Chauvin, Y. Neural networks for ﬁngerprint recognition. Neural Comput. 1993, 5, 402–418. [CrossRef]
63. Bromley, J.; Guyon, I.; LeCun, Y.; Säckinger, E.; Shah, R. Signature Veriﬁcation using a “Siamese” Time Delay Neural Network.
Int. J. Pattern Recognit. Artif. Intell. 1993, 7, 669–688. [CrossRef]
64. Wang, X.; Gupta, A. Unsupervised Learning of Visual Representations Using Videos. In Proceedings of the 2015 IEEE International
Conference on Computer Vision (ICCV), Santiago, Chile, 7–13 December 2015; pp. 2794–2802.
65. Tao, R.; Gavves, E.; Smeulders, A.W. Siamese Instance Search for Tracking. In Proceedings of the 2016 IEEE Conference on
Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV , USA, 27–30 June 2016; pp. 1420–1429.
66. Simo-Serra, E.; Trulls, E.; Ferraz, L.; Kokkinos, I.; Fua, P .; Moreno-Noguer, F. Discriminative Learning of Deep Convolutional
Feature Point Descriptors. In Proceedings of the 2015 IEEE International Conference on Computer Vision (ICCV), Santiago, Chile,
7–13 December 2015; pp. 118–126.
67. He, H.; Chen, M.; Chen, T.; Li, D. Matching of Remote Sensing Images with Complex Background Variations via Siamese
Convolutional Neural Network. Remote Sens. 2018, 10, 355. [CrossRef]
68. Zhang, W.; Lu, X. The Spectral-Spatial Joint Learning for Change Detection in Multispectral Imagery. Remote Sens. 2019, 11, 240.
[CrossRef]
69. Zhang, Z.; Vosselman, G.; Gerke, M.; Persello, C.; Tuia, D.; Yang, M.Y. Detecting Building Changes between Airborne Laser
Scanning and Photogrammetric Data. Remote Sens. 2019, 11, 2417. [CrossRef]
70. Hedjam, R.; Abdesselam, A.; Melgani, F. Change Detection in Unlabeled Optical Remote Sensing Data Using Siamese CNN. IEEE
J. Sel. T op. Appl. Earth Obs. Remote Sens. 2020, 13, 4179–4187. [CrossRef]
71. Mesquita, D.B.; dos Santos, R.F.; Macharet, D.G.; Campos, M.F.; Nascimento, E.R. Fully Convolutional Siamese Autoencoder for
Change Detection in UA V Aerial Images.IEEE Geosci. Remote Sens. Lett.2020, 17, 1455–1459. [CrossRef]
72. Chen, H.; Wu, C.; Du, B.; Zhang, L.; Wang, L. Change Detection in Multisource VHR Images via Deep Siamese Convolutional
Multiple-Layers Recurrent Neural Network. IEEE T rans. Geosci. Remote Sens. 2020, 58, 2848–2864. [CrossRef]
73. Jiang, H.; Hu, X.; Li, K.; Zhang, J.; Gong, J.; Zhang, M. PGA-SiamNet: Pyramid Feature-Based Attention-Guided Siamese Network
for Remote Sensing Orthoimagery Building Change Detection. Remote Sens. 2020, 12, 484. [CrossRef]
74. Chen, H.; Shi, Z. A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection.
Remote Sens. 2020, 12, 1662. [CrossRef]
75. Lee, H.; Lee, K.; Kim, J.H.; Na, Y.; Park, J.; Choi, J.P .; Hwang, J.Y. Local Similarity Siamese Network for Urban Land Change
Detection on Remote Sensing Images. IEEE J. Sel. T op. Appl. Earth Obs. Remote Sens. 2021, 1, 4139–4149. [CrossRef]
76. Wu, C.; Zhang, F.; Xia, J.; Xu, Y.; Li, G.; Xie, J.; Liu, R. Building Damage Detection Using U-Net with Attention Mechanism from
Pre- and Post-Disaster Remote Sensing Datasets. Remote Sens. 2021, 13, 905. [CrossRef]
77. Caye Daudt, R.; Le Saux, B.; Boulch, A. Fully Convolutional Siamese Networks for Change Detection. In Proceedings of the 2018
25th IEEE International Conference on Image Processing (ICIP), Athens, Greece, 7–10 October 2018; pp. 4063–4067.
78. Schmidt, G.; Jenkerson, C.B.; Masek, J.; Vermote, E.; Gao, F. Landsat Ecosystem Disturbance Adaptive Processing System (Ledaps)
Algorithm Description; U.S. Geological Survey: Reston, VA, USA, 2013. [CrossRef]
79. Vermote, E.; Roger, J.C.; Franch, B.; Skakun, S. LaSRC (Land Surface Reﬂectance Code): Overview, application and validation
using MODIS, VIIRS, LANDSAT and Sentinel 2 data’s. In Proceedings of the IGARSS 2018–2018 IEEE International Geoscience
and Remote Sensing Symposium, Valencia, Spain, 22–27 July 2018; pp. 8173–8176. [CrossRef]
80. Shen, Z.; Zhou, S.K.; Chen, Y.; Georgescu, B.; Liu, X.; Huang, T. One-to-one Mapping for Unpaired Image-to-image Translation.
In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, Snowmass, CO, USA, 1–5 March 2020;
pp. 1170–1179.