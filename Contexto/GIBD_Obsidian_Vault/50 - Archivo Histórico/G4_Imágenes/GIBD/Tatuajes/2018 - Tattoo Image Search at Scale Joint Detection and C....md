---
aliases: [2018 - Tattoo Image Search at Scale Joint Detection and C...]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2024-09-10
origen_zip: GIBD-20260521T205218Z-3-005.zip
ruta_interna: "GIBD/Tatuajes/CoNaIISI 2024/2018 - Tattoo Image Search at Scale Joint Detection and Compact Representation Learning.pdf"
tamanio_bytes: 2840746
---

# 2018 - Tattoo Image Search at Scale Joint Detection and Compact Representation Learning

Ruta interna: `GIBD/Tatuajes/CoNaIISI 2024/2018 - Tattoo Image Search at Scale Joint Detection and Compact Representation Learning.pdf`

---

1
Tattoo Image Search at Scale: Joint Detection
and Compact Representation Learning
Hu Han, Member, IEEE, Jie Li, Anil K. Jain, Fellow, IEEE,
Shiguang Shan, Senior Member, IEEE and Xilin Chen, Fellow, IEEE
Abstract—The explosive growth of digital images in video surveillance and social media has led to the signiﬁcant need for efﬁcient
search of persons of interest in law enforcement and forensic applications. Despite tremendous progress in primary biometric traits
(e.g., face and ﬁngerprint) based person identiﬁcation, a single biometric trait alone can not meet the desired recognition accuracy in
forensic scenarios. Tattoos, as one of the important soft biometric traits, have been found to be valuable for assisting in person
identiﬁcation. However, tattoo search in a large collection of unconstrained images remains a difﬁcult problem, and existing tattoo
search methods mainly focus on matching cropped tattoos, which is different from real application scenarios. To close the gap, we
propose an efﬁcient tattoo search approach that is able to learn tattoo detection and compact representation jointly in a single
convolutional neural network (CNN) via multi-task learning. While the features in the backbone network are shared by both tattoo
detection and compact representation learning, individual latent layers of each sub-network optimize the shared features toward the
detection and feature learning tasks, respectively. We resolve the small batch size issue inside the joint tattoo detection and compact
representation learning network via random image stitch and preceding feature buffering. We evaluate the proposed tattoo search
system using multiple public-domain tattoo benchmarks, and a gallery set with about 300K distracter tattoo images compiled from
these datasets and images from the Internet. In addition, we also introduce a tattoo sketch dataset containing 300 tattoos for
sketch-based tattoo search. Experimental results show that the proposed approach has superior performance in tattoo detection and
tattoo search at scale compared to several state-of-the-art tattoo retrieval algorithms.
Index Terms—Large-scale tattoo search, joint detection and representation learning, sketch based search, multi-task learning.
!
1 I NTRODUCTION
I
N the past few decades, because of the advances in
computing, imaging, and Internet technologies, digital
images and videos are now widely used for representing
information in video surveillance, and social media. In 2017,
IHS Markit estimated that the United States has approx-
imately 50 million surveillance cameras, and China has
about 176 million.1 Another statistics by YouTube in 2017
shows that the length of videos uploaded to YouTube every
minute is approximately 300 hours.2 Given the explosive
growth of image and video data, there is great demand for
efﬁcient instance search technologies, particularly for per-
sons of interest in law enforcement and forensics. Although
tremendous progress has been made in face recognition
based person identiﬁcation, many situations exist where
face recognition cannot identify an individual with sufﬁ-
ciently high accuracy. This is especially true when the face
image quality is poor or the persons of interest intention-
ally hide their faces. In such cases, it is critical to acquire
supplementary information to assist in person identiﬁcation
[1]. On the basis of this rationale, since 2009 the US Federal
• Hu Han, Jie Li, Shiguang Shan, and Xilin Chen are with the Key
Laboratory of Intelligent Information Processing of Chinese Academy
of Sciences (CAS), Institute of Computing Technology, CAS, Beijing,
100190, China
Anil K. Jain is with the Department of Computer Science and Engineering,
Michigan State University, East Lansing, MI 48824, USA.
E-mail: {hanhu, sgshan, xlchen }@ict.ac.cn; jain@cse.msu.edu;
jie.li@vipl.ict.cn
1. http://www.straitstimes.com/opinion/chinas-all-seeing-surveil
lance-state-is-reading-its-citizens-faces
2. https://www.statisticbrain.com/youtube-statistics
(a)
 (b)
(c) (d)
Fig. 1. Tattoo examples: (a) a tattoo on the right hand of a Chiribaya
mummy in southern Peru who lived from A.D. 900 to 1350,4 (b) a tattoo
on the head signifying gang membership association, 5 (c, d) tattoos of
a masked ringleader of the riots during Euro 2012 qualiﬁer, and the
suspect of the masked ringleader identiﬁed by tattoos.6
Bureau of Investigation (FBI) has been working on extend-
ing the capability of its Integrated Automated Fingerprint
Identiﬁcation System (IAFIS) by using additional biometric
modalities, including iris, palm print, scars, marks, and
tattoos3. This new person identiﬁcation system is named the
Next Generation Identiﬁcation (NGI) system [2], which is
able to offer state-of-the-art biometric identiﬁcation services
for homeland security, law enforcement, etc.
3. Scars, marks, and tattoos are collectively referred to as SMT.
4. https://www.smithsonianmag.com/history/tattoos-144038580
5. http://www.gangink.com/index.php?pr=GANG LIST
6. http://www.telegraph.co.uk/sport/football/teams/serbia/
8061619/Masked-ringleader-of-crowd-trouble-during-Italy-Serbia-cla
sh-identified-by-tattoos .html
arXiv:1811.00218v1  [cs.CV]  1 Nov 2018
2
Among the various soft biometric traits, tattoos, in par-
ticular, have received substantial attention over the past
several years due to their prevalence among the criminal
section of the population and their saliency in visual at-
tention. Humans have marked their bodies with tattoos to
express personal beliefs or to signify group association for
more than 5, 000 years (see Fig. 1 (a)). Figs. 1 (c, d) show an
example how a suspect of the masked ringleader of the riots
during Euro 2012 qualiﬁer was arrested based on the tattoos
on his arms. In fact, criminal investigations have leveraged
soft biometric traits as far back as the late 19th century
[3].For example, the ﬁrst personal identiﬁcation system, the
Bertillon system, tried to provide a precise and scientiﬁc
method to identify criminals by using physical measure-
ments of body parts, especially measurements of the head
and face, as well as the images of SMT on the body. Tattoos
were also reported to be useful for assisting in identifying
victims of terrorist attacks such as9/11 and natural disasters
like the 2004 Indian Ocean tsunami [4]. Nowadays, law
enforcement agencies in the US routinely photograph and
catalog tattoo patterns for use in identifying victims and
convicts. The NIST Tatt-C and Tatt-E challenges have been
helpful in advancing the development of tattoo detection
and identiﬁcation systems for real application scenarios [5].
Despite the value of tattoos for assisting in person
identiﬁcation, putting it to practical use has been difﬁcult.
Unlike primary biometric traits, much variability exists in
pattern types of tattoos. The early use of tattoo for assisting
in person identiﬁcation relied heavily on manual annota-
tions and comparisons. This has motivated the study of
automatic identiﬁcation algorithms for tattoos [4], [5], [6],
[7], [8], [9], [10], [11].
Despite the progress in tattoo retrieval, existing methods
have some serious limitations. In fact, most of the current
practice of tattoo matching aims at tattoo identiﬁcation,
and not learning a compact representation for efﬁcient
tattoo search at scale. More importantly, existing tattoo
identiﬁcation methods primarily focus on matching cropped
tattoos, which does not replicate the in-situ scenarios, where
the search must be operated in raw images or video frames.
In addition, while sketch to photo matching has been widely
studied in the areas such as image retrieval [12], [13] and
face recognition [14], [15], research on sketch based tattoo
search is very limited [10].
1.1 Proposed Approach
To overcome the above limitations of the current tattoo
search methods, we present a joint detection and compact
representation learning approach for tattoo search at scale.
The proposed approach is motivated by recent advances in
object detection and compact representation learning, but
takes into account the unique challenges in a tattoo search
domain, such as large intra-class variability, poor image
quality, and image deformations (see Fig. 2). In addition,
the proposed approach can be trained in a fully end-to-
end fashion, and can leverage additional operational data
to improve the tattoo search.
As shown in Fig. 3, the proposed approach handles
tattoo detection and compact representation learning in a
single convolutional neural network (CNN) via multi-task
(a)
(b)
Fig. 2. Tattoo images from the Tatt-C dataset [5] suggest that tattoo
search is challenging because (a) there are various types of tattoos
containing categories such as humans, animals, plants, ﬂags, objects,
abstract, symbols, etc., and (b) the intra-class variability in one class of
tattoos (i.e., eagle) can be very large.
learning. Given an input image, a shared feature map is ﬁrst
computed via a deep CNN network, which is then fed into
individual sub-networks, which aim at tattoo detection and
compact feature learning tasks, respectively.
The main contributions of this paper include: (i) the
ﬁrst end-to-end trainable approach for joint tattoo detection
and compact representation learning in a single network
allowing more robust and discriminative feature learning
via feature sharing; (ii) effective strategies in resolving small
batch size issue w.r.t. the compact representation learning
module of the network; (iii) superior performance and much
lower computational cost compared to the state-of-the-art
algorithms; and (iv) compiling a dataset with 300, 000 tattoo
images in the wild and thousands of annotations for large-
scale tattoo search, and a dataset with 300 tattoo sketches
(see Fig. 10 (b)) for sketch-based tattoo search; both datasets
will be put into the public domain.
Our preliminary work of this research is described in
[10]. Essential improvements over [10] include: (i) use of
task-driven learned features instead of hand-crafted features
for joint tattoo detection and compact representation learn-
ing in a single network; (ii) extensions to large-scale tattoo
search via compact feature learning; and (iii) compiling a
sketch dataset for studying sketch-based tattoo search.
The remainder of this paper is structured as follows. We
brieﬂy review related literature in Section 2. The details
of the proposed tattoo search approach are provided in
Section 3. In Section 4, we introduce the WebTattoo and
tattoo sketch datasets, and provide the experimental results
and analysis. Finally, we conclude this work in Section 5.
2 R ELATED WORK
2.1 Tattoo Identiﬁcation and Retrieval
In the following, we brieﬂy review the literature on tattoo
identiﬁcation and retrieval, covering detection, feature rep-
resentation, databases, and performance (see Table 1).
The early practice of tattoo image retrieval relied on
keywords or metadata based matching. For example, law
enforcement agencies usually follow the ANSI/NIST-ITL 1-
2000 standard [6] for assigning a single keyword to each
3
tattoo image in the database. However, a keyword-based
tattoo image retrieval has several limitations in practice
[1]: (i) The classes deﬁned by ANSI/NIST-ITL offer a lim-
ited vocabulary which is insufﬁcient for describing various
tattoo patterns; (ii) multiple keywords may be needed to
adequately describe a tattoo image; (iii) human annotation
is subjective and different subjects can give dramatically
different labels to the same tattoo image.
These shortcomings of keyword-based tattoo image re-
trieval systems have motivated the development of content-
based image retrieval (CBIR) techniques to improve the
tattoo search efﬁciency and accuracy [1], [4], [7], [8], [9],
[10], [16]. CBIR aims to extract features, e.g., edge, color,
and texture, that can reﬂect the content of an image, and
use them to identify images with high visual similarity. For
example, color histogram and correlogram, shape moments,
and edge direction coherence features were used in [4], [7]
for tattoo matching. Similarly, global and local features of
edge and color were used in [8], and vector-wise Euclidean
distance was computed to measure the similarity between
two tattoo images. The bag-of-words (BoW) model [25]
using SIFT [26] features is probably the most popular one
among the early CBIR systems for tattoo search [1], [9], [10],
[16], [17]. Besides SIFT features, LBP-like features and HoG
features were also used in [18], [19] with SVM and random
forest classiﬁers for tattoo classiﬁcation. While these CBIR
systems are reported to provide reasonably high accuracies
on various benchmarks, they require careful handcrafting of
feature descriptor, vocabulary size, and indexing algorithm.
With the success of deep learning in many computer
vision tasks [27], the focus of CBIR methods is shifting
from handcrafted features and models to deep learning
based methods [28]. In particular, AlexNet [29], winning the
ImageNet challenge of 2012, has been successfully used for
tattoo vs. non-tattoo classiﬁcation in [11], [23], [24]. Faster
R-CNN [30] was used for both tattoo vs. non-tattoo image
classiﬁcation and tattoo localization in [22]. Among these
methods [11], [22], [23], only [23], [24] studied the tattoo
identiﬁcation using a Siamese network with triplet loss.
There are some studies on logo or landmark
search, which faces similar challenges to tattoo search,
e.g., geometric deformation, inverted brightness, etc. Due
to these challenges, image search algorithms based on the
traditional BoW representation often fail. To resolve these
issues, great efforts have been made to improve the robust-
ness [31], [32] and efﬁciency [33], [34] of the descriptors.
Due to space limitation, we refer interested readers to recent
reviews of general image retrieval, e.g., [28].
The current practice of tattoo matching in the literature
is towards tattoo identiﬁcation, and not learning a compact
representation for efﬁcient large-scale tattoo search. Besides,
most of the existing tattoo identiﬁcation methods (including
the logo or landmark search algorithms) focus on match-
ing cropped instances (where instance of interest has been
segmented from the background), which is different from
real application scenarios, where the images are usually
uncropped. Even for the deep learning based methods such
as [11], [22], [23], none has addressed tattoo detection and
compact representation learning jointly. The most related
work of joint detection and representation learning was
reported in face recognition [35], in which two large-scale
face datasets, i.e., WIDER FACE [36] (containing 393,703 face
bounding boxes in 32,203 images, and an average of 12 faces
per image) and CASIA WebFace [37] (containing 494,414
face images of 10,575 subjects), were used to learn their
end-to-end face detection and recognition network. Given
the large number of faces per image on average, the small
batch size is not an issue in training the recognition part of
their joint detection and recognition network. In contrast,
there is usually a single tattoo instance in each image in
most of the tattoo datasets; this results in small batch size
issue in training the recognition part of our joint tattoo
detection and representation learning network because there
is only one input image in each iteration. These unique
challenges require design of novel end-to-end detection and
representation learning approach. In addition, while we aim
for efﬁcient large-scale tattoo search and sketch-based tattoo
search by performing detection and compact representation
learning jointly, [35] did not report results in such a scenario.
2.2 Compact Representation Learning
Compact representation learning is of particular interest
because of the need for efﬁcient methods in large-scale
visual search and instance retrieval applications [28], [38],
[39], [40], [41], [42]. Compared with high-dimensional real-
valued representations, compact representations aim to ob-
tain a compressive yet discriminative feature. Feature index-
ing, i.e., through various quantization or hashing functions,
is a major approach to obtain compact representations.
Quantization based feature indexing methods are de-
signed to quantize the original real-valued representation
with minimum quantization errors, and thus usually have
high search accuracy [31], [43], [44], [45], [46], [47]. Com-
pared with quantization based methods, hashing based
feature indexing methods generates binary codes, providing
faster retrieval speed since the Hamming distance of two
binary codes can be computed via the native bit-wise oper-
ations. The published hashing based methods for compact
representation can be categorized into two major classes:
unsupervised and supervised hashing.
Unsupervised hashing algorithms, e.g., [48], [49], [50],
[51], [52], [53], [54] use unlabeled data to generate binary
codes which aim to preserve the similarity information in
the original feature space. Unsupervised hashing methods
are often efﬁcient in computation, but their performance
in large-scale retrieval may not be the optimum since no
label information, including weak labels such as pairwise
relationship about a dataset is utilized. To address this lim-
itation, supervised hashing approaches, e.g., [55], [56], [57],
[58], [59] have been proposed to learn more discriminative
binary codes by leveraging both the label similarity and
semantic similarity in the feature values of the data and
label information. Deep neural networks have also been
used to learn compact binary codes from high-dimensional
inputs [39], [60], [61], [62]. With the advances in CNN
architectures and ﬁne-tuning strategies, the performance of
the deep hashing methods is improving, and has provided
good generalization ability into new datasets [28]. Due to
the limited space, we refer readers to [63], [64] for a survey
of data hashing approaches.
While there are a large number of approaches on hashing
based compact representation learning, most of the pub-
4
TABLE 1
A summary of published methods on tattoo identiﬁcation and retrieval.
Publication Detection model Feature and retrieval model Tattoo database
#images (query; target) Results
Jain et al. [1], [4]
(2007,2012) Gradient thresholding
Color histogram and correlogram;
shape moments;
edge direction coherence;
Fusion of per feature similarities
Tattoos from web
(2, 157; 43, 140)1 46% prec.@60% recall
Acton and Rossi [8]
(2008)
Active contour
segmentation and
skin detection
Global and local features of
edge and color;
Vector-wise Euclidean distance
Recreational (30;∼ 4, 000)2;
Gang (39;∼ 4, 000)2
Recreational: 94.7% acc.@rank-1;
Gang: 82.2% acc.@rank-1
Jain et al. [16]
(2009) Pre-cropped tattoos
SIFT features with geometric constraint;
indexing with location and keyword;
Keypoint-wise matching
MSP (1, 000; 63, 592) 85.9% acc.@rank-1
Li et al. [17]
(2009) n/a SIFT features;
Bag-of-words; Re-ranking
MSP and ESP
(995; 101, 754)3 67% acc.@rank-1
D. Manger [9]
(2012) n/a
SIFT features;
Bag-of-words, hamming embedding,
and weak geometry consistency
German police
(417; 327, 049) 78% acc.@rank-1
Heﬂin et al. [18]
(2012)
Automatic GrabCut
and quasi connected
components
LBP-like features, SVM Tattoo classiﬁcation
(50; 500)4
85% acc.@10% FAR on average,
for 15 classes
Han and Jain [10]
(2013) Pre-cropped tattoos SIFT features;
Sparse representation classiﬁcation
MSU Sketch Tattoo
(100; 10, 100) 48% acc.@rank-100
Wilber et al. [19]
(2014) Pre-cropped tattoos Exemplar code using HoG features;
Random forest classiﬁer 238 tattoos of 5 classes 63.8% avg. acc. for 5 classes
Xu et al. [20]
(2016)
Skin segmentation
and block based
decision tree
Boundary features;
Shape matching via
coherent point drift
Full body tattoo sketch
(547; 1, 641) 52.38% acc.@rank-50
Kim et al. [21]
(2016) Graphcut n/a
Tatt-C (Detection):
6, 308 images;
Evil (Detection):
1, 105 images
Tatt-C: 70.5% acc.@41%recall
Evil: 69.9% acc.@67.0%recall
Xu et al. [11]
(2016)
Modiﬁed AlexNet
(tattoo vs. non-tattoo) n/a
Tatt-C (tattoo vs. non-tattoo)
(1, 349; 1000)4;
Flickr (tattoo vs. non-tattoo)
(5, 740; 4, 260)4
Tatt-C (tattoo vs. non-tattoo): 98.8%
Flickr (tattoo vs. non-tattoo): 78.2%
Sun et al. [22]
(2016) Faster R-CNN n/a
Tatt-C: tattoo vs. non-tattoo
(1, 349; 1000)4;
Flickr: tattoo vs. non-tattoo
(5, 740; 4, 260)4
Tatt-C (tattoo vs. non-tattoo): 98.25%
Tatt-C (localization): 45%@0.1FPPI
Flickr (tattoo vs. non-tattoo): 80.66%
Di and Patel [23], [24]
(2016)
AlexNet and SVM
(tattoo vs. non-tattoo)
Siamese network with triplet or
contrastive loss
Tatt-C: tattoo vs. non-tattoo
(1, 349; 1, 000)4;
mixed media (181; 55)
Tattoo vs. non-tattoo: 99.83%
Mixed media: 56.9% acc.@rank-10
Proposed approach Deep end-to-end learning for joint detection and
compact representation learning
Tatt-C:
detection: 7, 526 images
identiﬁcation (157; 4, 375);
Flickr (detection):
5, 740 images;
DeMSI (identiﬁcation):
890 images;
WebTattoo (500,∼300K)
Detection (localization)
Tatt-C: 61.7% recall@0.1FPPI
WebTattoo:87.1% recall@0.1FPPI
Tattoo search
WebTattoo (photo):
60.1% mAP (w/o background)
25.3% mAP (300K background)
WebTattoo (sketch):
37.2% mAP (w/o background)
Tattoo identiﬁcation
WebTattoo:
63.5% acc.@rank-1 (w/o background)
28.0% acc.@rank-1 (300K background)
Tatt-C: 99.2% acc.@rank-1
1Twenty different image transformations were applied to 2, 157 tattoo images to generate 43, 140 synthetic tattoo images. 2Forty different image
transformations were applied to 100 tattoo images to generate 40, 000 synthetic tattoo images. 340, 000 images were randomly selected from the
ESP game dataset to populate the tattoo dataset. 4(a, b) denotes the number of positive and negative tattoo images per class. 5(a, b) denotes the
number of tattoo and non-tattoo images.
lished methods assume pre-cropped images of instances;
but in a fully automatic instance retrieval system, such an
assumption usually does not hold. In addition, most of
the published methods on compact representation learning
are designed for computer vision tasks such as face image
or natural image retrieval, their performance in large-scale
tattoo search is not known.
3 P ROPOSED METHOD
3.1 Review of Faster R-CNN
Faster R-CNN [30] is one of the leading object detection
frameworks to identify instances of objects belonging to
certain classes and localize their positions (bounding boxes)
in an end-to-end learning network. Faster R-CNN consists
of two modules. The ﬁrst module, called the Region Pro-
posal Network (RPN), is a fully convolutional network for
generating regions of interest (RoI) that denote the possible
presence of objects. The second module is Fast R-CNN [65],
whose purpose is to classify the RoI by RPN into individual
classes and reﬁne the positions of each foreground instance.
By sharing the deep features of the full image between RPN
and Fast R-CNN, Faster R-CNN is able to perform object
detection accurately and efﬁciently, and can be trained in an
end-to-end fashion.
As aforementioned, conventional methods usually break
down the tattoo search problem into two separate tasks,
i.e., tattoo detection [11], [22], and tattoo matching [7], [10],
5
Convolutional
layer
...
Pooling
layer
Fully connected
layer
...
RPN
cls_scores
bbox_pred
proposals
...
Compact Feature Learning
Inference
Query
1
0
.
.
0
1 1 0 . . . 0 1
0 1 . . . 1 0
0 1 . . . 1 1
1 1 . . . 0 1
. . .
Tattoo search
Gallery dataset
cls_scores
bbox_pred
Fast R-CNN
cls_loss
dispersity
sigmoidlatent
polarization
FCB2 FC lPB
( )WFC
FCB3
C 1 C 2 P
1 P
sC
t... ... ...
FCB1
CNN
(e.g., ResNet)
( )×= HB k
i
( )×D
( )k
iB
5.0G
Stitched tattoo image
Training
set
Fig. 3. Overview of the proposed approach for tattoo search at scale via joint tattoo detection and compact representation learning. Our approach
consists of a stem CNN for computing the shared features, an RPN [30] and Fast R-CNN [65] for tattoo detection, and a compact representation
learning module. The proposed approach can be trained end-to-end via stochastic gradient descent (SGD) [29] and back-propagation (BP) [66].
[19]. Such a scheme is not optimum because the matching
task could assist in the detection task, and the detection
accuracy inﬂuences the feature discriminability used by the
matching task. Therefore, while Faster R-CNN provides an
efﬁcient solution for object detection from images, it ad-
dresses only the front-end detection problem of an instance
retrieval system. Our observation is that the convolutional
feature maps used by RPN and Fast R-CNN can also be
used for learning compact representation.
3.2 Joint Detection and Compact Representation
Learning
We aim to handle tattoo detection and compact representa-
tion learning simultaneously via a single model (see Fig. 3).
A straightforward method for handling tattoo detection and
compact representation learning jointly is to use a cascade
of tattoo detection and compact representation learning,
i.e., the output of the detector is fed into the succeeding fea-
ture extraction module. However, such a cascaded method
does not leverage feature sharing to achieve efﬁcient and
robust representation learning.
Formally, let A ={X, Y} be a training tattoo dataset,
where X = {Xi}N
i=1 denotes N tattoo images from K
distinct tattoos, and Y =
{
{Yj
i}M
j=1
}N
i=1
denotes the M
labels for the corresponding tattoo images. Here, the label
of each tattoo image consists of two elements (thus M = 2),
i.e., the tattoo position (
{
Y1
i
}N
i=1) and class ID (
{
Y2
i
}N
i=1).
Given such a training dataset, we expect to jointly opti-
mize a tattoo detector D(·) and a compact representation
learning function H(·), which can minimize a regression
loss (ℓreg) between the predicted and ground-truth bound-
ing boxes, and a classiﬁcation loss ( ℓcls) of the compact
representations describing individual detected tattoos, re-
spectively. For the regression loss ( ℓreg), we choose to use
the robust smoothL1 loss [65]
ℓreg
(
D(Xi),Y 1
i
)
=
∑
d∈{u,v,w,h}
SL1 (D(Xi){d}− Y1,d
i ),
(1)
where the four-tuple{u,v,w,h } speciﬁes the top-left loca-
tion (u,v ) and the width and height (w,h ) of a detected
tattoo, and the four elements are indexed byd. The function
SL1 (·) is deﬁned as
SL1 (z) =
{
0.5z2 if|z|< 1
|z|− 0.5 otherwise. (2)
For the classiﬁcation loss ( ℓcls), deﬁned w.r.t. the detected
tattoos, we choose to use the cross-entropy loss [67]
ℓcls =−
N∑
i=1
K∑
k=1
1
(
ˆYk
i,Y 2
i
)
logp
(
ˆYk
i
)
, (3)
where ˆYk
i =WFC· Bk
i =WFC· H (Xi, D(Xi)){k} denoting
the k-th element of the output by a fully connected layer
with weightWFC , which takes feature Bk
i as its input. H(·)
takes image Xi and the detected tattoo location D(Xi) as
input, and outputs Bk
i . 1
(
ˆYk
i,Y 2
i
)
outputs 1 whenk =Y 2
i ,
and 0 otherwise. The probabilityp(·) is computed as
p( ˆYk
i ) = e ˆYk
i
∑K
k=1e ˆYk
i
. (4)
By minimizing the losses given in (1) and (3), we can
jointly perform tattoo detection and feature representation
learning from the detected tattoos. However, additional
constraints are still required to guarantee that the learned
features are compact binary features, which is important
for efﬁcient large-scale search. Therefore, we expect that the
features, i.e., Bk
i = {bk|bk ∈ {0, 1}}K
k=1,i = 1 , 2,··· ,N ,
learned by H(·) should be near-binary codes. This implies
that each element Bk
i of a feature vector should be close
to either 1 or 0. Such an objective can be approximated by
penalizing the learned feature to have elements close to 0.5
ℓpol(Bi) = 1
1
2K
∑K
k=1∥Bk
i− 0.5∥2
2 +ϵ
, (5)
in which ϵ is a small positive constant for avoiding divide-
by-zero, and we use ϵ = 0.01 in our experiments. We call
such a loss in (5) as a polarization loss.
6
Polarization 
loss
(a) Original feature (b) Polarized feature 
Dispersity loss
(c) Dispersed feature
 (d) Polarized and dispersed feature
Join
t loss
Fig. 4. The beneﬁt of using the polarization loss and dispersity loss. (a)
an original feature vector of real values in the range of [0, 1], (b) the
learned feature vector after using the polarization loss alone ( i.e., each
element is close to either 0 or 1), (c) the learned feature vector after
using the dispersity loss alone ( i.e., the elements are evenly distributed
on the two sides of 0.5), and (d) the learned feature vector after jointly
using the polarization and dispersity losses ( i.e., near-binary elements
are evenly distributed on the two sides of 0.5).
In addition, we expect that every bit in a binary code of
length ι can contribute to the representation of individual
tattoos. In other words, each bit of the binary code is
expected to have a 50% ﬁre rate [41]. Such an objective
can be approximated by constraining the average value of a
learned feature to be 0.5, i.e.,
ℓdis(Bi) = 1
ι
ι∑
k=1
Bk
i− 0.5 (6)
We call such a loss in (6) as a dispersity loss.
By minimizing the losses deﬁned in (3), (5) and (6)
jointly, the learned features are expected to be discriminative
near-binary codes that are evenly distributed in the feature
space (see Fig. 4). Such near-binary codes can be easily
converted into a binary code utilizing a threshold function
B′k
i =T (Bk
i ) =
{
0 if Bk
i < 0.5
1 otherwise . (7)
Finally, we compute the distance between a query tattoo
image and a gallery tattoo image using the Hamming dis-
tance of the binary vectors. In case more than one tattoos
are detected from an image, i.e.,u andv tattoos are detected
from a query image and a gallery image, respectively, we
compute u× v distances in total, and use the minimum
distance as the ﬁnal distance between the two tattoo images.
3.3 Network Structure
While our aim of joint tattoo detection and compact repre-
sentation learning is well deﬁned by (1), (3), (5), and (6),
the feature sharing between detection and representation
learning tasks is not simple.
We propose to perform joint tattoo detection and com-
pact representation learning based on a Faster R-CNN
model by embedding the above four losses into a single
network. Speciﬁcally, a CNN such as AlexNet [29], VGG
[68], or ResNet [69] can be used to extract the deep features
that are to be shared by RPN and Fast R-CNN. The RPN and
Fast R-CNN modules, and the tattoo regression loss deﬁned
in (1) in the proposed approach are the same as those in the
original Faster R-CNN (see Fig. 3).
We establish joint compact representation learning by
introducing a new compact representation learning (CRL)
sub-network (see Fig. 3). CRL consists of an instance pooling
layer ( e.g.,PB), a sequence of fully connected (FC) layers
(e.g.,FCB1 and FCB2), a latent layer with sigmoid activa-
tion (e.g.,FCl), and three sibling output layers (e.g., cls loss,
polarization, and dispersity). The instance pooling layer and
the sequence of FC layers are the same as the RoI pooling
and FC layers in Fast R-CNN, which compute a ﬁxed-
length feature vector from the shared feature map for each
tattoo detection by Fast R-CNN, and optimize this feature
w.r.t. the following CRL tasks. The latent layer (also an
FC layer) with sigmoid activation is expected to generate
near-binary representation Bi that will be binarized via
(7) for large-scale search. The three sibling output layers
model the corresponding constraints, i.e., polarization loss
in (5), dispersivity loss in (6), and classiﬁcation loss in (3),
respectively (see Fig. 3). Overall, the three loss functions
work in a multi-task way to perform CRL given a tattoo
detection. We use hyper-parameters to control the balance
between individual losses
ℓJ (Bi) = αℓcls +βℓpol +γℓdis (8)
We setα =β =γ = 1 based on empirical results. It should
be noted that although there are K classes of tattoos in the
training dataset, the number of outputs in both RPN and
Fast R-CNN remains two (corresponding to background
and tattoo). The number of outputs for classiﬁcation loss
layer in our CRL is K.
The proposed approach differs from [70], [71] in that:
(i) [70] optimizes the feature representation for instance
search by ﬁne-tuning the detection network w.r.t. the query
instances; however such features optimized for detection
tasks may not be optimal for retrieval tasks. By contrast,
the proposed approach jointly optimizes both detection and
feature representation end-to-end; (ii) while the features
used in [70], [71] are real-valued, the proposed approach
learns compact binary features, which are scalable; (iii)
the bounding boxes used for computing the features for
instance matching in [71] are not accurate compared to
the ﬁnal bounding box estimates used by the proposed
approach; however, as we will show in the experiments,
accurate tattoo bounding boxes are important for improving
the tattoo matching accuracy; (iv) while [71] studied person
search with a gallery set containing about 6, 978 images,
the scalability of the proposed approach is studied with a
gallery dataset containing more than 300K distracter tat-
too images; and (v) while cross-modality instance search,
i.e., sketch based tattoo search is studied in our work, the
performance of [71] under a cross-modality matching sce-
nario is not known.
3.4 Implementation Details
Data Augmentation. Compared with the large-scale
databases for object detection [72] and image classiﬁcation
[73], the public-domain tattoo datasets such as Tatt-C [5],
Flickr [11], and DeMSI [74] are of limited sizes (usually less
7
(a) (b) (c) (d) (e) (f) (g) (h)
(i) (j) (k) (l) (m) (n) (o) (p)
sketch
original
Fig. 5. An example of data augmentation for one tattoo image (referred to as “original” in the top row) in the training set to replicate various image
acquisition conditions, e.g., (a-d) illumination variation, (e-f) image blur, (g-l) deformation, and (m-p) perspective distortion. A tattoo sketch is also
generated for data augmentation (shown under the original tattoo).
than 10K). The limited datase size poses additional chal-
lenges to the proposed approach, i.e., the risk of overﬁtting,
particularly for our CRL module, which usually requires
multiple tattoo images per class and a large number of
tattoo classes to learn a robust model. Besides the commonly
used data augmentation methods ( e.g., random crop, trans-
lation, rotation, and reﬂection) [29], we have designed 16
additional transformations7 to replicate the diversity of one
tattoo instance caused by various acquisition conditions (see
Fig. 5). In addition, we have also generated a tattoo sketch
(see Fig. 5) for data augmentation so that the proposed
approach can generalize to sketch-based tattoo retrieval
task.
Network Training. We use ResNet-50 as our backbone
network for shared feature learning, which is pretrained on
ImageNet [73] for parameters initialization. We use a conﬁ-
dence threshold of 0.8 to ﬁlter the tattoo detections by Fast
R-CNN, which corresponds to about one false detection for
every ten images, on average. The ﬁltered tattoo detections
are fed into CRL. The reason why we use a relatively high
threshold is to avoid feeding non-tattoo detections into CRL,
wich may cause difﬁculty in network convergence. We use
a learning rate of 10−4 during the ﬁne-tuning of the pre-
trained CRL. For the parameters of RPN and Fast R-CNN,
we directly use the suggested values in [30], [65].
Since our approach performs tattoo detection and CRL
jointly, the detection module usually takes one image as in-
put (given a typical Titan X GPU), and outputs one detected
tattoo, which is then used as the input to CRL. Thus, the
batch size w.r.t. CRL is limited to one tattoo, which makes
it difﬁcult for CRL to converge. Such an issue cannot be
resolved by just compiling a large training dataset or using
data augmentation as in Fig. 5. To address this issue, we
make use of preceding feature buffering [71] to assist in CRL
training. In addition, we stitch multiple randomly selected
training images into a single image (see Fig. 3), and use it
as the input to our detection module so that it can output
multiple detected tattoos. In this way, multiple detected
tattoos can be used for training CRL, and thereby improving
the training batch size. We have found such stitched tattoo
images to be very for training our joint tattoo detection
and CRL network. We also use online hard example mining
(OHEM) [75] in Fast R-CNN to improve its robustness in
detection blurred, partial, and tiny tattoos. All the tattoo
7. All the augmentations were performed leveraging the Photoshop
plug-ins for MATLAB: https://helpx .adobe.com/photoshop/kb/do
wnloadable-plugins-and-content .html
images are scaled before they are input to the network so
that the shorter edge between width and height is600 pixels.
4 E XPERIMENTS
4.1 Databases
There are only a limited number of tattoo databases in the
public domain, such as Tatt-C [5], Flickr [11], and DeMSI
[74]. These tattoo databases are used in the evaluations of
our approach and comparisons with state-of-the-art.
Tatt-C. The NIST Tatt-C database was developed as
an initial tattoo research corpus that addresses use cases
representative of operational scenarios [5]. The tattoo vs.
non-tattoo classiﬁcation dataset in Tatt-C contains1, 349 and
1, 000 tattoo and non-tattoo images, respectively. The tattoo
identiﬁcation dataset in Tatt-C contains 157 and 215 probe
and gallery images, respectively. A background dataset with
4, 332 non-tattoo images was also used to populate the
gallery set. The tattoo mixed-media dataset in Tatt-C, con-
sisting of photos, sketches, and graphics, contains 181 and
272 probe and gallery images, respectively. A ﬁve-fold cross-
validation was used for the identiﬁcation experiment of each
dataset in Tatt-C [5]. We also notice that the bounding-box
annotations were provided for7, 526 tattoo images in Tatt-C,
so we also report the tattoo detection accuracy using these
image in Tatt-C. The tattoo images in the Tatt-C dataset
contain variations of illumination, partial occlusion, and
image blur (see Fig. 6 (a)).
Flickr. The Flickr tattoo database contains 5, 740 and
4, 260 tattoo and non-tattoo images that were collected from
Flickr [11]. We can notice that the ratio of the tattoo images
to the non-tattoo images is similar to that of the Tatt-C
database. While the tattoo images in the Tatt-C database
were collected from an indoor environment, the images
in the Flickr database were taken from both indoor and
outdoor environment, with diverse viewpoints, poses, and
complex backgrounds (see Fig. 6 (b)). The Flickr database
was original built for tattoo vs. non-tattoo classiﬁcation. We
have extended the Flickr database by providing bounding-
box annotations for each tattoo in the images. 8 Therefore, in
our experiments, we are able to use the Flickr database to
evaluate the tattoo detection (localization) performance.
DeMSI. The DeMSI dataset contains 890 tattoo images
from the ImageNet [73] database. The boundary of each
8. We will put the bounding-box annotations for the Flickr database
into the public domain.
8
(a) Tatt-C (b) Flickr (c) DeMSI (d) WebTattoo
Fig. 6. Examples of tattoo images from the four tattoo image databases used in our experiments: (a) Tatt-C [5], (b) Flickr [11], (c) DeMSI [74], and
(d) our WebTattoo dataset.
tattoo image was annotated for tackling the tattoo segmen-
tation problem [74]. Since the tattoo images are from the Im-
ageNet database, they are captured under an unconstrained
scenario (see Fig. 6 (c)). The tattoo images from the DeMSI
dataset are used together with our WebTattoo databases as
the background images to populate the gallery dataset.
WebTattoo. We can notice that the above tattoo datasets
are usually of limited sizes (less than 10K). Although the
NIST Tatt-E challenge is reported to have a much larger
tattoo testing dataset collected from real application sce-
narios9, there is no evidence this dataset will be put into
the public domain. To replicate the operational scenario
of tattoo search at scale, we have compiled a large tattoo
database (named as WebTattoo) by (i) combining the above
three public-domain tattoo databases together, (ii) collecting
over than 300K distracter tattoo images from the Internet
(see Fig. 6 (d)), and (iii) drawing 300 tattoo sketches by
volunteers, who were asked to take a look at a tattoo image
for one minute and then draw the tattoo sketch the next day
(see an example of the tattoo sketch in Fig. 5). 10
Based on the WebTattoo dataset, we use a semi-
automatic approach to ﬁnd the tattoo classes which have
multiple tattoo images per class. Speciﬁcally, a ResNet-50
network pre-trained on ImageNet is ﬁrst used for automatic
tattoo feature extraction and clustering (we used k-means
clustering [76]). The clusters are then manually veriﬁed to
assure that each cluster contains only one class of tattoo
images. Finally, we obtained about 600 tattoo classes, with
nearly three tattoo images per class on average. We ran-
domly choose about 1, 400 tattoo images from 400 tattoo
classes for training, and use tattoo images of the remaining
200 tattoo classes for testing. The training set is augmented
using the method described in Sect. 3.4. We have manually
annotated the tattoo bounding boxes for more than 78K
tattoo images (original and augmented tattoo images) in
total. Given such a large number of annotations, there are
inevitably some missed tattoos by human workers (see Fig.
7 (b)). However, such issues are not unique in our tattoo
search task; they exist in many databases for individual
computer vision tasks, such as face detection and recog-
nition, person detection and recognition, etc. In addition
to the data augmentation, we also use the 5, 740 tattoo
9. https://www.nist.gov/programs-projects/tattoo-recognition-tec
hnology-evaluation-tatt-e
10. We plan to put the WebTattoo dataset into the public-domain.
images from the Flickr dataset for training. Since no class
label is provided for the tattoo images in Flickr, these tattoo
images contribute only to the detection loss in the proposed
approach during network training. For each class of tattoos
in the testing set, we randomly choose one tattoo image
for the query, and use the remaining tattoo images for the
gallery. Overall, we have 200 tattoo images in the query and
350 tattoo images in the gallery. About 300K WebTattoo im-
ages that are not present in the training, gallery, and query
sets, are used as the distracter tattoo images to populate
the gallery set, and replicate the large-scale tattoo search
scenario. For the 300 pairs of tattoo sketches and the mated
tattoo photos, we randomly choose 240 pairs for training,
and the remaining 60 pairs for testing. For each pair of tattoo
sketch and image, the tattoo sketch is used for query, and the
tattoo image is used for gallery.
An operational tattoo dataset reported in [9] contains
327, 049 tattoo images collected by the German police.
Another operational tattoo dataset reported in [1] contains
about 64, 000 tattoo images, provided by the Michigan
State Police. Our extended gallery set contains more than
300K tattoo images, which should reasonably replicate the
operational tattoo search scenario.
4.2 Evaluation Metrics
The evaluations of the proposed approach and the com-
parisons with the state-of-the-art tattoo retrieval and
identiﬁcation methods cover the tasks of tattoo detection,
identiﬁcation, and large-scale search. For each task, we
choose to use the widely used evaluation metric in the
literature.
Tattoo detection. We use the detection error trade-off
(DET) curve to measure the tattoo detection performance,
i.e., the recall vs. false positives per image (FPPI).Given
an intersection-over-union (IoU) threshold (we use 0.5) be-
tween the detected tattoo bounding boxes and the ground-
truth tattoo bounding boxes, recall is deﬁned as the fraction
of detected bounding boxes with an IoU to the ground-truth
larger than the threshold over the total amount of ground-
truth bounding boxes.
Tattoo search. We use the precision-recall curve to mea-
sure the tattoo search performance. Precision is the fraction
of the mated tattoo images that have been retrieved over all
the retrieved results for a given query tattoo. Recall, similar
to that in the detection task, is the fraction of the mated
9
(a) (b)
Fig. 7. Examples of (a) good tattoo detections, and (b) poor tattoo detections by the proposed approach. The green and blue rectangles show the
ground-truth tattoo bounding boxes and the detected tattoo bounding boxes, respectively. The numbers shown above the bounding boxes are the
detection conﬁdence scores.
tattoo images that have been retrieved over the total amount
of mated tattoo images for a given query tattoo.
Tattoo identiﬁcation. We use the cumulative match char-
acteristic (CMC) curve to measure the tattoo identiﬁcation
performance. Each point on CMC gives the fraction of the
probe tattoo images that are correctly matched to their
mated gallery images at a given rank.
4.3 Tattoo Detection
Since the proposed approach can perform tattoo detec-
tion and compact representation learning jointly, we ﬁrst
evaluate the tattoo detection performance of the proposed
approach on the WebTattoo test and Tatt-C datasets. Specif-
ically, we train our approach using the WebTattoo training
set, and report the tattoo detection accuracy on the Web-
Tattoo test and Tatt-C datasets. Since the Tatt-C dataset was
primarily built for tattoo vs. non-tattoo classiﬁcation and
tattoo identiﬁcation tasks, only a limited number of pub-
lished methods have reported tattoo detection performance
on Tatt-C [22]. To provide more baseline performance, we
train a Faster R-CNN tattoo detector used in [22] on the
WebTattoo training dataset, and report its performance on
the WebTattoo test and Tatt-C datasets. We should note that
such a cross-database testing protocol is more challenging
than the intra-database testing protocol used in [22].
Table 2 lists the tattoo detection performance of the
proposed approach and the baseline methods. The state-of-
the-art tattoo detection method in [22] reported about 45%
recall @ 0.1FPPI on a Tatt-C dataset with about 2, 000 tattoo
images (one tattoo per image). The Faster R-CNN tattoo de-
tector we trained gives 56.2% and 80.2% recalls at 0.1FPPI
on the Tatt-C and WebTattoo test datasets, respectively. The
performance of the Faster R-CNN tattoo detector we trained
is much higher than that in [22], even though we are using
a challenging cross-database testing protocol, and the Tatt-C
subset we used for evaluation contains much more tattoo
images than that was used in [22] ( 7, 526 vs. 2, 000 tattoo
images). The possible reason is that the WebTattoo training
set contains more tattoo images than those used in the
intra-database testing, which is helpful for training a deep
TABLE 2
Tattoo detection (localization) performance of the proposed approach
and the state-of-the-art methods on the WebTattoo test and Tatt-C
datasets in terms of recall vs. FPPI.
Method Recalls (in %) @ different FPPIs
0.01 FPPI
Tatt-C/WebTatt
0.1 FPPI
Tatt-C/WebTatt
1.0 FPPI
Tatt-C/WebTatt
Sun et al. [22]1 8/− 45/− − /−
Faster
R-CNN [30]2 17.1/21.7 56 .2/80.2 72 .2/94.6
Proposed2 45.9/27.5 61.7 /87.1 80.0 /95.5
1The results are from [22], in which a Tatt-C dataset with about
2, 217 tattoo images was used. 2Similar to [22], we trained a Faster
R-CNN tattoo detector using the WebTattoo training set, and tested
it on the Tatt-C (with 7, 526 tattoo images) and WebTattoo test
datasets. This is a cross-database testing scenario, which is more
challenging than that used in [22]. We use an intersection-over-
union (IoU) threshold of 0.5 between the detected and ground-
truth bounding boxes.
learning based tattoo detector. In addition, our data aug-
mentation can replicate the appearance variations existing
in various tattoo images, and thus is helpful to improve
the robustness of the tattoo detector in unseen scenarios.
The proposed approach for joint tattoo detection and CRL
achieves 61.7% and 87.1% recalls at 0.1FPPI on the Tatt-C
and WebTattoo test datasets, respectively, which are much
better than the state-of-the-art tattoo detectors. The results
indicate that the proposed approach can leverage multi-
task learning to achieve robust feature learning and detector
modeling. Another baseline tattoo detection method in [21]
used a Graphcut based method, and reported 70.5% preci-
sion @ 41.0% recall on a Tatt-C dataset with 6, 308 tattoo
images. Under the same evaluation metric, the proposed
approach can achieve 99.0% precision @ 41.0% recall on the
above Tatt-C dataset with 7, 526 tattoo images.
Fig. 7 shows examples of good and poor tattoo detections
by our approach on the WebTattoo database. We ﬁnd that
the proposed approach is quite robust to large pose and
illumination variations as well as the diversity of tattoo
categories. Some of the false detections by the proposed
approach are due to the missed labeling of the tattoos (see
10
(a)
(b)
Fig. 8. Tattoo search performance (in terms of precision-recall) by
the proposed approach and the state-of-the-art methods (TattooID [1],
SSDH-VGG16 [41], Faster R-CNN + SSDH, and OIM-ResNet50 [71])
on the WebTattoo test dataset: (a) without background tattoo images
in the gallery set, and (b) with 300K background tattoo images in the
gallery set; for TattooID, we report its tattoo search performance using
an extended gallery set with only 100K background images because of
its long running time.
the bottom tattoo image in Fig. 7 (b)). However, we notice
that detecting tiny tattoos that are easily confused with the
background region remains a challenging problem.
4.4 Tattoo Search
Efﬁcient tattoo search is important for scenarios, where the
search must be operated in a large volume of raw images
or video frames. We evaluate the our approach for tattoo
search at scale, and provide comparisons with several state-
of-the-art methods [1], [4], [41], [71]. For TattooID [1], [4], we
reimplement it in Matlab because the early algorithm has
been licensed to MorphoTrak 11. For SSDH-VGG16 [41] and
OIM-ResNet50 [71], we directly use the code provided with
their papers, and train the models using the same training
set as our approach. Since SSDH-VGG16 is not able to detect
tattoos from an input image, we also consider another base-
line, i.e., Faster R-CNN is applied for tattoo detection ﬁrst,
and then SSDH-VGG16 is used to extract compact features
for the detected tattoos (Faster R-CNN + SSDH). We have
tried several conﬁdence thresholds (e.g., 0.3, 0.5, 0.7 and 0.9)
for tattoo detection using Faster R-CNN, and ﬁnally chosen
to use a threshold of 0.3, because of its good performance
for the ﬁnal tattoo search. For both SSDH-VGG16 and OIM-
ResNet50, we use 256D feature representations as suggested
in their papers. For our approach, we also use a 256-bit
compact feature for fair comparisons. When the gallery size
is too large, i.e., with 300K background tattoo images in the
gallery, for efﬁciency, we compute the precision-recall using
20 40 60 80 100Recall (in %)
0
10
20
30
40
50
60
70
80Precision (in %)
Proposed (GT-BBox)ProposedSSDH-VGG16Faster R-CNN + SSDH
Fig. 9. The importance of using accurate tattoo bounding boxes for
compact feature learning.
the top-100 retrieval results.
Fig. 8 (a) shows the precision-recall curves of the pro-
posed approach and the state-of-the-art methods for tattoo
search without using the 300K background tattoo images
to populate the gallery set. We are surprised to see that
TattooID, a non-learning based matcher based on SIFT
features, performs better than the deep learning based
method SSDH-VGG16. The main reason is that SSDH-
VGG16 alone is a holistic approach, which learns features
from the entire tattoo images. Since many tattoo images
contain large background regions around the tattoos, such
a feature representation may capture more characteristics
about the background regions than the tattoos, and thus
leads to incorrect matches of tattoo images. This is also
the reason why Faster R-CNN + SSDH achieves better
performance than SSDH-VGG16. OIM-ResNet50, which is
also a joint detection and feature learning method, is able
to leverage the tattoo detection to reduce the inﬂuence
of the background regions of the tattoos. As a result, the
learned features could better represent the content of a
tattoo, and achieves much higher tattoo search accuracy.
However, OIM-ResNet50 extracts the features based on the
region proposals of the tattoos instead of the ﬁnal location
estimations of the tattoos. Such a feature representation is
less accurate than the proposed approach, which utilizes the
ﬁnal location estimations of the tattoos. The proposed joint
tattoo detection and CRL approach performs better than all
the baseline methods. The results suggest that multi-task
learning used in our approach is helpful for learning more
informative representation for tattoo search. In addition,
the proposed approach leverages OHEM to improve the
tattoo detection robustness, and stitched training images
to increase the instance-level batch size during CRL. In
Fig. 9, we also provide the performance of tattoo search
using the ground-truth tattoo bounding boxes to extract our
compact features. The results clearly show that using more
accurate tattoo bounding boxes for compact feature learning
does improve the tattoo matching accuracy. However, we
also notice that the CRL module achieves only 72% rank-1
identiﬁcation rate using the ground-truth tattoo bounding
boxes. The possible reason is that the small size of the
training set (in terms of the number of tattoo classes and
images) has limited the training of the CRL module. There
is still room to improve the performance of the CRL module.
11. https://msutoday.msu.edu/news/2010/msu-licenses-tattoo-
matching-technology-to-id-criminals-victims
11
Fig. 10. Examples of tattoo image search results by the proposed approach using (a) tattoo photos and (b) tattoo sketches as queries. For each
query tattoo image, the top-5 tattoo gallery images in the returned list are given.
40 60 80 100Recall (in %)
0
10
20
30
40
50
60
70Precision (in %)
Proposed-512bitsProposed-256bitsProposed-128bits
Fig. 11. The inﬂuence of the compact binary code length (in bit) to the
tattoo search performance (in terms of precision-recall) by the proposed
approach reported on the WebTattoo test dataset without using back-
ground tattoo images in the gallery set.
After we populate the gallery set using300K background
tattoo images, as expected, all the approaches report de-
creased tattoo search performance (Fig. 8 (b)). Matching the
query tattoo images with the complete 300K gallery using
TattooID would take more than one week on CPU, so we
use 100K background tattoo images for TattooID. Again,
both OIM-ResNet50 and the proposed approach outperform
TattooID and SSDH-VGG16 by a large margin, and our ap-
proach performs better than OIM-ResNet50. This suggests
that the proposed approach remains effective under large-
scale tattoo search scenarios.
We also evaluate the inﬂuence of different code lengths
to the ﬁnal tattoo search performance of our approach. As
shown in Fig. 11, when we increase the code length to 512
bits, there is no performance improvement; instead, a minor
precision drop is observed around 60% recall. If we reduce
the code length to 128 bits, there will be a large performance
drop of the precision. Therefore, we choose to use 256-bit
compact features in all the experiments of our approach.
Fig. 10 (a) shows examples of tattoo search results by
our approach on the WebTattoo database, in which the top-
5 matched gallery images are given for each query tattoo
0 20 40 60 80 100Recall (in %)
0
10
20
30
40
50Precision (in %)
ProposedOIM-ResNet50SSDH-VGG16Faster R-CNN + SSDHTattooID
Fig. 12. Sketch based tattoo search performance (in terms of precision-
recall) by the proposed approach and the state-of-the-art methods
(TattooID [1], SSDH-VGG16 [41], Faster R-CNN + SSDH, and OIM-
ResNet50 [71]) on the WebTattoo test dataset without background tattoo
images in the gallery set.
image. We can see that the proposed approach is robust
against variations of body pose, illumination, and scale.
Some of the incorrectly matched gallery tattoo images by
the proposed approach show high visual similarity to the
query tattoo image (see the second row in Fig. 10 (a)).
4.5 Sketch Based Tattoo Search
In many scenarios, the surveillance image of the crime scene
is not available, so the query is in the form of a sketch of
a tattoo drawn based on the description provided by an
eyewitness (see Fig. 10 (b)). Therefore, it is important to
evaluate the performance of a tattoo search system under
a sketch based tattoo search scenario. SSDH-VGG16 [41],
Faster R-CNN + SSDH, OIM-ResNet50 [71], and our method
that are used in Sect. 4.4, are ﬁne-tuned on the tattoo sketch
training set consisting of 240 pairs of tattoo sketches and
photos, and then evaluated on the tattoo sketch test set.
Fig. 12 shows the precision-recall curves of the pro-
posed approach and the state-of-the-art methods for sketch-
based tattoo search. As expected, as a cross-modality search
problem, tattoo sketch-to-photo matching is much more
12
(a)
(b)
Fig. 13. Tattoo identiﬁcation performance (in terms of CMC) by the pro-
posed approach and the state-of-the-art methods (TattooID [1], SSDH-
VGG16 [41], Faster R-CNN + SSDH, and OIM-ResNet50 [71]) on the
WebTattoo test dataset: (a) without 300K background tattoo images
in the gallery set, and (b) with 300K background tattoo images in the
gallery set; for TattooID, we report its tattoo search performance using
an extended gallery set with only 100K background images because of
its long running time.
challenging than tattoo photo-to-photo matching. Different
from the observations in image-based tattoo search, SSDH-
VGG16 performs better than TattooID in sketch-based tattoo
search. The main reasons are two-fold: (i) TattooID detects
much less SIFT keypoints from the tattoo sketches drawn
on the papers than from the tattoo photos; (ii) the learning
based methods, such as SSDH-VGG16, are able to leverage
the tattoo sketch-photo pairs to learn a feature represen-
tation that mitigates the modality gap between the tattoo
sketches and photos. The methods that compute features
from the detected tattoos (e.g., the proposed approach, OIM-
ResNet50, and Faster R-CNN + SSDH) perform better than
the methods that directly extract features from the holistic
tattoo images. The proposed approach performs consis-
tently better than the state-of-the-art methods in sketch
based tattoo search. The results suggest that proposed joint
tattoo detection and CRL approach has good generalization
ability into the sketch-based tattoo search scenario.
Fig. 10 (b) shows examples of sketch-based tattoo search
results by our approach. Beneﬁted from the joint tattoo de-
tection in the proposed approach, our feature representation
can reduce the inﬂuence of the background regions of the
tattoos in the gallery set, and thus is able to match a tattoo
sketch to its mated tattoo image at a low rank.
4.6 Tattoo Identiﬁcation
Automatic tattoo identiﬁcation techniques are usually
utilized to generate a candidate suspect list, which is used
for human or forensic analysis. While high rank-1 accu-
racy is ideal, success in these forensic recognition scenarios
is generally measured by the accuracies from rank-1 to
rank-100 [77]. Therefore, a number of the published tattoo
identiﬁcation methods reported their performance in terms
40 60 80 100Recall (in %)
01020304050607080Precision (in %)
Three losses (Proposed)Cross-entropy + dispersity lossesCross-entropy loss
Fig. 14. Ablation studies involving three loss functions (cross-entropy,
dispersity, and polarization) in CRL for tattoo image search on the
WebTattoo test dataset without distracter images in the gallery set.
of CMC curves covering rank-1 to rank-100 [1], [4], [8], [10],
[20]. We report the CMC curves of the proposed approach
and the state-of-the-art methods on the WebTattoo test
dataset in Fig. 13. Again, the results show that the methods
that compute features from the detected tattoos ( e.g., the
proposed approach, OIM-ResNet50, and Faster R-CNN +
SSDH) perform better than the methods that directly extract
features from the holistic tattoo images ( e.g., SSDH and Tat-
tooID). However, the proposed approach, which leverages
multi-task learning to perform joint tattoo detection and
CFL, achieves the best accuracy. When the300K background
images are used to populate the gallery set, all the ap-
proaches are observed to have decreased identiﬁcation ac-
curacies, e.g., about 30% degradation at rank-1 identiﬁcation
accuracy. The proposed approach still performs better than
the baselines in such a challenging scenario. These results
indicate that the proposed approach has good generaliza-
tion ability into the scenario of tattoo identiﬁcation with a
large gallery set.
On the public Tatt-C identiﬁcation dataset, the proposed
approach achieves 99.2% rank-1 identiﬁcation accuracy.
The MorphoTrak and the Purdue teams reported 99.4%
and 98.7% rank-1 identiﬁcation accuracies on the Tatt-C
identiﬁcation dataset. While the results by MorphoTrak are
slightly better than ours, they used four folds of data of Tatt-
C for training, and the ﬁfth-fold data for testing. By contrast,
our approach is trained on the WebTattoo training dataset,
which is different from the tattoo images in Tatt-C.
In addition to the above evaluations, we also evaluate
the generalization ability of the proposed approach in other
instance-level retrieval tasks, such as on Paris [78] and
Oxford [79], which contain challenging viewpoint and scale
variations. Following the same testing protocol as the state-
of-the-art method [41], our approach achieves 83.24% and
53.04% mAP on Paris and Oxford, respectively, which are
comparable to the performance of state-of-the-art method
[41] (83.87% and 63.79% mAP on Paris and Oxford, respec-
tively). These results show that the proposed approach has
good generalization ability to new application scenarios.
4.7 Ablation Study
We provide ablation studies of our approach in terms of
three loss functions, i.e., (i) basic cross-entropy loss, (ii)
cross-entropy loss and dispersity loss, and (ii) all three losses
together. The precision-recall curves calculated using the
top-100 retrieval results of the three experiments are shown
in Fig. 14. We can see that using dispersity loss together
13
with the cross-entropy loss does not improve the rank-
1 tattoo search accuracy compared to using cross-entropy
loss alone, but it does improve the overall performance
beyond rank-1. Jointly using all three losses leads to the
best performance, particularly for the rank-1 tattoo search
accuracy; this is important for practical applications. The
reason why jointly using all three losses works better for
compact feature learning is that while cross-entropy loss is
helpful for generating real-valued codes that are discrim-
inative between individual tattoo classes, dispersity and
polarization losses assure the real-valued codes are near-
binary and evenly distributed in the code space (see our
explanations in Sect. 3.2 and Fig. 4).
4.8 Computational Cost
We summarize the computational cost of the proposed ap-
proach and several state-of-the-art methods. Our approach
takes about 0.2 sec. in total to perform joint detection and
CRL on a Titan X GPU. After obtaining the compact feature
representation (256-bit), the average time of computing the
Hamming distance of two 256-D binary codes is 0.06ms on
an Intel i7 3.6GHz CPU without using bitwise operation
based optimizations, which is 5 times faster than computing
the cosine distance of two 256-D real-valued codes ( 0.3ms
on average). Such a difference in computational cost mat-
ters particularly for scenarios of tattoo search from huge
volumes of surveillance video frames or handling multi-
ple parallel searching requests. At the same time, compar-
isons with the state-of-the-art methods based on real-valued
codes, e.g., [71], show that our compact binary codes of the
same length can achieve better accuracy. Only a few of the
published tattoo identiﬁcation and tattoo retrieval methods
have reported their computational costs. For example, [16]
reported an average of 24 sec. in comparing one query tattoo
against 10K gallery tattoos after obtaining the SIFT features
on an Intel Core2 2.66 GHz CPU, which is much slower
than the proposed compact representation. We also proﬁled
the tattoo detection time by a Faster R-CNN detector in
[22] (using the same input image size as our approach)
and the feature extraction time by a VGG-16 CNN network
used in [41]. Tattoo detection and feature extraction per
tattoo detection take 0.2 sec. and 0.04 sec., respectively. This
indicates that the proposed approach is more efﬁcient in real
application scenarios, in which there are usually more than
one tattoo detections per image.
5 C ONCLUSIONS
This paper presents a joint detection and compact feature
learning approach for tattoo image search at scale. While
existing tattoo search methods mainly focus on matching
cropped tattoos, the proposed approach models tattoo de-
tection and compact representation learning in a single
convolutional neural network via multi-task learning. The
WebTattoo dataset consisting of 300K tattoo images was
compiled from the public-domain tattoo datasets and im-
ages from the Internet. In addition, 300 tattoo sketches
were created for sketch-based tattoo search to replicate
the scenario where the surveillance image of the tattoo is
not available. These datasets help evaluate the proposed
approach for tattoo image search at scale and in operational
scenarios. Our approach performs well on a number of
tasks including tattoo detection, tattoo search at scale, and
sketch-based tattoo search. The proposed data augmenta-
tion method is able to replicate various tattoo appearance
variations, and thus is helpful to improve the robustness of
the tattoo detector in unconstrained scenarios. Experimental
results with cross-database testing protocols show that the
proposed approach generalizes well to the unseen scenarios.
ACKNOWLEDGMENTS
This research was supported in part by the Natural Sci-
ence Foundation of China (grants 61732004, 61390511,
and 61672496), External Cooperation Program of Chinese
Academy of Sciences (CAS) (grant GJHZ1843), and Youth
Innovation Promotion Association CAS (2018135). The pre-
liminary work appeared in the Proceedings of the 6th Inter-
national Conference on Biometrics (ICB), 2013 [10]. Anil K.
Jain is the corresponding author.
REFERENCES
[1] J. Lee, R. Jin, A. K. Jain, and W. Tong, “Image retrieval in forensics:
Tattoo image database application,” IEEE MultiMedia , vol. 19,
no. 1, pp. 40–49, Jan. 2012.
[2] FBI, “FBI next generation identiﬁcation (NGI) overview,” Sept.
2010.
[3] A. Bertillon, Signaletic Instructions Including the Theory and Practice
of Anthropometrical Identiﬁcation. The Werner Company, 1896.
[4] A. K. Jain, J. Lee, and R. Jin, “Tattoo-ID: Automatic tattoo image
retrieval for suspect and victim identiﬁcation,” inProc. IEEE PCM,
2007, pp. 256–265.
[5] M. L. Ngan, G. W. Quinn, and P . J. Grother, “Tattoo recognition
technology - challenge (Tatt-C): Outcomes and recommendations,”
NIST Interagency/Internal Report, Tech. Rep. 8078, 2015.
[6] R. McCabe, “Information technology: American national standard
for information systems: Data format for the interchange of ﬁn-
gerprint, facial, & scar mark & tattoo (SMT) information,” NIST
Special Publication, Tech. Rep. 500-245, 2000.
[7] J. Lee, R. Jin, and A. K. Jain, “Rank-based distance metric learning:
An application to image retrieval,” in Proc. IEEE CVPR, Jun. 2008,
pp. 1–8.
[8] S. T. Acton and A. Rossi, “Matching and retrieval of tattoo images:
Active contour cbir and glocal image features,” in Proc. IEEE
SSIAI, Mar. 2008, pp. 21–24.
[9] D. Manger, “Large-scale tattoo image retrieval,” in Proc. CRV, May
2012, pp. 454–459.
[10] H. Han and A. K. Jain, “Tattoo based identiﬁcation: Sketch to
image matching,” in Proc. ICB, Jun. 2013, pp. 1–8.
[11] Q. Xu, S. Ghosh, X. Xu, Y. Huang, and A. W. K. Kong, “Tattoo
detection based on CNN and remarks on the NIST database,” in
Proc. ICB, Jun. 2016, pp. 1–7.
[12] Y. Cao, C. Wang, L. Zhang, and L. Zhang, “Edgel index for large-
scale sketch-based image search,” in Proc. CVPR , Jun. 2011, pp.
761–768.
[13] Y. Cao, H. Wang, C. Wang, Z. Li, L. Zhang, and L. Zhang,
“MindFinder: Interactive sketch-based image search on millions
of images,” in Proc. ACM MM, Oct. 2010, pp. 1605–1608.
[14] X. Wang and X. Tang, “Face photo-sketch synthesis and recogni-
tion,” IEEE Trans. Pattern Anal. Mach. Intell. , vol. 31, no. 11, pp.
1955–1967, Nov. 2009.
[15] H. Han, B. F. Klare, K. Bonnen, and A. K. Jain, “Matching compos-
ite sketches to face photos: A component-based approach,” IEEE
Trans. Inf. Forensics Security, vol. 8, no. 1, pp. 191–204, Jan. 2013.
[16] A. K. Jain, J. E. Lee, R. Jin, and N. Gregg, “Content-based image
retrieval: An application to tattoo images,” inProc. IEEE ICIP, Nov.
2009, pp. 2745–2748.
[17] F. Li, W. Tong, R. Jin, A. K. Jain, and J. Lee, “An efﬁcient key point
quantization algorithm for large scale image retrieval,” in Proc.
ACM Workshop LS-MMRM, Oct. 2009, pp. 89–96.
14
[18] B. Heﬂin, W. Scheirer, and T. E. Boult, “Detecting and classifying
scars, marks, and tattoos found in the wild,” in Proc. BTAS, Sept.
2012, pp. 31–38.
[19] M. J. Wilber, E. Rudd, B. Heﬂin, Y. Lui, and T. E. Boult, “Exemplar
codes for facial attributes and tattoo recognition,” in Proc. WACV,
Mar. 2014, pp. 205–212.
[20] X. Xu and A. Kong, “A geometric-based tattoo retrieval system,”
in Proc. ICPR, Dec. 2016, pp. 3019–3024.
[21] J. Kim, H. Li, J. Yue, J. Ribera, E. J. Delp, and L. Huffman,
“Automatic and manual tattoo localization,” in Proc. HST , May
2016, pp. 1–6.
[22] Z. Sun, J. Baumes, P . Tunison, M. Turek, and A. Hoogs, “Tattoo
detection and localization using region-based deep learning,” in
Proc. ICPR, Dec. 2016, pp. 3055–3060.
[23] X. Di and V . M. Patel, “Deep tattoo recognition,” in Proc. CVPR
Workshop, Jun. 2016, pp. 119–126.
[24] ——, Deep Learning for Tattoo Recognition, B. Bhanu and A. Kumar,
Eds. Springer, 2017.
[25] J. Sivic and A. Zisserman, “Video Google: A text retrieval approach
to object matching in videos,” inProc. ICCV, Oct. 2003, pp. II: 1470–
1477.
[26] D. G. Lowe, “Distinctive image features from scale-invariant key-
points,” Int. J. Comput. Vision, vol. 60, no. 2, pp. 91–110, Nov. 2004.
[27] Y. LeCun, Y. Bengio, and G. Hinton, “Learning representations by
back-propagating errors,” Nature, vol. 521, no. 14539, pp. 436–444,
May. 2015.
[28] L. Zheng, Y. Yang, and Q. Tian, “SIFT meets CNN: A decade
survey of instance retrieval,” IEEE Trans. Pattern Anal. Mach. Intell.,
May 2017.
[29] A. Krizhevsky, I. Sutskever, and G. E. Hinton, “ImageNet classi-
ﬁcation with deep convolutional neural networks,” in Proc. NIPS,
2012, pp. 1097–1105.
[30] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards
real-time object detection with region proposal networks,” IEEE
Trans. Pattern Anal. Mach. Intell., vol. 39, no. 6, pp. 1137–1149, Jun.
2017.
[31] W. Zhou, H. Li, J. Sun, and Q. Tian, “Collaborative index embed-
ding for image retrieval,” IEEE Trans. Pattern Anal. Mach. Intell. ,
vol. 40, no. 5, pp. 1154–1166, May 2018.
[32] L. Xie, Q. Tian, and B. Zhang, “Max-SIFT: Flipping invariant
descriptors for web logo search,” inProc. ICIP, Oct. 2014, pp. 5716–
5720.
[33] W. Zhou, Y. Lu, H. Li, and Q. Tian, “Scalar quantization for large
scale image search,” in Proc. ACM MM, Oct./Nov. 2012, pp. 169–
178.
[34] L. Xie, R. Hong, B. Zhang, and Q. Tian, “Image classiﬁcation and
retrieval are one,” in Proc. ACM ICMR, 2015, pp. 3–10.
[35] L. Chi, H. Zhang, and M. Chen, “End-to-end spatial transform face
detection and recognition,” ArXiv e-prints.
[36] S. Yang, P . Luo, C. C. Loy, and X. Tang, “Wider face: A face
detection benchmark,” in Pro. CVPR, 2016.
[37] D. Yi, Z. Lei, S. Liao, and S. Z. Li, “Learning face representation
from scratch,” ArXiv e-prints, 2014.
[38] M. Norouzi and D. J. Fleet, “Minimal loss hashing for compact
binary codes,” in Proc. ICML, Jun. 2011, pp. 353–360.
[39] V . E. Liong, J. Lu, G. Wang, P . Moulin, and J. Zhou, “Deep hashing
for compact binary codes learning,” in Proc. CVPR, Jun. 2015, pp.
2475–2483.
[40] H. Jain, J. Zepeda, P . Perez, and R. Gribonval, “SUBIC: A super-
vised, structured binary code for image search,” in Proc. ICCV ,
Oct. 2017.
[41] H. F. Yang, K. Lin, and C. S. Chen, “Supervised learning of
semantics-preserving hash via deep convolutional neural net-
works,” IEEE Trans. Pattern Anal. Mach. Intell. , vol. 40, no. 2, pp.
437–451, Feb. 2018.
[42] J. Lu, V . E. Liong, and J. Zhou, “Deep hashing for scalable image
search,” IEEE Trans. Image Process. , vol. 26, no. 5, pp. 2352–2367,
May 2017.
[43] H. Jegou, M. Douze, and C. Schmid, “Product quantization for
nearest neighbor search,” IEEE Trans. Pattern Anal. Mach. Intell. ,
vol. 33, no. 1, pp. 117–128, Jan. 2011.
[44] A. Babenko and V . Lempitsky, “Additive quantization for extreme
vector compression,” in Proc. CVPR, Jun. 2014, pp. 931–938.
[45] T. Zhang, C. Du, and J. Wang, “Composite quantization for ap-
proximate nearest neighbor search,” in Proc. ICML, Jun. 2014, pp.
II:838–846.
[46] X. Wang, T. Zhang, G. Qi, J. Tang, and J. Wang, “Supervised
quantization for similarity search,” in Proc. CVPR, Jun. 2016, pp.
2018–2026.
[47] T. Yu, Z. Wang, and J. Yuan, “Compressive quantization for fast
object instance search in videos,” in Proc. ICCV , Oct. 2017, pp.
726–735.
[48] P . Indyk and R. Motwani, “Approximate nearest neighbors: To-
wards removing the curse of dimensionality,” in Proc. STOC, May
1998, pp. 604–613.
[49] M. Datar, N. Immorlica, P . Indyk, and V . S. Mirrokni, “Locality-
sensitive hashing scheme based on p-stable distributions,” in Proc.
SCG, Jun. 2004, pp. 253–262.
[50] Y. Weiss, A. Torralba, and R. Fergus, “Spectral hashing,” in Proc.
NIPS, Dec. 2009, pp. 1753–1760.
[51] B. Kulis and K. Grauman, “Kernelized locality-sensitive hashing
for scalable image search,” in Proc. ICCV , Sept. 2009, pp. 2130–
2137.
[52] W. Liu, J. Wang, S. Kumar, and S.-F. Chang, “Hashing with
graphs,” in Proc. ICML, Jun. 2011, pp. 1–8.
[53] W. Kong and W. Li, “Isotropic hashing,” in Proc. NIPS, Dec. 2012,
pp. 1646–1654.
[54] Q. Jiang and W. Li, “Scalable graph hashing with feature transfor-
mation,” in Proc. IJCAI, Jul. 2015, pp. 2248–2254.
[55] B. Kulis, P . Jain, and K. Grauman, “Fast similarity search for
learned metrics,” IEEE Trans. Pattern Anal. Mach. Intell. , vol. 31,
no. 12, pp. 2143–2157, Dec. 2009.
[56] W. Liu, J. Wang, R. Ji, Y. G. Jiang, and S. F. Chang, “Supervised
hashing with kernels,” in Proc. CVPR, Jun. 2012, pp. 2074–2081.
[57] Y. Lin, R. Jin, D. Cai, S. Yan, and X. Li, “Compressed hashing,” in
Proc. CVPR, Jun. 2013, pp. 446–451.
[58] F. Shen, C. Shen, W. Liu, and H. T. Shen, “Supervised discrete
hashing,” in Proc. CVPR, Jun. 2015, pp. 37–45.
[59] J. Gui, T. Liu, Z. Sun, D. Tao, and T. Tan, “Fast supervised discrete
hashing,” IEEE Trans. Pattern Anal. Mach. Intell., Mar. 2017.
[60] R. Salakhutdinov and G. Hinton, “Learning a nonlinear embed-
ding by preserving class neighbourhood structure,” inProc. ICAIS,
2007, pp. 412–419.
[61] B. Kulis and T. Darrell, “Learning to hash with binary reconstruc-
tive embeddings,” in Proc. NIPS, Dec. 2009, pp. 1042–1050.
[62] W.-J. Li, S. Wang, and W.-C. Kang, “Feature learning based deep
supervised hashing with pairwise labels,” in Proc. IJCAI, Jul. 2016,
pp. 1711–1717.
[63] J. Wang, W. Liu, S. Kumar, and S. F. Chang, “Learning to hash
for indexing big data - a survey,” Proceedings of the IEEE , vol. 104,
no. 1, pp. 34–57, Jan. 2016.
[64] L. Chi and X. Zhu, “Hashing techniques: A survey and taxonomy,”
ACM Comput. Surv., vol. 50, no. 1, pp. 11:1–36, Apr. 2017.
[65] R. Girshick, “Fast R-CNN,” in Proc. ICCV, Dec. 2015.
[66] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning
representations by back-propagating errors,” Nature, vol. 323, no.
6088, pp. 533–536, Oct. 1986.
[67] P . de Boer, D. P . Kroese, S. Mannor, and R. Y. Rubinstein, “A tu-
torial on the cross-entropy method,” Annals of Operations Research,
vol. 134, no. 1, pp. 19–67, Feb. 2005.
[68] K. Simonyan and A. Zisserman, “Very deep convolutional net-
works for large-scale image recognition,” ArXiv e-prints , Sept.
2014.
[69] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
image recognition,” in Proc. CVPR, Jun. 2016, pp. 770–778.
[70] A. Salvador, X. Gir ´o-i-Nieto, F. Marqu ´es, and S. Satoh, “Faster r-
cnn features for instance search,” in Proc. CVPRW, Jun. 2016, pp.
394–401.
[71] T. Xiao, S. Li, B. Wang, L. Lin, and X. Wang, “Joint detection and
identiﬁcation feature learning for person search,” in Proc. CVPR,
Jul. 2017, pp. 3376–3385.
[72] T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays,
P . Perona, D. Ramanan, C. L. Zitnick, and P . Doll ´ar, “Microsoft
COCO: Common objects in context,” ArXiv e-prints, May 2014.
[73] O. Russakovsky, J. Deng, H. Su, J. Krause, S. Satheesh, S. Ma,
Z. Huang, A. Karpathy, A. Khosla, M. Bernstein, A. C. Berg, and
F.-F. Li, “ImageNet large scale visual recognition challenge,” Int. J.
Comput. Vision, vol. 115, no. 3, pp. 211–252, Dec. 2015.
[74] T. Hrka ´c, K. Brki ´c, and Z. Kalafati ´c, “Tattoo detection for soft bio-
metric de-identiﬁcation based on convolutional neural networks,”
in Proc. The 1st OAGM-ARW Joint Workshop - Vision Meets Robotics,
May 2016, pp. 131–138.
15
[75] A. Shrivastava, A. Gupta, and R. Girshick, “Training region-based
object detectors with online hard example mining,” ArXiv e-prints,
Apr. 2016.
[76] A. K. Jain, M. N. Murty, and P . J. Flynn, “Data clustering: A
review,” ACM Comput. Surv., vol. 31, no. 3, pp. 264–323, Sept. 1999.
[77] A. K. Jain, B. Klare, and U. Park, “Face matching and retrieval in
forensics applications,” IEEE MultiMedia, vol. 19, no. 1, pp. 20–28,
Jan. 2012.
[78] J. Philbin, O. Chum, M. Isard, J. Sivic, and A. Zisserman, “Lost in
quantization: Improving particular object retrieval in large scale
image databases,” in Proc. IEEE CVPR, Jun. 2008, pp. 1–8.
[79] ——, “Object retrieval with large vocabularies and fast spatial
matching,” in Proc. IEEE CVPR, Jun. 2007, pp. 1–8.
Hu Han is an Associate Professor of the Insti-
tute of Computing Technology (ICT), Chinese
Academy of Sciences (CAS). He received the
B.S. degree from Shandong University, and the
Ph.D. degree from ICT, CAS, in 2005 and 2011,
respectively, both in computer science. Before
joining the faculty at ICT, CAS in 2015, he has
been a Research Associate at PRIP lab in the
Department of Computer Science and Engineer-
ing at Michigan State University, and a Visiting
Researcher at Google in Mountain View. His re-
search interests include computer vision, pattern recognition, and image
processing, with applications to biometrics, forensics, law enforcement,
and security systems. He is a member of the IEEE.
Jie Li received the B.S. degree from Qingdao
University in 2017, and he is working toward
the M.S. degree in ICT, CAS and the University
of Chinese Academy of Sciences. His research
interests include computer vision, pattern recog-
nition, and image processing, with applications
to biometrics.
Anil K. Jain is a University Distinguished Pro-
fessor in the Department of Computer Science
and Engineering at Michigan State University.
His research interests include pattern recogni-
tion and biometric authentication. He served as
the editor-in-chief of the IEEE T RANSACTIONS
ON PATTERN ANALYSIS AND MACHINE INTELLI -
GENCE (1991-1994). He served as a member of
the United States Defense Science Board and
The National Academies committees on Whither
Biometrics and Improvised Explosive Devices.
He has received Fulbright, Guggenheim, Alexander von Humboldt, and
IAPR King Sun Fu awards. He is a member of the National Academy
of Engineering and foreign fellow of the Indian National Academy of
Engineering. He is a Fellow of the AAAS, ACM, IAPR, SPIE, and IEEE.
Shiguang Shan is a Professor of ICT, CAS, and
the Deputy Director with the Key Laboratory of
Intelligent Information Processing, CAS. His re-
search interests cover computer vision, pattern
recognition, and machine learning. He has au-
thored over 200 papers in refereed journals and
proceedings in the areas of computer vision and
pattern recognition. He was a recipient of the
China’s State Natural Science Award in 2015,
and the China’s State S&T Progress Award in
2005 for his research work. He has served as
the Area Chair for many international conferences, including ICCV’11,
ICPR’12, ACCV’12, FG’13, ICPR’14, and ACCV’16. He is an Associate
Editor of several journals, including the IEEE T RANSACTIONS ON IM-
AGE PROCESSING , the Computer Vision and Image Understanding, the
Neurocomputing, and the Pattern Recognition Letters. He is a Senior
Member of IEEE.
Xilin Chen is a Professor of ICT, CAS. He
has authored one book and over 200 papers
in refereed journals and proceedings in the ar-
eas of computer vision, pattern recognition, im-
age processing, and multimodal interfaces. He
served as an Organizing Committee/Program
Committee member for over 70 conferences. He
was a recipient of several awards, including the
China’s State Natural Science Award in 2015,
the China’s State S&T Progress Award in 2000,
2003, 2005, and 2012 for his research work.
He is currently an Associate Editor of the IEEE T RANSACTIONS ON
MULTIMEDIA , a Leading Editor of the Journal of Computer Science and
Technology, and an Associate Editor-in-Chief of the Chinese Journal of
Computers. He is a Fellow of the China Computer Federation (CCF),
IAPR, and IEEE.