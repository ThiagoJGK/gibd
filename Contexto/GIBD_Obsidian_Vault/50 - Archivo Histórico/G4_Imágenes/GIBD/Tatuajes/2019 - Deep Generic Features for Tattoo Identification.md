---
aliases: [2019 - Deep Generic Features for Tattoo Identification]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2024-09-10
origen_zip: GIBD-20260521T205218Z-3-005.zip
ruta_interna: "GIBD/Tatuajes/CoNaIISI 2024/2019 - Deep Generic Features for Tattoo Identification.pdf"
tamanio_bytes: 618291
---

# 2019 - Deep Generic Features for Tattoo Identification

Ruta interna: `GIBD/Tatuajes/CoNaIISI 2024/2019 - Deep Generic Features for Tattoo Identification.pdf`

---

See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/336815752
Deep Generic Features for Tattoo Identiﬁcation
Chapter · October 2019
DOI: 10.1007/978-3-030-33904-3_25
CITATIONS
4
READS
362
3 authors:
Miguel Nicolás-Díaz
Centro de Aplicaciones de Tecnologias de Avanzada
6 PUBLICATIONS   143 CITATIONS   
SEE PROFILE
Annette Morales-González
Centro de Aplicaciones de Tecnologias de Avanzada
47 PUBLICATIONS   294 CITATIONS   
SEE PROFILE
Heydi Mendez Vazquez
Centro de Aplicaciones de Tecnologias de Avanzada
95 PUBLICATIONS   798 CITATIONS   
SEE PROFILE
All content following this page was uploaded by Annette Morales-González on 29 October 2019.
The user has requested enhancement of the downloaded file.
Deep Generic Features for Tattoo Identiﬁcation
Miguel Nicol´ as-D´ ıaz, Annette Morales-Gonz´ alez, and Heydi M´ endez-V´ azquez
Advanced Technologies Application Center (CENATAV)
7A #21406 b/214 and 216, Siboney, Playa, P.C.12200, Havana, Cuba
{mnicolas,amorales,hmendez}@cenatav.co.cu
Abstract. Recently, interest has grown in using tattoos as a biomet-
ric feature for person identiﬁcation. Previous works used handcrafted
features for the tattoo identiﬁcation task, such as SIFT. However, deep
learning methods have shown better results than this kind of methods
in many computer vision tasks. Taking into account that there are lit-
tle research on tattoo identiﬁcation using deep learning, we asses sev-
eral publicly available CNNs models, pre-trained on large generic image
databases, for the task of tattoo identiﬁcation. We believe that, since
tattoos mostly depict objects of the real world, their semantic and visual
features might be related to those learned from a generic image database
with real objects. Our experiments show that these models can out-
perform previous approaches without even ﬁne-tuning them for tattoo
identiﬁcation. This allows developing tattoo identiﬁcation applications
with minimum implementation cost. Besides, due to the diﬃcult access
to public tattoo databases, we created two tattoo datasets and put one
of them in public domain.
Keywords: Tattoo identiﬁcation· Deep learning· pre-trained models·
tattoo datasets.
1 Introduction
In recent years, several forensic techniques have been developed in order to iden-
tify victims and criminals in forensic scenarios [7]. These systems are mainly
based on biometric traits such as the face, ﬁngerprints and iris. However, there
are many situations in which primary biometric traits like these are not available,
and therefore it is necessary to resort to other types of information [11]. The so
called “soft biometric traits” are physiological or behavioral characteristics that
provide some identifying information about an individual [2], but lack distinc-
tiveness and permanence to suﬃciently diﬀerentiate any two individuals [11].
Eye color, gender, ethnicity, skin color, height, weight, hair color, scars, birth-
marks and tattoos are examples of soft biometric traits. Several techniques have
been proposed to identify or verify the identity of a person, automatically, based
on soft biometric traits [2, 8]. In particular, person identiﬁcation and retrieval
systems based on tattoos have gained much interest in recent years [4, 16]. This
is due to several reasons, including the fact that the tendency of people to have
tattoos has increased [11]. Furthermore, tattoos provide additional information
2 M. Nicol´ as-D´ ıaz et al.
about the person: aﬃliation to groups or gangs, religious beliefs, years in prison,
etc. These have been used to assist law enforcement authorities in investigations
that lead to the identiﬁcation of oﬀenders and victims of natural disasters and
accidents [11].
This paper focuses on the identiﬁcation of individuals based on tattoos. Most
of existing works in this topic [9, 11] propose to use handcrafted features to
describe the images and then evaluate the similarity between two images by
comparing their features. However, in recent years deep learning methods have
shown better results than this kind of features in similar computer vision tasks
[3]. That is the reason why some works [3, 4] have proposed the use of deep
neural networks for tattoo identiﬁcation.
Due to the great variability that usually exists between tattoos of diﬀerent
individuals, in [3], it is explored the idea of ﬁne-tuning deep networks trained
on large generic databases such as ImageNet [17] for the task of tattoo image
classiﬁcation considering two classes: Tattoo and Not-Tattoo. This database con-
tains many varied images of a large number of object types, something similar
to what happens with tattoos. Therefore, it has a certain logic to think that a
neural network trained to discern the images that belong to the same class and
those that do not, can also do it for tattoo images. This way, it is not necessary
to train a network from scratch or count on the large volume of data that this
training requires. Nevertheless, for the task of tattoo matching, the authors of
[3] used a Siamese network, which has the disadvantage that it is necessary to
perform the inference, every time a comparison is made between two images.
This incurs in a high computational cost when used in an identiﬁcation scenario
since it is necessary to compare the image to be identiﬁed with all the images
of a database. In contrast to that approach, if a network is used to extract fea-
tures from the database images, they can be stored and then used for future
comparisons. In this way, the network is executed only once for each requested
identiﬁcation. In this paper we adopt this strategy as a way to perform tattoo
identiﬁcation more eﬃciently.
On the other hand, recent papers [1] have shown that top layers of a large
convolutional neural network (CNN) provide high-level descriptors of the visual
content of the image. They proved that these descriptors can be used for diﬀerent
tasks other than that for which the network was trained. This allows using them
eﬀectively without the need of having large databases to train the network,
which is the case for tattoos, where there are few public databases. Based on the
previous ideas, the main contribution of our work is a study to assess the use of
some deep neural networks trained on generic databases such as ImageNet, for
tattoo identiﬁcation. We extract the features from intermediate layers of these
networks and used them as descriptors of the tattoo images. The diﬀerence with
previous works [3, 4] is that we show that it is possible to achieve competitive
results without training or even ﬁne-tuning the networks.
The remainder of this paper is structured as follows. We brieﬂy review related
literature in Section 2. In Section 3, the details of the proposed deep networks for
Deep Generic Features for Tattoo Identiﬁcation 3
tattoo identiﬁcation are provided. We show some experimental results in Section
4. Finally, we conclude this work in Section 5.
2 Related works
The early practice of tattoo image retrieval relied on keywords or metadata
based matching [4]. However, a keyword-based tattoo image retrieval has several
limitations in practice : (i) Labels are insuﬃcient for describing all visual infor-
mation of a tattoo; (ii) multiple keywords may be needed to adequately describe
a tattoo image; (iii) human annotation is subjective and diﬀerent subjects can
give dramatically diﬀerent labels to the same tattoo image [4].
Due to these problems, interest has grown in developing content-based image
retrieval techniques (CBIR) to improve the eﬃciency and accuracy of the tattoo
search [9, 14]. CBIR aims to extract features such as edges, color and texture,
that can reﬂect the content of an image, and use them to identify images with
high visual similarity [4]. The scale-invariant feature transform (SIFT) [13] and
its variants have been the most used among this kind of methods for tattoo
identiﬁcation [9, 14].
Recently, most researches are focusing on deep learning methods due to its
success in many computer vision tasks [3]. In particular, AlexNet [10], which won
the ImageNet challenge of 2012, has been successfully used for tattoo vs. non-
tattoo classiﬁcation in [3, 22]. Other works [21] focus on tattoo localization using
Faster R-CNN. However, little research exists on tattoo image identiﬁcation. To
the best of our knowledge, only [3] and [4] studied the identiﬁcation of tattoos
using deep learning. In [3] a Siamese network pre-trained on ImageNet and ﬁne-
tuned for tattoo identiﬁcation is used. As mentioned in the Introduction, the
use of Siamese networks for matching is less eﬃcient because it is required to
perform the network inference for every comparison. This is not suitable for
an identiﬁcation scenario where it is necessary to match a query image with
thousands of images in an operational database.
In [4] a network based on a Faster R-CNN was used to learn tattoo detection
and a compact representation of the tattoo in the same network. The features
that return this network are binarized in order to make search eﬃcient. This
work obtains comparable results with other state-of-the-art methods and gen-
eralizes well to other retrieval tasks. However, this network was trained with
hundred of thousands of images, which are not available for everyone neither the
computational resources to do it. Moreover, a pre-trained model of this network
is not publicly available.
3 Generic neural networks for tattoo identiﬁcation
The proposal of this paper is to use the features from intermediate layers of
neural networks, trained on large generic databases, to describe the content of
a tattoo image. These features should be matched with the features of a tattoo
gallery in order to identify the tattoo.
4 M. Nicol´ as-D´ ıaz et al.
In order to evaluate the proposal, we selected some neural networks that have
obtained good results in the ImageNet classiﬁcation challenge. It is worth noting
that none of these networks was ﬁne-tuned with images of tattoos as was done
in [3], instead we used their publicly available models trained on ImageNet.
We believe that, since tattoos mostly depict objects of the real world, their
semantic and visual features might be related to those learned from a generic
image database with real objects.
– MobileNetV1 [5]: network designed to be used in mobile and embedded de-
vices thanks to its low complexity. It is trained on ImageNet, where it ob-
tained 70.6 % classiﬁcation accuracy.
– MobileNetV2 [18]: improved version of MobileNetV1. It obtains 74.7 % of
accuracy, improving the previous version at a similar processing cost.
– Inception21k [6]: it is trained on ImageNet, but with 21841 classes, unlike
the 1000 that are generally used. It obtains 68.3% accuracy.
– Resnet50 CVGJ [19]: variant with batch normalization [6] of the ResNet50
network trained on ImageNet.
– VGG CVGJ [19]: variant with batch normalization [6] of the VGG19 [20]
network trained on ImageNet.
We also evaluate some networks that have been proposed for image retrieval.
These are designed to return a feature vector as a descriptor of the visual content
of the image instead of a classiﬁcation output.
– DeepBit [12]: network trained in an unsupervised manner to build high-level
compressed features. During the training, restrictions were used so that the
features met three requirements: invariance to the rotation of the image,
high entropy of the features and high standard deviation.
– SSDH [23]: network that constructs hash functions as a latent layer in a deep
network. It was designed so that classiﬁcation and retrieval were uniﬁed in
a single learning model.
Both networks were designed so that their features were binarized allowing
a more eﬃcient matching using the Hamming distance. Both deﬁne a function
to do that, however, we do not binarize the features because we obtained better
results with the real values.
Table 1 shows the layer used for each network, as well as the vector dimension
of output features. For DeepBit and SSDH we used the last layer, while for the
other networks we report the layers that achieve better results.
All networks have a ﬁxed input of 224 x 224 pixels, except SSDH that has
an input of 227 x 227. The euclidean distance was used to match the features,
except for DeepBit, for which was used the cosine distance that gaves much
better results.
4 Experimental evaluation
In this section we aim at evaluating and comparing the selected networks in the
task of tattoo identiﬁcation. Diﬀerent tattoo databases have been used in the
Deep Generic Features for Tattoo Identiﬁcation 5
T able 1. Deep networks characteristics.
Network Layer used Output size
MobileNetV1 pool6 1024
MobileNetV2 pool6 1280
Inception21k global pool 1024
Resnet50 CVGJ global pool 2048
VGG CVGJ fc7 4096
DeepBit - 32
SSDH - 32
literature [15, 16], but most of them are not public or their access is diﬃcult. In
particular, most of the works have used the Tatt-C database [16] for experiment-
ing, training and comparing with other methods, but it is not public any more
due to legal issues. The authors of [4] created a large dataset named WebTattoo
combining other tattoo datasets that they have access. However, this dataset
is not public yet; should be released soon. Therefore, in order to validate our
proposal, we have created our own datasets.
The conducted experiments were performed in a PC with a CPU Intel Core
i7-4470 with 8 GB of RAM.
4.1 Proposed databases and evaluation protocol
Two databases of tattoo images were created in order to evaluate the proposal.
The ﬁrst one (BIVTatt)1 was collected by the authors of this paper and contains
210 images belonging to 159 individuals (some individuals have only one image).
The second (PinTatt) is composed of 454 images downloaded from Pinterest be-
longing to 160 individuals. Images from BIVTatt have higher resolution than
PinTatt images and their content is sharper. All the images are cropped around
the tattoo so there is not much background information. Fig. 1 shows some
sample images from both databases. For each image, 20 new images were gener-
ated applying transformations with two diﬀerent intensities of illumination, two
diﬀusions, four aﬃne transformations, four aspect ratio transformations, four
diﬀerent rotations and four color changes, as described in [11]. This way, the
databases were increased obtaining a total of 4410 images for BIVTatt and 9534
for PinTatt. Fig. 2 shows examples of transformations of an image from the
BIVTatt dataset.
For the aim of identiﬁcation experiments on each database, the probe set was
conformed by the transformed images, while the gallery, was composed of the
original images. For every test image, the original image that originate it (by
some transformation) was excluded from the comparison. Thus, we simulate a
real scenario of forensic identiﬁcation were diﬀerent images from the same tattoo
can be available, but not exactly the same image. The ﬁnal BIVTatt probe set
1 The BIVTatt dataset is available at https://github.com/mnicolas94/BIVTatt-
Dataset.
6 M. Nicol´ as-D´ ıaz et al.
Fig. 1. Sample images of both datasets: (a) BIVTatt, (b) PinTatt.
Fig. 2. Examples of image transformations in the BIVTatt dataset: (a) original; (b)
aﬃne transformation; (c) aspect ratio; (d) blurring; (e) color change; (f) illumination;
(g) rotation.
consists of 1540 images and the gallery of 209 images. In the case of PinTatt it
has 9080 and 453, respectively.
To evaluate the performance of the compared methods, we employed the
cumulative match characteristic (CMC) curve. Each point of the CMC curve is
the fraction of images of the probe set that were correctly matched with any of
its pairs in the gallery, in a given range.
4.2 Experimental results
Besides the experiments with the networks analyzed in Section 3, all the tests
were also carried out using SIFT in order to compare the proposal of this work
with a general approach used in previous works. The matching method used for
SIFT was a knn-based matcher implemented in the Fast Library for Approximate
Nearest Neighbors (FLANN) of the OpenCV library with k=2. The Fig. 3 and
the Fig. 4 show the CMC curves for BIVTatt and PinTatt databases respectively.
The Table 2 and the Table 3 show the identiﬁcation recognition rates at diﬀerent
rank values for BIVTatt and PinTatt databases respectively.
As can be seen, the best general performance was achieved by MobileNetV2
on both datasets. The two image retrieval networks, DeepBit and SSDH, ob-
tained poor results. The image classiﬁcation networks obtain good results in
general. All of them outperformed SIFT, except for Rank-1 on BIVTatt dataset.
In the case of Resnet50 CVGJ, it only exceeds SIFT starting from Rank-20 on
this dataset.
Deep Generic Features for Tattoo Identiﬁcation 7
0.000% 
10.000% 
20.000% 
30.000% 
40.000% 
50.000% 
60.000% 
70.000% 
80.000% 
90.000% 
100.000% 
0 5 10 15 20 25 30 35 40 45 50 
Cumulative Matching Characteristic 
SIFT MobileNetV1 MobileNetV2 Inception21k 
Resnet50_CVGJ VGG_CVGJ DeepBit SSDH 
Fig. 3. Curve CMC on the BIVTatt dataset.
0.000% 
10.000% 
20.000% 
30.000% 
40.000% 
50.000% 
60.000% 
70.000% 
80.000% 
90.000% 
100.000% 
0 5 10 15 20 25 30 35 40 45 50 
Cumulative Matching Characteristic 
SIFT MobileNetV1 MobileNetV2 Inception21k 
Resnet50_CVGJ VGG_CVGJ DeepBit SSDH 
Fig. 4. Curve CMC on the PinTatt dataset.
8 M. Nicol´ as-D´ ıaz et al.
T able 2. BIVTatt dataset identiﬁcation results.
Algorithm/network Rank1 (%) Rank5 (%) Rank10 (%) Rank20 (%) Rank50 (%)
SIFT 68.571 77.727 81.429 85.519 92.078
MobileNetV1 67.857 83.766 88.442 92.532 96.883
MobileNetV2 70.909 85.519 89.935 93.636 96.818
Inception21k 66.688 82.792 87.468 91.169 94.740
Resnet50 CVGJ 54.221 73.701 80.584 86.169 93.182
VGG CVGJ 62.792 79.156 84.805 89.805 95.195
DeepBit 27.338 50.195 62.468 74.351 86.688
SSDH 9.221 24.351 34.221 45.584 67.078
T able 3. PinTatt dataset identiﬁcation results.
Algorithm/network Rank1 (%) Rank5 (%) Rank10 (%) Rank20 (%) Rank50 (%)
SIFT 25.936 34.967 40.485 46.806 59.515
MobileNetV1 55.033 68.689 75.154 81.432 89.405
MobileNetV2 54.350 71.244 76.806 83.095 90.694
Inception21k 44.901 62.952 70.540 78.304 88.062
Resnet50 CVGJ 31.278 50.000 59.152 69.053 80.617
VGG CVGJ 34.945 52.390 59.835 69.097 81.465
DeepBit 18.469 34.317 43.249 53.590 70.363
SSDH 4.615 11.861 17.478 25.760 42.026
It is also necessary to evaluate the eﬃciency of these networks to know if it
is feasible to use them in real applications and for which scenarios. We measure
the time each method takes to extract the features and the time required by the
matching method to match an image against all images in the gallery. The sum
of both times is the identiﬁcation time of a query tattoo image. Fig. 5 shows the
time for the feature extraction of an image and the time to match it with all
images in the gallery, averaged over 100 images in the BIVTatt dataset. Similar
results were observed for the same experiment on the PinTatt dataset.
As can be seen in Fig. 5, SIFT is the fastest method for feature extraction
but it is too slow on the matching step. In an identiﬁcation scenario the match-
ing eﬃciency is critical because the matching algorithm must be executed for
each image in the gallery. We think that this fact makes SIFT a questionable
alternative for this kind of scenario. On the other hand, MobileNetV2 has the
best accuracy/eﬃciency relation which makes it a good option for tattoo iden-
tiﬁcation. These experiments show that CNNs trained for image classiﬁcation
on generic databases such as ImageNet, can be extended to the tattoo domain,
achieving good results as well.
5 Conclusions
In this article, we studied the use of intermediate features of deep neural net-
works, trained on generic databases, as descriptors of tattoo images. Unlike pre-
vious works where they used transfer learning to adjust the network to the
Deep Generic Features for Tattoo Identiﬁcation 9
0.0204 0.0269 0.0335 0.0647 0.0969 
0.3029 0.2449 0.2444 
2.3967 
0.0028 0.0028 0.0028 0.0031 0.0035 0.0071 0.0024 
0
0.5 
1
1.5 
2
SIFT MobileNetV1 MobileNetV2 Inception21k Resnet50 VGG CVGJ DeepB it SSDH 
Time (s) 
Time metrics (seconds) 
Mean feature extraction time Matching time of one image against all dataset images 
Fig. 5. Mean feature extraction and matching times in BIVTatt dataset.
context of tattoo identiﬁcation, we used the original pre-trained network mod-
els. The results of the identiﬁcation tests showed that by using this approach we
can obtain better results, both in eﬃciency and accuracy, than the handcrafted
solutions previously adopted, such as SIFT. In addition, the implementation cost
is minimal since there are many publicly available CNNs similar to those used
in this work. In future research, we will consider the use of tattoo detection and
segmentation methods to extract the background of the image.
References
1. Babenko, A., Slesarev, A., Chigorin, A., Lempitsky, V.: Neural codes for image
retrieval. In: European conference on computer vision. pp. 584–599. Springer (2014)
2. Dantcheva, A., Elia, P., Ross, A.: What else does your biometric data reveal? a sur-
vey on soft biometrics. IEEE Transactions on Information Forensics and Security
11(3), 441–467 (2016)
3. Di, X., Patel, V.M.: Deep Learning for tattoo recognition, pp. 241–256. Springer,
Cham (2017)
4. Han, H., Li, J., Jain, A.K., Chen, X.: Tattoo image search at scale: Joint detection
and compact representation learning. IEEE transactions on pattern analysis and
machine intelligence (2019)
5. Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., An-
dreetto, M., Adam, H.: Mobilenets: Eﬃcient convolutional neural networks for
mobile vision applications. arXiv preprint arXiv:1704.04861 (2017)
6. Ioﬀe, S., Szegedy, C.: Batch normalization: Accelerating deep network training by
reducing internal covariate shift. arXiv preprint arXiv:1502.03167 (2015)
7. Jain, A., Nandakumar, K., Ross, A.: 50 years of biometric research: Accom-
plishments, challenges, and opportunities. Pattern Recognition Letters 79, 80–105
(2016)
10 M. Nicol´ as-D´ ıaz et al.
8. Jain, A.K., Park, U.: Facial marks: Soft biometric for face recognition. In: Image
Processing (ICIP), 2009 16th IEEE International Conference on. pp. 37–40. IEEE
(2009)
9. Kim, J., Parra, A., Yue, J., Li, H., Delp, E.J.: Robust local and global shape context
for tattoo image matching. In: Image Processing (ICIP), 2015 IEEE International
Conference on. pp. 2194–2198. IEEE (2015)
10. Krizhevsky, A., Sutskever, I., Hinton, G.: Imagenet classiﬁcation with deep convo-
lutional neural networks. In: Advances in neural information processing systems.
pp. 1097–1105 (2012)
11. Lee, J.E., Jain, A.K., Jin, R.: Scars, marks and tattoos (smt): Soft biometric for sus-
pect and victim identiﬁcation. In: Biometrics Symposium, 2008. BSYM’08. pp. 1–8.
IEEE (2008)
12. Lin, K., Lu, J., Chen, C.S., Zhou, J.: Learning compact binary descriptors with
unsupervised deep neural networks. In: Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition. pp. 1183–1192 (2016)
13. Lowe, D.G.: Distinctive image features from scale-invariant keypoints. Interna-
tional journal of computer vision 60(2), 91–110 (2004)
14. Manger, D.: Large-scale tattoo image retrieval. In: Computer and Robot Vision
(CRV), 2012 Ninth Conference on. pp. 454–459. IEEE (2012)
15. Martin, M., Dawson, J., Bourlai, T.: Large scale data collection of tattoo-based
biometric data from social-media websites. In: Intelligence and Security Informatics
Conference (EISIC), 2017 European. pp. 135–138. IEEE (2017)
16. Ngan, M., Grother, P.: Tattoo recognition technology-challenge (tatt-c): an open
tattoo database for developing tattoo recognition research. In: Identity, Security
and Behavior Analysis (ISBA), 2015 IEEE International Conference on. pp. 1–6.
IEEE (2015)
17. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z.,
Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L.: Imagenet large
scale visual recognition challenge. International journal of computer vision 115(3),
211–252 (2015)
18. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C.: Mobilenetv2: In-
verted residuals and linear bottlenecks. In: Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition. pp. 4510–4520 (2018)
19. Simon, M., Rodner, E., Denzler, J.: Imagenet pre-trained models with batch nor-
malization. arXiv preprint arXiv:1612.01452 (2016)
20. Simonyan, K., Zisserman, A.: Very deep convolutional networks for large-scale
image recognition. arXiv preprint arXiv:1409.1556 (2014)
21. Sun, Z.H., Baumes, J., Tunison, P., Turek, M., Hoogs, A.: Tattoo detection and
localization using region-based deep learning. In: Pattern Recognition (ICPR),
2016 23rd International Conference on. pp. 3055–3060. IEEE (2016)
22. Xu, Q., Ghosh, S., Xu, X., Huang, Y., Kong, A.W.K.: Tattoo detection based on
cnn and remarks on the nist database. In: Biometrics (ICB), 2016 International
Conference on. pp. 1–7. IEEE (2016)
23. Yang, H.F., Lin, K., Chen, C.S.: Supervised learning of semantics-preserving hash
via deep convolutional neural networks. IEEE transactions on pattern analysis and
machine intelligence 40(2), 437–451 (2017)
View publication stats