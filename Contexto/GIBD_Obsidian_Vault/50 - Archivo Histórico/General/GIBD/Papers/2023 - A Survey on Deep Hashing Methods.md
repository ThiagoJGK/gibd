---
aliases: [2023 - A Survey on Deep Hashing Methods]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2025-07-22
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Papers/Índices/2023 - A Survey on Deep Hashing Methods.pdf"
tamanio_bytes: 1146928
---

# 2023 - A Survey on Deep Hashing Methods

Ruta interna: `GIBD/Papers/Índices/2023 - A Survey on Deep Hashing Methods.pdf`

---

See discussions, stats, and author profiles for this publication at: https://www.researchgate.net/publication/360233783
A Survey on Deep Hashing Methods
Article  in  ACM Transactions on Knowledge Discovery from Data · April 2022
DOI: 10.1145/3532624
CITATIONS
160
READS
176
7 authors, including:
Xiao Luo
University of California, Los Angeles
154 PUBLICATIONS   1,851 CITATIONS   
SEE PROFILE
Haixin Wang
University of California, Los Angeles
28 PUBLICATIONS   312 CITATIONS   
SEE PROFILE
Daqing Wu
Peking University
20 PUBLICATIONS   303 CITATIONS   
SEE PROFILE
Chong Chen
Peking University
75 PUBLICATIONS   1,087 CITATIONS   
SEE PROFILE
All content following this page was uploaded by Xiao Luo on 27 April 2023.
The user has requested enhancement of the downloaded file.
15
A Survey on Deep Hashing Methods
XIAO LUO, HAIXIN WANG,a n dDAQING WU,Peking University, China
CHONG CHEN,Alibaba Group, China
MINGHUA DENG,Peking University, China
JIANQIANG HUANGand XIAN-SHENG HUA,Alibaba Group, China
Nearestneighborsearchaimsatobtainingthesamplesinthedatabasewiththesmallestdistancesfromthem
to the queries,which isa basic taskin a range offields, including computer vision and data mining. Hashing
is one of the most widely used methods for its computational and storage efficiency. With the development
of deep learning, deep hashing methods show more advantages than traditional methods. In this survey, we
detailedlyinvestigatecurrentdeephashingalgorithmsincludingdeepsupervisedhashinganddeepunsuper-
vised hashing. Specifically, we categorize deep supervised hashing methods into pairwise methods, ranking-
basedmethods,pointwisemethodsaswellasquantizationaccordingtohowmeasuringthesimilaritiesofthe
learnedhashcodes.Moreover,deepunsupervisedhashingiscategorizedintosimilarityreconstruction-based
methods, pseudo-label-based methods,and prediction-free self-supervisedlearning-based methods based on
theirsemanticlearningmanners.Wealsointroducethreerelatedimportanttopicsincludingsemi-supervised
deep hashing, domain adaption deep hashing, and multi-modal deep hashing. Meanwhile, we present some
commonly used public datasets and the scheme to measure the performance of deep hashing algorithms.
Finally, we discuss some potential research directions in conclusion.
CCS Concepts: • Information systems →Collaborative search; Similarity measures; Top-k retrieval in
databases;
Additional Key Words and Phrases: Approximate nearest neighbor search, learning to hash, top-k retrieval,
similarity preserving, deep supervised hashing
ACM Reference format:
Xiao Luo, Haixin Wang, Daqing Wu, Chong Chen, Minghua Deng, Jianqiang Huang, and Xian-Sheng Hua.
2023.ASurveyonDeepHashingMethods. ACMTrans.Knowl.Discov.Data. 17,1,Article15(February2023),
50 pages.
https://doi.org/10.1145/3532624
The work was done when Xiao Luo and Daqing Wu interned in Damo Academy, Alibaba Group.
This work was supported by the National Key Research and Development Program of China (2021YFF1200902) and the
National Natural Science Foundation of China (31871342).
Authors’ addresses: X. Luo, H. Wang, D. Wu, and M. Deng (corresponding author), Peking University, Beijing, 100871,
China; emails: xiaoluo@pku.edu.cn, wang.hx@stu.pku.edu.cn, wudq@pku.edu.cn, dengmh@math.pku.edu.cn; C. Chen
(corresponding author), J. Huang, and X.-S. Hua, Alibaba Group, Hangzhou, 310000, China; emails: cheung.cc@alibaba-
inc.com, {jianqiang.jqh, huaxiansheng}@gmail.com.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and
the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions frompermissions@acm.org.
© 2023 Association for Computing Machinery.
1556-4681/2023/02-ART15 $15.00
https://doi.org/10.1145/3532624
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.

15:2 X. Luo et al.
1 INTRODUCTION
Nearest neighbor search is among the most basic tasks in various domains including data min-
ing [196] and image retrieval [16]. There have been a variety of algorithms for exact nearest
neighbor search, such as KD-tree [42, 165]. Unfortunately, when it comes to high-dimensional
and large-scale data, the time cost of accurately identifying the sample nearest to the query is
substantial. To tackle the challenge, the approximate nearest neighbor search has received ever-
increasing attention since it could significantly decrease the search complexity under most cir-
cumstances [36, 120, 165]. Hashing is one of the most widely-used methods, because it is very
efficient in terms of computation and storage [10]. Its purpose is to convert the high-dimensional
features vectors into low-dimensional hash codes, so that the hash codes of the similar objects
are as close as possible, and the hash codes of dissimilar objects are as different as possible. The
existinghashingmethodsconsistoflocalsensitivehashing[ 22,75]andlearningtohash.Thepur-
poseoflocalsensitivehashingistomaptheoriginaldataintoseveralhashbuckets.Thecloserthe
originaldistance betweenobjectsis,the greaterthe probabilityoffalling in thesame hashbucket.
Throughthismechanism,manyalgorithmsbasedonlocallysensitivehashinghavebeenproposed
[6, 7, 32, 33, 127, 131], which show high superiority in both calculation and storage. However, in
ordertoimprovetherecallrateofsearch,thesemethodsusuallyneedtobuildmanydifferenthash
tables, so their applications on particularly large data sets are still limited.
Since local sensitive hashing is data-independent, researchers try to get high-quality hashing
codes by learning good hash functions. As two pioneering methods, i.e., spectral hashing and
semantic hashing, have been proposed [136, 171], learning to hash has sparked considerable aca-
demicinterestinbothmachine-learninganddatamining.Withthedevelopmentofdeeplearning
[93], getting hash codes through deep learning gets more and more attention for two reasons.
The first reason is that the powerful representation capabilities of deep learning can learn very
complex hash functions. The second reason is that deep learning can achieve end-to-end hashing
codes, which is very useful in many applications. In this survey, we mainly focus on deep super-
vised hashing methods and deep unsupervised hashing methods, which are two mainstreams in
hashing research. Moreover, three related important topics including semi-supervised deep hash-
ing, domain adaption deep hashing, and cross-modal deep hashing are also included.
Deep supervised hashing has been explored over a long period. The design of the deep super-
vised hashing method mainly includes two parts: the design of the network structure and the
design of the loss function. For small datasets like MINST [94] and CIFAR-10 [89], shallow archi-
tecture such as AlexNet [90] and CNN-F [23] are widely used. While for complex datasets like
NUS-WIDE [29] and COCO [109], deeper architecture such as VGG [149] and ResNet50 [61]a r e
needed. The loss objectives are designed with the intention of maintaining similarity structures.
Thesemethods[ 15,104]usuallyaimatnarrowingthedifferencebetweenthesimilaritystructures
in the original and Hamming spaces. Researchers usually obtain the similarities in the original
spacebyusinglabelinformationinsupervisedscenarios,whichiswidelystudiedindifferentdeep
hashing methods. Hence, how obtaining the similarities of learned hash codes are important for
different algorithms. We further categorize the deep supervised hashing algorithms according to
how measuring the similarities of learned hash codes into four classes, i.e., pairwise methods,
ranking-based methods, pointwise methods and quantization. For each manner, we comprehen-
sively analyze how the related articles design the optimization objective and take advantage of
semantic labels, as well as what additional tricks are used.
Another area of research along this line is deep unsupervised hashing, which does not require
anylabelinformation.Deepunsupervisedhashinghasdrawnwidespreadattentionrecently,since
itiseasilyappliedinpractice.Inunsupervisedsettings,thesemanticinformationisusuallyderived
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:3
Fig. 1. The overall structure of this survey.
from the relationship in the original space. Based on manners of learning semantic information,
we categorize the deep unsupervised hashing algorithms into pseudo-label-based methods, simi-
larity reconstruction-based methods, and prediction-free self-supervised learning-based methods.
In addition, we also introduce some other related important topics such as semi-supervised deep
hashing, domain adaptation deep hashing, and multi-modal deep hashing methods. The overall
structure of this survey is shown in Figure1. Meanwhile, we also present some commonly used
publicdatasetsandtheschemetomeasuretheperformanceofdeephashingalgorithms.Atlast,a
comparison of some key algorithms was given.
Compared to other surveys on hashing methods [18, 163–165], our article mainly centers on
recentdeephashingmethodsratherthantraditionalhashingmethodsandhowtheyoptimize the
hashingnetwork.Moreover,westudybothdeepsupervisedhashingaswellasdeepunsupervised
hashing extensively. Finally, we classify two topics in a brand-new view based on the different
manners of optimization. As far as we know, this is the most comprehensive survey about deep
hashing, which is beneficial to researchers in understanding the mechanisms and trends of deep
hashing.
2 BACKGROUND
2.1 Nearest Neighbor Search
Givena d-dimensionalEuclideanspace Rd,thenearestneighborsearchaimsatfindingthesample
NN(xr) in a finite setΠ⊂Rd such that
NN(xr) = arg min
xb∈Π
ρ(xr,xb), (1)
in whichxr ∈Rd represents the query point. Note thatρcould be any metrics such as Euclidean
distance, cosine distance along with generalℓp distance. Many exact nearest neighbor search
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:4 X. Luo et al.
methods [42] have been developed by the researchers, which works quite well whend is small.
However, nearest neighbor search is intrinsically costly due to the curse of dimensionality [3, 4].
Although KD-tree can be extended to high-dimensional situations, its efficiency is far from
satisfactory.
To solve this problem, a series of algorithms for approximate nearest neighbors have been pro-
posed[33,46,79,128].Theprincipleofthesemethodsistofindthenearestpointwithahighprob-
ability,ratherthantofindthenearestpointaccurately.TheseANNalgorithmsaremainlydivided
into three categories: hashing-based methods [1, 33, 123], product quantization-based methods
[44,79,86,190],andgraph-basedmethods[ 56,125,126].Thesealgorithmshavegreatlyimproved
theefficiencyofsearchingwhileensuringarelativelyhighaccuracy,sotheyarewidelyusedinthe
industry. Compared to the other two types of methods, hashing-based algorithms are the longest
studiedandthemoststudiedbyresearchers,becauseithasgreatpotentialinimprovingcomputing
efficiency and reducing memory cost.
2.2 Hashing Algorithms
Fornearestneighborsearch,hashingalgorithmsareveryefficientintermsofbothcomputingand
storage. Two main types of hashing-based search methods have been developed, i.e., hash table
lookup [96,151] and hash code ranking [80, 116].
The primary goal of hash table lookup is to decrease the number of distance calculations for
speeding up searches. The structure of the hash table contains various buckets, each of which is
indicated by one separate hash code. Each point is associated with a hash bucket that shares the
same hash code. Thus, the manner to learn hash codes for this kind of algorithm is to increase
thelikelihoodofproducingthesamehashcodesforadjacentpointsintheoriginalspace.Whena
queryisgiven,wecanfindthecorrespondinghashbucketaccordingtothehashcodeofthequery,
so as to find the corresponding candidate set. After this step, we usually re-rank the points in the
candidatesettogetthefinalsearchtarget.However,therecallofselectingasinglehashbucketas
acandidatesetwillberelativelylow.Twomethodsareusuallyadoptedtoovercomethisproblem.
The first method is to select some buckets that are close to the target bucket at the same time.
Thesecondmethodistoindependentlycreatemultipledifferenthashtablesaccordingtodifferent
hash codes. Then, we can select the corresponding target bucket from each hash table.
Hash code ranking is a relatively easier way than hash table lookup. When a query comes, we
compute the Hamming distance between the query and each point in the searching dataset, then
selectthepointswithrelativelysmallerHammingdistancesasthecandidatesfornearestneighbor
search. After that, a re-ranking process by the original features is usually followed to obtain the
final nearest neighbor. Different from hash table lookup methods, hash code ranking methods
prefer hash codes that preserve the similarities or distances in the original space.
2.3 Deep Neural Networks
Deep neural networks [137] have achieved significant success in various areas including com-
puter vision [38, 61] and natural language processing [52, 154]. Early works such as deep belief
network[63],andautoencoder[ 129]aremostlybasedonmulti-layerperceptions.However,these
networks do not show much better performance compared with traditional methods such as sup-
port vector machine and k-nearest neighbors algorithm. As convolutional neural networks have
been introduced to process image data, various popular deep networks have been proposed and
achieved promising results. AlexNet [90] consists of five convolutional layers followed by three
fully connected layers. VGGNet [149] increases the model depth and improves the performance
of image classification. NIN [107] is further proposed to promote the discriminability of image
patches within the receptive field. Researchers have found that the depth of representations
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:5
is the key to high performance for various visual recognition tasks. However, the problem of
vanishing/exploding gradients makes it difficult to build very deep neural networks. ResNet [61]
tackles this problem by leveraging the residual learning to deepen the network and benefits
from very deep models. Recently, Vision Transformer [38] has achieved great success on image
classification tasks due to its high model capacity and easy scalability. These powerful neural
network architectures has become the backbone networks in various applications, including
semantic segmentation [155] and object detection [57]. In virtue of the strong representation
ability of deep neural networks, deep hashing has shown great performance in image retrieval
and drawn increasing attention recently.
2.4 Learning to Hash
Given an input itemx, learning to hash aims at obtaining a hash functionf,w h i c hm a p sx to a
binary codeb for the convenience of the nearest neighbor search. The hash codes obtained by a
good hash function should preserve the distance order in the original space as much as possible,
i.e., those items that are close to the specific query in Hamming space should also be close to
the query in the original space. Many traditional hash functions include sphericalfunction, linear
projection, and even a non-parametric function. A wide range of traditional hashingmethods [46,
48, 58, 113, 140, 153, 165, 171, 196] have been proposed by researchers to learn compact hash
codes, and achieved significant progress. For instance, AQBC [47] utilizes the angle between two
vectors to measure the similarity and maps feature vectors into the most similar vertices of a
binary hypercube. FSDH [53] regresses the semantic labels of samples to their binary codes and
optimizesthehashcodesinanalternativemanner.Foramorecomprehensiveunderstanding,refer
toasurveyarticle[ 163].However,thesesimplehashfunctionsdonotscalewellforhugedatasets.
For the strong representation ability of deep learning, more and more researchers pay attention
to deep supervised hashing and develop a range of promising methods. These methods generally
achieve better performance than traditional methods.
3 DEEP SUPERVISED HASHING
In this article, we first talk about deep supervised hashing methods, which are the basis of the
subsequent deep unsupervised hashing techniques.
3.1 Overview
Deep supervised hashing uses deep neural networks as hash functions, which can generate hash
codes in an end-to-end manner. We focus on the following four key problems: (1) what deep neu-
ralnetworkarchitectureisadopted;(2)howtodesign thelossfunctionforpreservingthesimilar-
ity structure; (3) how to optimize the deep neural network with the discretization problem; and
(4) what other skills can be used to improve the performance. We first answer the first three prob-
lems in a nutshell and the last problem is left in the subsequent detailed introduction. Figure2
shows a representative framework of deep supervised hashing.
3.1.1 Network Architecture. Traditional hashing methods usually utilize linear projection and
kernels, which show poor representation ability. After AlexNet and VGGNet [90, 149]w e r ep r o -
posed, deep learning shows its superiority in computer vision, especially for classification prob-
lems.Andmoreandmoreexperimentshaveprovedthatthedeeperthenetwork,thebettertheper-
formance.Asaresult,ResNet[ 61]takesadvantageofresiduallearning,whichcantrainverydeep
networks, achieved significantly better results. After that, ResNet and its variants have become
basic architectures in deep learning [61, 71, 175]. The latest researches often utilize the popular
architectures with pre-trained weights in large datasets such as ImageNet, following the idea of
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:6 X. Luo et al.
Fig.2. BasicFrameworkofDeepSupervisedHashingwithPairwiseSimilarityMeasurement.Thehashcodes
are produced by a hashing network. Afterwards, the pairwise similarity information of hash codes and
groundtruthismatchedandthusgetsimilaritypreservingloss.MoredetailswillbediscussedinSection 3.2.
transfer learning. Most of the researchers utilize shallower architectures such as AlexNet, CNN-F,
and design stacked convolutional neural networks for simple datasets, e.g., MNIST, CIFAR-10.
Deeper architectures such as VGGNet and ResNet50 are often utilized for complex datasets such
as NUS-WIDE and COCO. To be more precise, for deep supervised hashing methods, the hashing
networkisusuallymodifiedfromtheseaforementionedstandardnetworksbyreplacingtheclassi-
ficationheadwithafully-connectedlayercontaining Lunitsforhashcodelearning.Thenetwork
outputs are usually continuous codes. The hash codes can be obtained using a sign activation.
Graph neural networks, which capture the dependence between the nodes of graphs via message
passing mechanisms, have been popular in various applications. They have also been adopted in
recent hashing methods to learn the correlation of datasets [30, 157,167].
The architecture of the hashing network is one of the most important factors for deep super-
vised hashing, and it affects both the accuracy of the search and the time cost of inference. If the
architecturedegeneratesintoMLPorlinearprojections,deepsupervisedhashingwilldegradeinto
traditionalhashingmethods.Althoughthedeeperthenetworkarchitecture,thegreaterthesearch
accuracy,italsoincreasesthetimecost.Wethinkthatthearchitectureneedstobeconsideredcom-
binedwiththecomplexityofdatasets.Asweknow,themajorityofexistingdeephashingmethods
canuseanynetworkarchitectureasneeded.Therefore,wedonotadoptthenetworkarchitectures
for categorizing the deep supervised hashing algorithms.
3.1.2 SimilarityMeasurementandObjectiveFunction. Wefirstprovideformalnotationsandkey
concepts in Table1 for the sake of clarity.X ={xi}N
i=1 is denoted as the training set.H ={hi}N
i=1
denotes the outputs of the hashing network, i.e.,hi = Ψ(xi).B ={bi}N
i=1 is the obtained binary
codes. We denote the similarity between pair of items(xi,xj) in the input space and Hamming
space asso
ij and sh
ij , respectively. In the input space, the similarity is the ground truth, which is
mainly based on sample distancedo
ij and semantic labels. The former refers to the distance of
features, e.g., Euclidean distance||xi −xj||2, and the similarity can be computed using Gaussian
function or Characteristic function, i.e., exp(−
(do
ij )2
2σ2 ) and Ido
ij <τwhereτis a given threshold. The
cosine similarity is also popular for measurement. The latter is more popular in deep supervised
hashing, where the value is 1 if the two examples share a common semantic label and 0 vice visa.
Thepairwisedistance dh
ij intheHammingspaceisHammingdistancenaturally,whichisdefined
as follows:
dh
ij =
L∑
l =1
δ
[
bi(l) /nequalbj(l)
]
. (2)
If the hash code is valued by 1 and 0, we have:
dh
ij =∥bi−bj∥1, (3)
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:7
Table 1. Summary of Symbols and Notation
Symbol Description
xi (X) input images (in matrix form)
bi (B) output hash codes (in matrix form)
hi (H) network outputs (in matrix form)
yi (Y) one-hot image labels (in matrix form)
Ψ(·) hashing network
N the number of input images
L hash code length
E a set of pair items
so
ij the similarity of item pair(xi,xj) in the input space
sh
ij the similarity of item pair(xi,xj) in the Hamming space
do
ij the distance of item pair(xi,xj) in the input space
dh
ij the distance of item pair(xi,xj) in the Hamming space
ϵ margin threshold parameter
W weight parameter matrix
Θ set of neural parameters
anditvariesfrom0to L.Asaresult,thesimilarityinthiscircumstanceisdenotedas sh
ij = (L−dh
ij )/L.
If the code is valued by 1 and−1, we have:
dh
ij = 1
2(L−bT
i bj). (4)
The similarity is defined using the inner product, i.e.,sh
ij = (b⊤
i bj +L)/2L. We can also extend
this to the weighted circumstance. In formulation,
dh
ij =
L∑
l =1
λlδ
[
bi(l) /nequalbj(l)
]
, (5)
where each bit is associated with a weightλl, and if the values of codes are 1 and−1, we have
sh
ij = (b⊤
i Λbj +tr(Λ))/2tr(Λ), (6)
in whichΛ = diag(λ1,λ2,..., λl) is diagonal andtr(·) denotes the trace of the matrix. The weight
of the associated hash bit fills each diagonal element of the matrix.
After defining the similarity measurement, we focus on the objective functions in deep super-
visedmethods.Awell-designedobjectivefunctionsisoneofthemostimportantfactorstopromise
theperformanceofdeepsupervisedhashing.Themainguidelinefordesigningtheobjectivefunc-
tion is to keep the similarity structure, which means to minimize the difference between the simi-
laritiesintheoriginalandHammingspaces.Asaresult,mostoftheobjectivefunctionscontainthe
terms of similarity information. Among them, the typical loss functions are in a pairwise manner,
making similar pairs of images have similar hash codes (small Hamming distance) and dissim-
ilar pairs of images have dissimilar hash codes (large Hamming distance). Besides, a variety of
researchers adopt ranking-based similarity preserving loss terms. For example, triplet loss is of-
ten used to maintain as much consistency as possible between the ordering of numerous items
calculated from the original and Hamming spaces. There are also several listwise loss terms that
consider the whole datasets for similarity preserving.
Besides similarity information, the pointwise label information is also well-explored in the de-
signoftheobjectivefunction.Therearethreepopularwaystotakeadvantageoflabelinformation
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:8 X. Luo et al.
summarized below. The first way is a regression on hash codes with labels. The label is encoded
into one-hot format matrix and regression loss, i.e.,||Y−WH||F are added into the loss function.
Thesecondwayisaddingaclassificationlayerafterthehashingnetwork,andaclassificationloss
(e.g.,cross-entropyloss)isaddedtotheobjectivefunction.ThelastoneisutilizingLabNet,which
was first proposed in [99]. LabNet aims at capturing the ample semantic relationships among ex-
ample pairs.
The quantization loss term is also commonly used in deep supervised hashing, especially in
quantization-based hashing methods. The typical form of quantization is to penalize the distance
between continuous codes (i.e., network outputs) and binary codes. As a common technique in
deephashing,bitbalancelosspenalizesthesituationthateachbithasalargechanceofbeing1or
−1amongthewholedataset.Severalregularizationlossescanbeaddedtothelossfunction,which
is also important for improving the performance.
3.1.3 Optimization Algorithm. It is difficult to optimize the hashing network parameters be-
cause of the vanishing gradient issue resultingfrom the sign activation function, which is used to
obtain binary hash codes. Specifically, the sign function is in-differentiable at zero and its gradi-
ent is zero for all nonzero input, which is fatal to the hashing network using gradient descent for
optimization.
Almost all the works adopt that continuous relaxation by smoothing the sign function using
the sigmoid function or the hyperbolic tangent function, and apply sign function to obtain final
binarycodeslaterintheevaluationphase.Thefirsttypicalwayisquantizationfunctionbyadding
a penalty term in loss function, which is often formulated as|||hi|−1||1,o r−||hi|| with tanh
activation. This penalty term helps the neural network to obtainsдn(hi)≈hi.N o t et h a tt h i sl o s s
can be considered as a prior for every binary codehi on basis of a variant of certain distribution,
e.g., bimodal Laplacian and Cauchy distribution. From this view, we can get a few variants, e.g.,
pairwisequantization[ 200]andCauchyquantizationloss[ 15].Ifthelossfunctionisanon-smooth
functionanditsderivativeishardtocalculate,amodifiedversioncanbeadoptedinstead,e.g., |x|≈
log(coshx) [200]. The second way is an alternative scheme, which resolves the optimization into
severalsub-problems.Then,thesesub-problemscouldbeiterativelysettledthroughalternatingthe
minimizationofobjectives.Inthisalternativeprocess,backpropagationcanonlyworkinonesub-
problem, and the other sub-problems can be solved by other optimization methods. For example,
DSDH [100] utilizes the discrete cyclic coordinate descend algorithm. These methods can keep
the discrete constraint during the whole optimization process, while it can not lead to end-to-end
training, which has limited application for solving the unknown sub-problems. The third method
is named continuation, which utilizes a smoothed functiony =tanh(βx) to approach the discrete
activation function by increasingβ[19]. There are some other ways to solve this problem by
changing the calculation and the propagation of gradients, e.g., Greedy Hash [156] and Gradient
AttentionNetwork[ 72],whichimprovetheeffectivenessandaccuracyofdeepsupervisedhashing.
3.1.4 Summarization. In this survey, we divide the current methods into the following four
classes mainly based on how to measure the similarities in the Hamming space: the pairwise
methods, the ranking-based methods, the pointwise methods, and the quantization methods. The
quantization methods are separated from the pairwise methods due to their specificity. The key
motivation we select how to measure the similarities of the learned hash codes for categorization
isthatthefundamentalcoreoflearningtohashistomaintainthesimilaritystructureandtheman-
nersofsimilaritymeasurementdecidethelossfunctionsindeepsupervisedhashing.Additionally,
neural network architectures, optimization manners as well as other skills are also significant for
theretrievalperformance.Foreachclass,wewilldiscussthecorrespondingdeephashingmethods
in detail one by one. The detailed summarization of these methods is shown in Table2.
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:9
Table 2. A Summary of Deep Supervised Hashing Methods w.r.t the Different Manner of Similarity
Measurement (Pairwise Methods, Ranking-based Methods, and Pointwise Methods),
Binarization as well as Other Skills
Approach Pairwise Ranking-based Pointwise Binarization Other skills
SDH [40] Prod. - - Quan. Bit Bal. + Orthogonality
DSH [111] Prod. + Margin - - Quan. -
PCDH [27] Prod. + Margin - Cla. Layer Drop Pairwise Correlation
WMRSH [98] Prod. + Margin - Cla. Layer Quan. Bit and Table Weight
SHBDNN [36] Diff. - - Quan. + Alternation Bit Bal. + Independence
DDSH [82] Diff. - - Alternation Splitting Training Set
CNNH [173] Diff. - Part of Hash Codes - Two-step
ADSH [84] Diff. - - Quan. + Alternation Asymmetry
DIH [172] Diff. - - Quan. + Alternation Incremental Part + Bit Bal.
HBMP [9] Diff. - - Drop Bit Weight + Two-step
DOH [85] Diff. - - Ranking FCN
DPSH [101] Like. - - Quan. -
DHN [200] Like. - - Quan. + Smooth -
HashNet [19] Weighted Like. - - Tanh + Continuation -
DSDH [100] Like. - Linear Reg. + L2 Quan. + Alternation -
DAPH [139] Like. - - Quan. + Alternation Bit Bal. + Ind.
DAgH [180] Like. + Diff. - - - Two-step
DCH [15] Cauchy Like. - - Cauchy Quan. -
DJSEH [99] Like. - LabNet + Linear Reg. Quan. Two-step + Asymmetry
ADSQ [181] Diff. + Like. - LabNet Quan. + Alternation Bit Bal. + Two-step
MMHH [87] t-Distribution. Like. - - Quan. Semi-Batch Optimization
DAGH [26] Like. - Linear Reg. Drop + Alternation Reg. with Anchor Graph
HashGAN [14] Weighted Like. - - Cosine Quan. GAN
DPH [20] Priority Like. - - Priority Quan. Priority CE Loss
DFH [103] Like. + Margin - - Quan. + Alternation Quantized Center Loss
DRSCH [189] Diff. + Margin Triplet + Margin - Drop Bit Weight
DNNH [92] - Triplet + Margin - Piecewise Thresholding -
DSRH [197] - Weighted Triplet - Quan. Bit Bal.
DTSH [169] - Triplet + Like. - Quan. -
DSHGAN [133] - Triplet + Margin Cla. Layer Drop GAN
AnDSH [198] - Matrix Optimization Angular-softmax Drop Bit Bal.
HashMI [8] - Mutual Information - Drop -
TALR [59] - Relaxed AP + NDCG - Tie-Awareness
MLRDH [117] - - Multi-linear Reg. Alternation Hash Boosting
HCBDH [25] - - Cla. Layer - Hadamad Loss
DBH [106] - - Cla. Layer - Transform Learning
SSDpH [180] - - Cla. Layer Quan. Bit Bal.
VDSH [195] - - Linear Reg. Drop + Alternation -
PMLR [144] - - Cla. Layer - Distribution Regu.
CSQ [185] - - Center + Binary CE Quan. -
DPH [41] - - Center + Polarization - -
OrthHash [64] - - Center + CE - -
PSLDH [158] - - Center + Partial Loss Quan. -
DVsQ [16] - Triplet Loss + Margin - Inner-Product Quan. Label Embeddings
DPQ [88] - - Cla. Layer - Joint Central Loss
DSQ [39] - - Cla. Layer Quan. + Alternation Joint Central Loss
SPDAQ [24] Diff. - Cla. Layer Drop + Alternation Asymmetry
DQN [17] Diff. - - Product Quan. Asymmetric Quan. Distance
DTQ [110] - Triplet + Margin - Weak-Orthogonal Quan. Group Hard
Drop = Drop the sign operator in the neural network and treat the binary code as an approximation of the network
output, Two-step = Two-step optimization. Reg. = Regression, Quan. = Quantization Loss, Cla. = Classification, Ind. =
Independence, Regu. = Regularization, Bal. = Balance, Diff. = Difference Loss, Prod. = Product Loss, and Like. =
Likelihood Loss.
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:10 X. Luo et al.
3.2 Pairwise Methods
Wefurtherdividethetechniques,whichmatchthedistancesorsimilaritiesofimagepairsderived
from two spaces, i.e., the original space and the Hamming space into two parts as follows:
—Difference loss minimization. The kind of losses minimizes the difference between the simi-
larities, i.e., min∑
(i,j)∈E(so
ij −sh
ij )2 [9, 36, 82, 84, 85, 172, 173].sh
ij can be derived with inner
productofbinarycodes,i.e., sh
ij =bT
i bj/L andso
ij isnowvaluedby1or −1.However,binary
optimization is difficult to implement. Early methods utilizes the relaxed outputs of neural
networks to replace the hash codes, i.e.,sh
ij = hT
i hj/L [36, 173]. Subsequent methods utilize
a asymmetric manner to calculate the similarity, i.e.,sh
ij = bT
i hj/L, which releases the im-
pactofquantizationerror[ 82,85,172].Therearealsoworkscombiningbothsymmetricand
asymmetricsimilarities[ 84].Weightedbitsarealsointroducedforadaptivesimilaritycalcu-
lation[9].Notethatthedifferencelossescanbetransformedintoaproductform.Hence,we
alsocategorizethemethodsminimizingproductlossasthisgroup.Theyusuallyadoptaloss
in the product form, i.e., min∑
(i,j)∈Eso
ijdh
ij [40], which expects that if the similarities in the
original space are higher, the distances in the Hamming space should be less. Subsequent
methods usually involve a margin in the loss for better relaxation [27, 98,111].
—Likelihood loss minimization. This kind of losses is derived from the probabilistic model.
Given similarity matrixS = {so
ij}(i,j)∈Eand hash codesB = [b1,..., bN]T, the posterior
estimation of binary codes is formulated as follows:
p(B|S)∝p(S|B)p(B) =
∏
(i,j)∈E
p
(
so
ij|B
)
p(B), (7)
wherep(B) denotes a prior distribution andp(S|B) is the likelihood. The conditional proba-
bility ofso
ij given their hash codes is denoted byp(so
ij|B).N o t et h a tt h esh
ij is derived fromB.
In formulation,
p
(
so
ij|B
)
=p
(
so
ij|sh
ij
)
=
⎧⎪⎨⎪⎩
σ
(
sh
ij
)
, so
ij = 1
1−σ
(
sh
ij
)
, so
ij = 0, (8)
inwhich σ(x) = 1/(1+ex).FromEquation( 8),theprobabilisticmodelexpectsthesimilarities
intheHammingspacetobelargerifthesimilaritiesintheoriginalspacearelarger.Theloss
function is thenegative log-likelihood (NLL)[101, 200], i.e.,
LNLL =−logp(S|H) =
∑
(i,j)∈E
log(1 +esh
ij )−so
ijsh
ij. (9)
Similarly, the hashing network usually cannot directly obtain the hash codes. Hence, these
codes B will be replaced by the network outputsH to generatesh
ij . The majority of meth-
odsadoptthesymmetricsimilarities,whileseveralmethodsutilizetheasymmetricformfor
similaritycalculation[ 99,139].However,thesigmoidfunctioninEquation( 8)isnotoptimal,
and there are a number of works that utilize different tools to design valid probability func-
tions, e.g., priority weighting [20], Cauchy distribution [16], imbalance learning [19]a n d
t-Distribution [87]. Subsequent works combine label information with pairwise similarity
learning for better semantic preserving [26, 99, 100]. Li et al. [103] associate the likelihood
losswithFisher’sLineardiscriminant,andintroduceamarginfordiscriminativehashcodes.
Chenetal.[ 26]reducethecomputationalcostbyintroducinganchorsforsimilaritycalcula-
tion.Therearealsosomeworkscombiningbothdifferencelossminimizationandlikelihood
loss minimizing for comprehensive optimization [182].
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:11
Although these methods in each group could utilize the same pairwise loss term, they may in-
volve differentarchitectures,optimizationmannersandregularizationterms,asshowninTable 2.
These details of different variants will be shown below.
3.2.1 Difference Loss Minimization. Deep Supervised Hashing (DSH)[ 111]. DSH unitizes a
network consisting of three convolutional-pooling layers and two fully connected layers. Recall
that the outputs of the hashing network are{hi}N
i=1. The origin pairwise loss function is defined
as follows:
LDSH =
∑
(i,j)∈E
1
2so
ijdh
ij + 1
2(1−so
ij )[ϵ−dh
ij ]+
s.t.∀hi,hj ∈{−1,1}L,
(10)
where dh
ij = ||hi −hj||2
2,[·]+ denotes max(·,0) and ϵ> 0 is a given threshold parameter. The
loss function obeys a distance-similarity product minimization formulation that expects similar
examples mapped to similar binary codes and rewards dissimilar examples transferred to distinct
binary codes when the Hamming distances are smaller compared with the margin thresholdm.I t
is noticed that whendh
ij is larger thanm, the loss does not produce gradients. This idea is similar
to the hinge loss function.
As we discuss before, DSH relaxes the binary constraints and a regularizer is added to the con-
tinuous outputs of the hashing network, which approximates the binary codes, i.e.,h≈sдn(h).
The pairwise loss is rewritten as
LDSH = 1
2so
ij||hi−hj||2
2 + 1
2(1−so
ij )[ϵ−||hi−hj||2
2]+ +λ1
∑
k=i,j
|||hk|−1||1, (11)
where 1 denotes a all-one vector and||·|| p produces theℓp-norm of the vector.λ1 is a parameter
to balance the effects of the regularization loss. DSH does not utilize saturating non-linearities
because it may slow down the training process. With the above loss function, the neural network
is able to be trained with an end-to-end back propagation algorithm. For the evaluation process,
thebinarycodescanbederivedusingthesignactivationfunction.DSHisastraight-forwarddeep
supervisedhashingmethodintheearlyperiod,anditsideaoriginatesfromSpectralHashing[ 171]
but with a deep learning framework.
Pairwise Correlation Discrete Hashing(PCDH) [27]. PCDH utilizes four fully connected layers
aftertheconvolutional-poolinglayer,nameddeepfeaturelayer,hash-likelayer,discretehashlayer
aswellasclassificationlayer,respectively.Thethirdlayercandirectlygeneratediscretehashcode.
Different from DSH, PCDH leveragesℓ2 norm of deep features and hash-like codes. Besides, the
classification loss is included in the final function:
LPCDH =Ls +λ1Lp +βLl
=
∑
(i,j)∈E
(1
2(1−so
ij )[ϵ−∥hi−hj∥2
2]2
+ + 1
2sij∥hi−hj∥2
2
)
+λ1
∑
(i,j)∈E
(1
2
(
1−so
ij
)
[ϵ−||zi−zj||2
2]2
+ + 1
2so
ij∥zi−zj∥2
2
)
+λ2
/parenlefttpA/parenleftexA
/parenleftbtA
N∑
i=1
ϕ
(
wT
i bi,yi
)
+
N∑
j=1
ϕ
(
wT
j bj,yj
)/parenrighttpA/parenrightexA
/parenrightbtA
,
(12)
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:12 X. Luo et al.
where zi,hi, andbi denote the outputs of the first three fully connected layers. The last term is
theclassificationcross-entropyloss. 1 Notethatthesecondtermiscalledpairwisecorrelationloss,
which guides the similarity learning of deep features to avoid overfitting. The classification loss
provides semantic supervision, which helps the model achieve competitive performance. Besides,
PCDH proposes a pairwise construction module named Pairwise Hard, which samples positive
pairs with the maximum distance between deep features and negative pairs with the distances
smaller than the threshold randomly. It is evident that Pairwise Hard chooses the hard pairs with
the large loss for effective hash code learning.
Supervised Deep Hashing(SDH) [40]. SDH utilizes the fully-connected neural network for deep
hashing and has a similar loss function except for a term that enforces a relaxed orthogonality
constraint on all projection matrices (i.e., weight matrices in a neural network) for the property
of fully-connected layers. Bit balance regularization is also included, which will be introduced in
Equation (14).
Supervised Hashing with Binary Deep Neural Network(SH-BDNN) [36]. The architecture of SH-
BDNN is stacked by a fully connected layer, in whichWi denotes the weights in theith layer.
SH-BDNN not only considers the bit balance, i.e., each bit obeys a uniform distribution, but also
considers the independence of different hash bits. Given the hash code matrixB = [b1,..., bN]T,
the two conditions are formulated as
BT 1 = 0, 1
NBTB = I, (13)
where 1 is aL-dimension vector whose elements are all one, andI is an identity matrix of sizeN
by N. The loss function is
LSH-BDNN = 1
2N
/bardblex/bardblex/bardblex/bardblex
1
LHHT −S/bardblex/bardblex/bardblex/bardblex
2
+ λ1
2
K−1∑
k=1
||W(k)||2 + λ2
2N||H−B||2
+ λ3
2
/bardblex/bardblex/bardblex/bardblex
1
NHTH−I /bardblex/bardblex/bardblex/bardblex
2
+ λ4
2N||HT 1||2
s.t. B∈{−1,1}N×L.
(14)
H isstackedbytheoutputsofnetworkand B isstackedbythebinarycodestobeoptimizedfrom
the Equation (14). S is the pairwise similarity matrix valued 1 or−1. The first term is similarity
differencelossminimization,thesecondtermisthe ℓ2 regularization,thethirdtermisthequantiza-
tionloss,andthelasttwotermsaretopunishthecorrelationandtheimbalanceofbits,respectively.
Note that theB is not the sign ofH. As a result, the loss function is optimized by updating the
networkparameterand B alternatively.Tobespecific, B isoptimizedwithafixedneuralnetwork,
whiletheneuralnetworkistrainedwithfixed B alternatively.SH-BDNNhasawell-designed loss
function, which follows Kernel-based Supervised Hashing [113]. However, the architecture does
not include the popular convolutional neural network, and it is not an end-to-end model. As a
result, the efficiency of this model is low in large-scale datasets.
Convolutional Neural Network Hashing(CNNH) [173]. CNNH is the earliest deep supervised
hashing framework to our knowledge. It adopts a two-step strategy. In the first step, it optimizes
the objective function using a coordinate descent strategy as follows:
LCNNH = /bardblex/bardblex/bardblex/bardblex
1
LHHT −S/bardblex/bardblex/bardblex/bardblex
2
, (15)
1In our survey,{λ1,λ2,λ3,... } always denote the balance coefficients.
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:13
whichgeneratesapproximatebinarycodes.Inthesecondstep,CNNHutilizesobtainedhashcodes
totraintheconvolutionalneuralnetworkwith Loutputunits.Besides,ifclasslabelsareavailable,
the fully connected layer withK output units is added, which correspond to theK class labels
of images and the classification loss is added to the loss function. Although CNNH uses labels in
a clumsy manner, this two-step strategy is still popular in deep supervised hashing and inspires
many other state-of-the-art methods.
DeepDiscreteSupervisedHashing (DDSH)[82].DDSHusesacolumn-samplingmannerforparti-
tioningthetrainingdatainto {xi}i∈Ωand{xi}i∈Γ,where ΩandΓaretheindexes.Thelossfunction
is designed in an asymmetric form:
LDDSH =
∑
i∈Ω,j∈Γ
L
(
so
ij −bT
i hj
)2
+
∑
i,j∈Ω
L
(
so
ij −bT
i bj
)2
, (16)
wherebi andhi arethebinarycodetobeoptimizedandtheoutputofthenetwork,respectively. bi
andhi are updated alternatively following [36]. It is notable because DDSH takes an asymmetric
strategy for learning to hash, which aids in both binary code generation and continuous feature
learning through the pairwise similarity structure.
Hashing with Binary Matrix Pursuit(HBMP) [9]. HBMP also takes advantage of the two-step
strategyintroducedabove.DifferentfromCNNH,HBMPutilizestheweightedHammingdistances
andadoptsadifferenttraditionalhashingalgorithmcalledbinarycodeinferencetogethashcodes.
In the first step, the objective function is written in the following equation
LHBMP = 1
4
∑
i,j
(
bT
i Λbj−so
ij
)2
, (17)
where Λ is a diagonal weight matrix. It is noticed that the similarity matrix with each element
Sh
ij = bT
i Λbj can be approximated by a step-wise algorithm. HBMP also trains a convolutional
neuralnetworkbytheobtainedhashcodeswithpoint-wisehingelossandshowsthatdeepneural
networks help to simplify the optimization problem and get robust hash codes.
AsymmetricDeepSupervisedHashing (ADSH)[84].ADSHconsidersthesamplesinthedatabase
and query set using an asymmetric manner, which can help to train the model more effectively,
especially for large-scale nearest neighbor search. ADSH contains two critical components, i.e., a
featurelearningpartandalossfunctionpart.Thefirstoneistoutilizeahashingnetworktolearn
discrete codes for queries. The second one is used to directly learn discrete codes for database
points by minimizing the same objective function with supervised information. The loss function
is formulated as
LADSH =
∑
i∈Ω,j∈Γ
(
hT
i bj−Lso
ij
)2
,
s.t. bj ∈{−1,1}L,
(18)
where Ωis the index of query points,Γis the index of database points. Network parameters
Θand binary codesbj are updated alternatively following SH-BDNN [36] during the optimiza-
tion process. If only the database points are available, we letΩ⊂Γand add a quantization loss∑
i∈Ω(bi−hi)2 with the coefficientγ. This asymmetric strategy combines deep hashing and tradi-
tional hashing, which can help achieve better performance.
Deep Incremental Hashing Network(DIHN) [172]. DIHN tries to learn hash codes in an incre-
mentalmanner.SimilartoADSH[ 84],thedatasetisdividedintotwoparts,i.e.,originalandincre-
mental databases, respectively. When a new image comes from an incremental database, its hash
codeislearnedwhilekeepingthehashcodesoftheoriginaldatabaseunchanged.Theoptimization
process still uses the strategy of alternately updating parameters.
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:14 X. Luo et al.
Deep Ordinal Hashing(DOH) [85]. DOH generates ordinal hash codes by taking advantage of
both local and global features. Specifically, two subnetworks learn the local semantics using a
spatial attention module-enhanced fully convolutional network and the global semantics using a
convolutional neural network, respectively. Afterward, the two outputs are combined to produce
R ordinal outputs{hr
i}R
r =1. For each segmenthr
i , the corresponding hash code can be obtained as
follows:
br
i = argmax
θ
θThi,
s.t.θ∈{0,1}L,∥θ∥1 = 1.
(19)
Thefullhashcodecanbeobtainedbyconcatenating {br
i}R
r =1.DOHadoptsanend-to-endranking-
to-hashingframework,whichavoidsusingtheundifferentiablesignfunction.Furthermore,ituses
a relatively complex network that is able to handle large datasets with higher performance.
3.2.2 Likelihood Loss Minimization. Deep Pairwise Supervised Hashing(DPSH) [101]. DPSH
adopts CNN-F [23] as the backbone of the hashing network and the standard form of likelihood
loss based on similarity information. Besides similarity information, quantization loss is also in-
troduced to the final loss function, i.e.,
LDPSH =−
∑
(i,j)∈E
(
so
ijsh
ij −log
(
1 +esh
ij
))
+λ1
N∑
i=1
||hi−sдn(hi)||2
2, (20)
wheresh
ij = 1
2hT
i hj andhi is the output of the hashing network. Although triplet loss was popular
atthattime,DPSHadoptsthepairwiseformtosimultaneouslylearndeepfeaturesandhashcodes,
which improves both accuracy and efficiency. This likelihood loss function can easily introduce
differentBayesianpriors,makingitflexibleinapplicationsandachievingbetterperformancethan
different loss functions.
DeepHashingNetwork (DHN)[200].IthasasimilarlikelihoodlossfunctiontoDPSH.Differently,
DHNconsidersthequantizationlossasBayesianpriorandproposesabimodalLaplacianpriorfor
the outputhi, i.e.,
p (hi) = 1
2ϵexp
(
−
/bardblex/bardblex/barex/barexhi /barex/barex−1/bardblex/bardblex1
ϵ
)
, (21)
and the negative log likelihood (i.e., quantization loss) is
LQuan =
N∑
i=1
|||hi−1||1, (22)
which can be smoothed by a smooth surrogate [74] into
LQuan =
N∑
i=1
L∑
l =1
loд(cosh(|hil|−1)), (23)
wherehik is thekth element ofhi. We notice that the DHN replacedℓ2 norm (ITQ quantization
error [48]) byℓ1 norm. [200] also shows that theℓ1 norm is an upper bound of theℓ2 norm, and
theℓ1 norm encourages sparsity and is easier to optimize.
HashNet [19]. As a variant of DHN, HashNet considers the imbalance training problem that
the positive pairs are much more than the negative pairs. Hence, it adoptsWeighted Maximum
LikeLihood (WML) loss with different weights for each image pair. The weight is formulated as
wij =cij ·
{ |S|/|S1|, so
ij = 1
|S|/|S0|, so
ij = 0, (24)
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:15
whereS1 ={(i,j)∈E:so
ij = 1}comprisessimilarimagepairs,while S0 =E/S1 comprisesdissim-
ilar image pairs.cij = y i∩y j
y i∪y j
for multi-label datasets and equals 1 for single-label datasets. Besides,
the sigmoid function in condition probability is substituted by 1/1 +e−αxcalled adaptive sigmoid
function, which equals to adding a hyper-parameter into the hash code similarity computation,
i.e.,sh
ij = αbT
i bj.Differentfromothermethods,HashNetcontinuouslyapproximatessignfunction
through the hyperbolic tangent function
lim
β→∞
tanh(βz) = sgn(z). (25)
The activation function for outputs is tanh(βt·) through updatingβt →∞step-wise and the
optimalnetworkwithsgn (·) canbederived.Besides,thisoperationcanbeillustratedusingmulti-
stage pretraining, which means that the deep network using activation functiontan(βt +1·) is ini-
tializedusingthewell-trainednetworkusingactivationfunction tan(βt·).Thetwoskillsproposed
by HashNet greatly increase the performance of deep supervised hashing.
Deep Priority Hashing(DPH) [20]. DPH also adds different weights to different image pairs,
but reduces the weights of pairs with higher confidence, which is similar to AdaBoost [138]. The
difficultyismeasuredby qij ,whichindicateshowdifficultapairisclassifiedassimilarwhen so
ij = 1
or classified as dissimilar whenso
ij = 0. In formulation,
q
(
so
ij|hi,hj
)
=
⎧⎪⎪⎨⎪⎪⎩
1+sh
ij
2 , so
ij = 1
1−sh
ij
2 , so
ij = 0
= /parenlefttpA
/parenleftbtA
1 +sh
ij
2
/parenrighttpA
/parenrightbtA
so
ij
/parenlefttpA
/parenleftbtA
1−sh
ij
2
/parenrighttpA
/parenrightbtA
1−sij o
.
(26)
Besides, the weight characterizing class imbalance is measured byαij :
αij =
⎧⎪⎪⎪⎪⎨⎪⎪⎪⎪⎩
|Si||Sj|√
|S1
i|/barex/barex/barexS1
j
/barex/barex/barex
,sij = 1
|Si||Sj|√
|S0
i|/barex/barex/barexS0
j
/barex/barex/barex
,sij = 0
, (27)
whereSi ={(i,j)∈E:∀j},a n d
S1
i =
{
(i,j)∈E:∀j,so
ij = 1
}
S0
i =
{
(i,j)∈E:∀j,so
ij = 0
}
.
(28)
The final priority weight is formulated as
wij = αij (1−qij )γ, (29)
whereγis a hyper-parameter. With the priority cross-entropy loss, DPH down-weighs confident
image pairs and prioritizes on difficult image pairs with low confidence. Similarly, priority quan-
tization loss changes the weight for different images to bew′
i = (1−qi)γand qi measures how
likelyacontinuousoutputcanbeperfectlyquantizedintoabinarycode.Inthisway,DPHachieved
better performance than HashNet.
Deep Supervised Discrete Hashing(DSDH) [100]. Besides leveraging the pairwise similarity in-
formation,DSDHalsotakesadvantageoflabelinformationbyaddingalinearregressionlosswith
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:16 X. Luo et al.
regularization to the loss function. By dropping the binary restrictions, the loss is formulated as
LDSDH =−
∑
(i,j)∈E
(
so
ijsh
ij −log
(
1 +esh
ij
))
+λ1
N∑
i=1
||hi−sдn(hi)||2
2 +λ2||yi−WTbi|| +λ3||W||F,
(30)
wheresh
ij = 1
2hT
i hj andthelabelisencodedinone-hotformat yi.ThesecondterminEquation( 30)
is the linear regression term and the last term is anℓ2 regularization.{hi}N
i=1,{bi}N
i=1,a n dW are
updated alternatively by using gradient descent method and discrete cyclic coordinate descend
method. DSDH greatly increases the performance of image retrieval since it takes advantage of
both label information and pairwise similarity information. It should be noted that in the linear
regressionterm,thebinarycodeisupdatedbydiscretecycliccoordinatedescend,sotheconstraint
of discreteness is met.
Deep Cauchy Hashing(DCH) [15]. DCH is a Bayesian learning framework similar to DHN, but
itreplacedthesigmoidfunctionwiththefunctionbasedonCauchydistributionintheconditional
probability. DCH aims at improving the search accuracy with Hamming distances smaller than
radius 2. Probability on the basis of generalized sigmoid function could be extremely large when
Hammingdistancesaregreaterthan2.ThiscouldbedetrimentaltocurrentHammingballretrieval.
DCH tackles this problem via incorporating the Cauchy distribution, since the probability drops
rapidly if Hamming distances are greater than 2. The Cauchy distribution is formulated as
σ
(
dh
ij
)
= γ
γ+dh
ij
, (31)
whereγisahyper-parameterand dh
ij ismeasuredbythenormalizedEuclideandistance,i.e., dh
ij =
d(hi,hj) = L
2 (1−cos(hi,hj). Besides, the prior is based on a variant of the Cauchy distribution,
i.e.,
P (hi) = γ
γ+d (/barex/barexhi /barex/barex, 1). (32)
The final loss function is formulated as the log-likelihood plus the quantization loss based on
the prior weight. However, this loss function will get almost the same hash code for images with
the same label. Even worse, the relationship for the dissimilar pairs is not considered.
Maximum-Margin Hamming Hashing (MMHH) [87]. In view of the shortcomings of DCH,
MMHH utilizes the t-Distribution and contains different objective functions for similar and dis-
similar pairs. The total loss is the weighted sum of two losses. Besides, a marginζis utilized to
avoid producing the exact same hash codes. The Cauchy distribution in DCH is replaced by
σ
(
dh
ij
)
=
⎧⎪⎪⎨⎪⎪⎩
1
1+max
(
0,dh
ij−ζ
),so
ij = 1
1
1+max
(
ζ,dh
ij
), so
ij = 0 (33)
The loss function is the weighted log-likelihood of conditional probability, i.e.,
LMMHH =
∑
(i,j)∈E
wij
(
so
ij
)
log
(
1 +max
(
0,dh
ij −ζ
))
+
∑
(i,j)∈E
wij
(
1−so
ij
)
log /parenlefttpA/parenleftexA
/parenleftbtA
1 + 1
max
(
ζ,dh
ij
) /parenrighttpA/parenrightexA
/parenrightbtA
+λ1
N∑
i=1
||hi−sдn(hi)||2
2
. (34)
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:17
The last term is a standard quantization loss. MMHH also proposed a semi-batch optimization
strategy to alleviate the imbalance problem. Specifically, the binary codes of the training data are
storedasextramemory.Thepairwiselossiscalculatedbythenewcodescomputedinthecurrent
epoch and their similar and dissimilar pairs are added into the memory bank for a new epoch. In
general, MMHH solves the shortcomings of DCH, which greatly improves search performance.
Deep Fisher Hashing(DFH) [103]. DFH points out that the pairwise loss minimization is similar
to Fisher’s Linear discriminant, which maximizes the gaps between inter-class examples whilst
minimizingthegapsbetweentheintra-classexamples.ItslogisticlossfunctionissimilartoMMHH
and the final loss function is formulated as
LDFH =
∑
(i,j)∈E
so
ij log
(
1 +edh
ij +ϵ
)
+
∑
(i,j)∈E
(
1−so
ij
)
log
(
1 +e−dh
ij +ϵ
)
+λ1
N∑
i=1
||hi−sдn(hi)||2
2, (35)
in whichϵis a margin parameter. Besides, the quantized center loss is added to the objective
function, which not only minimizes intra-class distances but also maximizes inter-class distances
between binary hash codes of each image.
Deep Asymmetric Pairwise Hashing(DAPH) [139]. Similar to ADSH, DAPH also adopted an
asymmetric strategy. The difference is that DAPH uses two networks with different parameters
for the database and queries. Besides, the bit independence, bit balance and quantization loss are
added to the loss function following SH-BDNN. The loss function is optimized by updating the
two neural networks alternatively.
Deep Attention-guided Hashing(DAgH) [182]. DAgH adopts a two-step framework similar to
CNNH, while it utilizes neural networks to learn hash codes in both two steps. In the first step,
the objective function is the combination of the log-likelihood loss and the difference loss with a
margin. In the second step, DAgH utilizes binary point-wise cross-entropy for optimization. Be-
sides,thebackboneofDAgHincludesafullyconvolutionalnetworkwithanattentionmodulefor
obtaining accurate deep features.
Deep Joint Semantic-Embedding Hashing(DSEH) [99]. DSEH is the first work to introduce Lab-
Net in deep supervised hashing. It also adopts a two-step framework with LabNet and ImgNet,
respectively. LabNet is a neural network designed to capture abundant semantic correlation with
image pairs, which can help to guide the hash code learning in the second step.fi denotes the
label embedding produced from one-hot labelyi. LabNet replaces the input from images to their
labelandlearnsthehashcodesfromlabelswithageneralhashinglossfunction.Inthesecondstep,
ImgNetutilizesanasymmetriclossbetweenthelabeledfeaturesinthefirststepandthenewlyob-
tained features from ImageNethj,i . e .sh
ij = fi
Thj along with the binary cross-entropyloss similar
to DAgH [182]. DSEH fully makes use of the label information from the perspectivesof both pair-
wiselossandcross-entropyloss,whichcanhelpgeneratediscriminativeandsimilarity-preserving
hash codes.
AsymmetricDeepSemanticQuantization (ADSQ)[181].ADSQincreasestheperformancebyuti-
lizingtwohashingnetworksandreducingthedifferencebetweenthecontinuousnetworkoutputs
and the desired hash codes, and the difference loss is also involved.
Deep Anchor Graph Hashing(DAGH) [26]. In the anchor graph, a minimal number of anchors
are used to link the whole dataset, allowing for implicit computation of the similarities between
distinct examples. At first, it samples a number of anchors and builds an anchor graph between
trainingsamplesandanchors.Then,thelossfunctioncanbedividedintotwoparts.Thefirstpart
containsatypicalpairwiselikelihoodlossandalinearregressionloss.Inthesecondpart,thelossis
calculatedbythedistancesbetweentrainingsamplesandanchorsinthesameclass,andbothdeep
featuresandbinarycodesareusedtocomputethedistances.Besidesageneralpairwiselikelihood
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:18 X. Luo et al.
lossandalinearregressionloss,DAGHminimizesthedistancesbetweendeepfeaturesoftraining
samples and binary codes of anchors belonging to the same class. This method fully utilizes the
remaininglabeleddataduringmini-batchtrainingandhelpstoobtainefficientbinaryhashcodes.
3.3 Ranking-based Methods
In this section, we will review the category of deep supervised hashing algorithms that use the
ranking to preserve the similarity structure. Specifically, these methods attempt to preserve the
similarity relationships for over two examples that are calculated in the original and Hamming
spaces. We further divide ranking-based methods into two groups:
—Triplet methods. Due to the ease with which triplet-based similarities could be obtained,
triplet ranking losses are popular in deep supervised hashing. These losses attempt to keep
therankingsconsistentintheHammingspaceandtheoriginalspaceforeachsampledtriplet.
Foreachtriplet (xi,xj,xk) withso
ij > so
ik, theyusuallyattemptto minimize adifference loss
with margin [92], i.e.,
LTriplet (hi,hj,hk) =max
(
0,m +dh
ij −dh
ik
)
, (36)
wherem is a margin parameter. Subsequent works introduce the weights based on ranking
for each triplet [197] or utilize the likelihood loss for preserving triplet ranking [169]. The
triplet loss can also be combined with the pairwise loss above [189].
—List-wisemethods.Thissub-classusuallyconsiderstherankingsinthewholedatasetrather
than in sampled triplet. An example is to optimize ranking-based metrics, i.e., Average Pre-
cision and Normalized Discounted Cumulative Gain [59]. Other works utilize the mutual
information [8] and matrix optimization [198] for optimizing the hash network from the
view of whole datasets. These methods can release the bias during triplet sampling but usu-
ally suffer from poor efficiency.
3.3.1 TripletMethods. DeepNeuralNetworkHashing (DNNH)[92].DNNHmodifiesthepopular
tripletrankingobjective[ 130]topreservetherelativerelationshipsofsamples.Tobemoreprecise,
given a triplet(xi,xj,xk) withso
ij > so
ik, the ranking loss with margin is formulated as
LDNNH (hi,hj,hk) =max
(
0,1 +dh
ij −dh
ik
)
. (37)
The loss encourages the binary codebj to be closer to thebi than bk. By substituting the Eu-
clideandistancefortheHammingdistance,thelossfunctionbecomesconvex,allowingforstraight-
forward optimization:
LDNNH (hi,hj,hk) =max(0,1 +||hi−hj||2
2−||hi−hk||2
2). (38)
Besides,DNNHintroducesasigmoidactivationfunctionalongwithapiece-wisethresholdfunc-
tion, which encourage the continuous outputs to approach discrete codes. The piece-wise thresh-
old function is defined as
д(s) =
⎧⎪⎪⎨⎪⎪⎩
0, s< 0.5−ϵ
s, 0.5−ϵ≤s≤0.5 +ϵ
1, s> 0.5 +ϵ
, (39)
inwhich ϵdenotesapositivehyper-parameter.Itisevidentthatmostelementsoftheoutputswill
beexact0or1byusingthispiece-wisethresholdfunction,thusresultinginlessquantizationloss.
DeepRegularizedSimilarityComparisonHashing (DRSCH)[189].Besidesthetripletloss,DRSCH
alsotookadvantageofpairwiseinformationbyintroducingadifferenceloss astheregularization
term. The bit weights are also included when calculating the distances in the Hamming space.
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:19
Deep Triplet Supervised Hashing(DTSH) [169]. DTSH replaces the ranking loss by the negative
log triplet label likelihood as
LDTSH (hi,hj,hk) =loд
(
1 +esh
ij−sh
ik−m
)
−
(
sh
ij −sh
ik −m
)
, (40)
which considers the conditional probability [100], andm is a margin parameter.
DeepSemanticRanking-basedHashing (DSRH) [197]. DSRHleverages a surrogatelossbasedon
triplet loss. Given queryq and database{xi}N
i=1, the rankings{ri}N
i=1 in database is defined as the
number of labels shared with the query. The ranking loss is defined in a triplet form
LDSRH =
N∑
i=1
∑
j:rj <ri
w(ri,rj)δmax
(
0,ϵ+dh
qi −dh
qj
)
, (41)
whereδandϵare two hyper-parameters, andw(ri,rj) is the weight for each triplet:
ω(ri,rj) = 2ri −2rj
Z . (42)
TheformofweightscomesfromNormalizedDiscountedCumulativeGains[ 78]score ,andZ is
anormalizationconstant,whichcanbeomitted.Besides,thebitbalancelossandweightregulariza-
tionare addedtothelossfunction.DSRHimproves deephashingbythesurrogateloss,especially
on multi-label image datasets.
3.3.2 Listwise Methods. Hashing as Tie-Aware Learning to Rank(HALR) [59]. HALR explicitly
optimizes popular ranking-based assessment metrics including average precision and normalized
discounted cumulative gain, which improves the retrieval performance based on ranking. It is no-
ticedthattiedranksmayoccurduetointeger-valuedHammingdistance.Hence,HALRintroduces
a tie-aware formulation of these metrics and trains the hashing network using their continuous
relaxations for effective optimization.
Hashing with Mutual Information(HashMI) [8]. HashMI follows the idea of minimizing neigh-
borhood ambiguity and derives a loss term based on mutual information, which is sufficiently
connected to the aforementioned ranking-based assessment metrics. Given an imagexi,t h er a n -
domvariableVi,Φisdefinedasamappingfrom xj todh
ij ,where Φisthehashingnetwork. Ci isthe
set of images that share the same label withxi, i.e., the neighbor ofxi. The mutual information is
defined as
IHashMI (Vi,Φ;Ci) = H(Ci)−H(Ci|Vi,Φ). (43)
Themutualinformationisincorporatedoverthedeepfeaturespaceforanyhashingnetwork Φ,
such that a measurement of the quality is obtained which desires to be maximized
O =−
∫
Ω
I(Vi,Φ;Ci)pidxi, (44)
where Ωis the sample space andpi denotes the prior distribution, which can be removed. After
discretion, the loss function turns into:
LHashMI =−
N∑
i=1
I(Vi,Φ;Ci), (45)
whose gradient can be calculated by relaxing the binary constraint and effective minibatch back
propagation. The minibatch back propagation is able to effectively retrieve one example against
the other example within a minibatch cyclically similar to leave-one-out validation.
Angular Deep Supervised Hashing(AnDSH) [198]. AnDSH calculates the Hamming distance be-
tweenimages ofdifferentclassestoforman uppertriangularmatrixwithsize K byK,wher eK is
the number of categorizations. The mean of Hamming distance matrices is maximized, while the
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:20 X. Luo et al.
varianceofthematricesisminimizedtomakesurethatallelementsinthematrixcouldbecovered
by these hash codes and from the view of bucket theory there is no weakness, i.e., achieving bit
balance. Besides, this method utilizes classification loss similar to PCDH but replaces the softmax
loss by A-softmax objective [115] that could obtain potentially larger inter-class variation along
with larger inter-class separation.
3.4 Pointwise Methods
In this section, we review pointwise methods that directly take advantage of label information
insteadofsimilarityinformation.Earlymethodsusuallyaddaclassificationlayertomapthehash-
like representations into label distributions [76, 106, 124, 180, 193–195]. Then, the hash codes are
enhancedwiththestandardclassificationlossinlabelspace.Furtherworksincludetheprobabilis-
tic models for better binary optimization [144]. Recent methods usually build the classification
lossintheHammingspaceinstead.Specifically,theywillgeneratesomecentralhashcodes, 2 each
of which is associated with a class label. These methods enforce the network outputs to approach
their corresponding hash centers with different loss terms, i.e., binary cross-entropy [185], differ-
ence loss [25], polarization loss [41], softmax loss [64], and partial softmax loss [158]. These hash
centersaremostlyproducedbyHadamardmatrix[ 25,64,185],randomsampling[ 64,185]asw ell
as adaptive optimization [158], which achieves better performance compare with the former two
manners.
Deep Binary Hashing(DBH) [106]. After pre-training of a convolution neural network on the
ImageNet, DBH adds a latent layer with sigmoid activation, where the neurons are utilized to
learnhash-likerepresentationswhilefine-tuningwithclassificationlossonthetargetdataset.The
outputs of the latent layer are discretized into binary hash codes. DBH also emphasizes that the
obtained hash codes are for coarse-level search because the quality of hash codes is limited.
SupervisedSemantics-preservingDeepHashing (SSDpH)[180].SSDpHutilizesasimilararchitec-
turetoDBHandaddsthequantizationlossandthebitbalancelossforregularization.Inthisway,
SSDpH can produce high-quality hash codes for better retrieval performance.
Very Deep Supervised Hashing(VDSH) [195]. VDSH builds a very deep hashing network and
trains the network with an efficient strategy layer-wise motivated byalternating direction
method of multipliers (ADMM)[5]. In virtue of the strong representation ability of the deeper
neural network, VDSH can produce better hash codes for effective image retrieval.
SUBIC [76]. SUBIC generates structured binary hash codes consisting of the concatenation of
severalone-hotencodedvectors(i.e.,blocks)andobtainseachone-hotencodedvectorwithseveral
softmax functions (i.e., block softmax). Besides classification loss and bit balance regularization,
SUBIC utilizes the mean entropy for quantization loss for each block. SUBIC can also be applied
to a range of downstream search tasks including instance retrieval and image classification.
JustMaximizingLikelihoodHashing (PMLR)[144].PMLRintegratestwodenselayersabovethe
topofthehashingnetwork.Itutilizestheprobabilitymodelstoparameterizethehashingnetwork
for binary constraints. Then, PMLR utilizes a classification loss along with a regularization term
for better hash code distributions in Hamming space.
Central Similarity Quantization(CSQ) [185]. CSQ also utilizes a classification model but in a
differentway.First,CSQgeneratessomecentralhashcodesbythepropertiesofaHadamardmatrix
or random sampling from Bernoulli distributions, such that the distance between each pair of
centroids is large enough. Each label is corresponding to a centroid in the Hamming space and
thus each image has its corresponding semantic hash center according to its label. Afterward, the
modelistrainedbythecentralsimilarityloss(i.e.,binarycross-entropy)withthesupervisedlabel
2They can be also called target codes.
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:21
information as well as the quantization loss. In formulation,
LCSQ =
N∑
i=1
L∑
l =1
[ci,l loghi,l + (1−ci,l)log(1−hi,l)] +λ1
N∑
i=1
(|||hi−1|−1||1), (46)
whereci ∈{0,1}L is the hash center generated from labels andhi ∈(0,1)L is the output of the
hashing network. It is evident that CSQ directly enforces the generated hash codes to approach
thecorrespondingcentroidswithsomerelaxations.ThecoreofCSQistomapthesemanticlabels
into Hamming space to guide hash code learning directly. Thus, samples with comparable labels
areconvertedintosimilarhashcodes,maintainingtheglobalsimilaritiesbetweenimagepairsand
then resulting in effective hash codes for image retrieval.
Hadamard Codebook-based Deep Hashing(HCDH) [25]. HCDH also utilizes the Hadamard ma-
trixbyminimizingthe ℓ2 differencebetweenhash-likeoutputsandthetargethashcodeswiththeir
correspondinglabels(i.e.,Hadamardloss).DifferentfromCSQ,HCDHtrainstheclassificationloss
andHadamardlosssimultaneously.Hadamardlosscanbeinterpretedaslearningthehashcenters
guided by their supervised labels inL2 norm. Note that HCDH is able to yield discriminative and
balanced binary codes for the property of the Hadamard codebook.
DeepPolarizedNetwork (DPN)[41].DPNcombinesmetriclearningframeworkwithlearningto
hash and develops a novel polarization loss which minimizes the distance between hash centers
and hashing network outputs. In formulation,
LDPN =
N∑
i=1
L∑
l =1
max(ϵ−hil ·cil,0), (47)
where ci ∈{0,1}L is the hash center andhi ∈(−1,1)L is the output of the hashing network.
Different from CSQ, the hash centers can be updated after a few epochs. It has been proved that
minimizing polarization loss can simultaneously minimize inter-class and maximize intra-class
Hamming distances theoretically. In this way, the hash codes can be easily derived for effective
image retrieval.
OrthHash [64]. OrthHash is a one-loss model that gets rid of the hassles of tuning the balance
coefficients of various losses. Similar to CSQ, OrthHash generates hash centers using Bernoulli
distributions. Then, it maximizes the cosine similarity between the hashing network outputs and
their corresponding hash centers. In formulation,
LOrthHash =−
N∑
i=1
log
exp
(
c⊤
i hi
)
∑
c∈Cexp(c⊤hi), (48)
whereC denotes the set of all hash centers. Compared with CSQ and DPN, OrthHash not only
compare the network outputs and corresponding hash centers, but also considers the other hash
centers of different labels. In this way, OrthHash improves the discriminativeness of hash codes.
Moreover, since Hamming distance is equivalent to cosine distance for hash codes, OrthHash can
promisequantizationerrorminimization.Withasingleclassificationobjective,itrealizestheend-
to-end training of deep hashing with promising performance.
Partial-Softmax Loss based Deep Hashing (PSLDH) [158]. PSLDH generates a semantic-
preservinghashcenterforeachlabelinsteadofusingHadamardmatrixorrandomsampling[ 185].
Specifically, it not also minimizes the inner product of each hash center pair, but also maximizes
theinformationofeachhashbitwithabitbalancelossterm.Moreover,PSLDHtrainsthehashing
network with a partial-softmax loss, which compares the network outputs with both their corre-
sponding hash centers and other centers of partial categories in the datasets. Letcj denote the
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:22 X. Luo et al.
hash center associated with thejth category. The loss is formulated as
LPSLDH =
N∑
i=1
∑
j∈Γi
−log
exp
(
η
(
hT
i cj−μL
))
exp
(
η
(
hT
i cj−μL
))
+ ∑
q∈Ψi exp
(
ηhT
i cq
), (49)
where Γi denotes the index set of categories associated withxi,a n dΨi denotes the index set of
categories unassociated withxi.
3.5 Quantization
The quantization techniques have been presented to be derivable from our aforementioned differ-
ence loss minimization in Section3.2 [165]. From a statistical standpoint, the quantization error
could bound the distance reconstruction error [79]. Consequently, quantization could be utilized
for deep supervised hashing. These methods usually leverage deep neural networks to generate
deepfeaturesandthenadoptproductquantizationapproachesforsubsequentquantization.Hence,
they optimize the deep features with pairwise difference loss [17], pairwise likelihood loss [39],
and triplet loss [110] for better retrieval performance. Further works combine label semantic in-
formation for discriminative deep features [16]. Recent works [39, 88] integrate deep neural net-
works into the process of product quantization rather than feature generation and achieve better
performance. Other than product quantization, composite quantization can also be enhanced by
deep learning [24]. Then, we will review the typical deep supervised hashing methods based on
quantization.
Deep Quantization Network(DQN) [17]. DQN generates hash codebi from the obtained rep-
resentation zi ∈RD with semantics preserved using the product quantization method. First, it
decomposesthefeaturespaceintothetargetspace,i.e.,aCartesianproductof M low-dimensional
subspaces,andeachsubspaceisquantizedinto T codewordsviaclustering.Moreprecisely,theorig-
inalfeatureispartitionedinto M sub-vectorsi.e., zi = [zi1;... ;ziM ],i = 1,..., N andzim ∈RD/M
is the sub-vector ofzi in themth subspace. Thus, all sub-vectors in each subspace are quantized
intoT codewords using K-means without mutual influences. The total loss is defined as follows:
LDQN =
∑
i,j
(
so
ij −cos(zi,zj)
)2
+λ1
M∑
m=1
N∑
i=1
∥zim−Cmbim∥2
2,
s.t.∥bim∥0 = 1,bim ∈{0,1}T,
(50)
wherecos(·)denotesthecosinesimilaritymetricand Cm = [cm1,..., cmT]represents T codewords
of themth subspace, andbim is the one-hot embedding to guide which codeword inCm should
be used to approach theith pointzim. Mathematically, the second term, i.e., product quantization
can be reformulated as
N∑
i=1
||zi−Cbi||2
2, (51)
whereC is aD×MT matrix can be written asC = diaд(C1,..., CM). Note that the quantization
loss of converting the featurezi into binary codebi can be restricted via minimizingQ. Besides,
quantization-basedhashingalsoaddspairwisesimilaritypreservinglosstothefinallossfunction.
Finally, Asymmetric Quantizer Distance (AQD) is widely used for approximate nearest neigh-
bor search, which is formulated as
AQD(q,xi) =
M∑
m=1
||zqm−Cmbim||2
2, (52)
wherezqm is themth sub-vector for the feature of queryq.
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:23
Deep Triplet Quantization(DTQ) [110]. DTQ uses a triplet loss to preserve the similarity infor-
mationandasmoothorthogonalityregularizationisaddedtothecodebooks,whicharesimilarto
the bit independence. LetT denote the set of all triplet. Each triplet(xi,xj,xk) satisfiesso
ij > so
ik.
The total loss function is as follows:
LDTQ =
∑
(xi,xj,xk)∈T
(max(0,ϵ+||zi−zj||2
2−||zi−zk||2
2) +λ1
M∑
m=1
N∑
i=1
∥zim−Cmbim∥2
2
+λ2
M∑
m=1
M∑
m′=1
||CT
mCm′ −I||2.
(53)
Thelasttermistheorthogonalitypenaltyterm.Inaddition,DTQselectstripletsbyGroupHard
to make sure that the number of explored valid triplets is suitable for optimization. Specifically,
the training data are split into various groups, and a hard (i.e., with positive triplet loss) negative
example is picked randomly as an anchor-positive image pair from every group.
DeepVisual-semanticQuantization (DVsQ)[16].DVsQoptimizesthequantizationnetworkusing
labeled image samples along with the semantic messages from their latent text domains. Specifi-
cally,byusingtheimagerepresentations zi fromthepre-trainednetwork,itproducesdeepvisual-
semantic representations. They are then trained to forecast the word embeddingsv (i.e., vi for
labeli),whicharefurtherestimatedbyaskip-grammodel.Thelossfunctionincludestheadaptive
margin ranking loss and a quantization loss:
LDVsQ =
N∑
i=1
∑
j∈y i
∑
k/nelementy i
max(0,δjk −cos(vj,zi) +cos(vk,zi)) +λ1
N∑
i=1
|y|∑
j=1
||vT
j (zi−Cbi)||2
2, (54)
whereyi is the label set of theith image, andδjk is an adaptive margin and the quantization loss
is inspired by the maximum inner-product search. DVsQ adopts the same strategy as LabNet and
combines the visual information and semantic quantization in a uniform framework instead of a
two-step approach. By this means, DVsQ greatly improves the retrieval performance.
Deep Product Quantization(DPQ) [88]. DPQ leverages both the powerful capacity ofproduct
quantization(PQ)andtheend-to-endlearningabilityofdeeplearningtooptimizetheclustering
results of product quantization through classification tasks. Specifically, for each inputxi,i tfi r s t
uses an embedding layer and an MLP to obtain the deep representationzi ∈RMF. Then, the
representation is sliced intoM sub-vectors withzi,m ∈RF similar to PQ. Different from DQN, an
MLPisusedtoturneachsub-vectorintoaprobabilisticvectorwith T elementspm(t),t = 1,..., T
bysoftmaxfunction.Thematrix Cm ∈RT×D denotesthe T centroids.pm(k)denotestheprobability
that themth sub-vectoris quantizedbythe tth row ofCm. The soft representationofthemth sub-
vector is calculated by combining the row vectors ofCm.
soft m =
T∑
t =1
pm(t)Cm(t). (55)
Consideringtheprobability pm(k) inone-hotformat,given t∗=arдmaxtpm(t),thehardproba-
bility is denoted asem(t) =δtt∗in one-hot format and we have
hardm =
T∑
t =1
em(t)Cm(t). (56)
The obtained sub-vectors of soft and hard representations are then concatenated to produce
the ultimate representations, i.e., soft= [soft1,..., softM]a n dh a r d= [hard1,..., hardM]∈RMD .
Eachrepresentationisfollowedbyafully-connectedclassificationlayer.Besidestwoclassification
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:24 X. Luo et al.
losses,thejointcentrallossisalsoaddedbyfirstlearningthecentervectorforeachcategorization
and minimizing the distances between deep features. It is noticed that both the soft and hard
representations come from the same centers in DPQ, which encourages both representations to
approachthecenters,reducingthedisparitybetweenthesoftandhardrepresentations.Thishelps
toimprovethediscriminativepowerofthefeaturesandtocontributetotheretrievalperformance.
Gini batch loss and Gini sample loss are also introduced for the class balance and encourage the
two representations of the same image to be closer. Overall, DPQ replaces the k-means process
in PQ and DQN technique with deep learning combined with a classification model and is able to
create compressed representations for fast classification and fast image retrieval.
Deep Spherical Quantization(DSQ) [39]. DSQ first uses the deep neural network to obtain the
ℓ2 normalized features and then quantizes these features on a unit hypersphere with an elaborate
quantization manner. After constraining the continuous representations to staying on a unit hy-
persphere,DSQattemptstoreducethereconstructionlossusing multi-codebook quantization
(MCQ). Different from PQ, MCQ draws near the representation vectors with the summation of
multiple codewords instead of the concatenation.ˆyi denotes the predicted label distribution.ϕyi
denotes the feature center of theyith class. The overall loss for training the model is as follows:
LDSQ =
N∑
i=1
−logyT
i loдˆyi +λ1
N∑
i=1
||zi−[C1,..., CM]bi||2
2
+λ2
N∑
i=1
||zi−ϕyi||2
2 +λ3
N∑
i=1
||ϕyi −Cbi||2
2
s.t.||bim||0 = 1,bi ∈{0,1}K,bi = [bT
i,1,..., bT
i,M]T,
(57)
where the first, second, third and last term is the softmax loss, quantization loss, the center loss
andthediscriminativeloss,respectively.Thelasttwolossesencourageboththequantizedvectors
and deep features to approach their centers, respectively.
SimilarityPreservingDeepAsymmetricQuantization (DPDAQ)[24].DPDAQadoptsAsymmetric
QuantizerDistancetoapproachthedesiredsimilaritymetric,whichissimilartoADSH.Moreover,
itusescompositequantizationinsteadofproductquantizationandtherepresentationsinthetrain-
ing set come from the deep neural network in an unquantized form. SPDAQ also takes advantage
of similarity information and label information to achieve better retrieval performance.
3.6 Other Techniques for Deep Hashing
3.6.1 Hashing with Generative Adversarial Networks. Generative Adversarial Networks
(GANs)[ 49] are popular neural network models to generate virtual examples without needing
supervised knowledge. There are also several hashing methods leveraging GANs to enhance the
performance.
Deep Semantic Hashing with GAN(DSH-GAN) [133]. DSH-GAN is the first hashing method
that takes advantage of GANs for image retrieval. It typically includes four components, i.e., a
neural network to produce image representations, an adversarial discriminator for differentiating
between synthetic images and real images, a hashing network for projecting representations into
binary codes and a classification head. Specifically, the generator network attempts to generate
synthetic images after concatenating the label embedding as well as generated noise embedding.
The discriminator attempts to jointly differentiate between real samples and synthetic ones and
categorizetheinputsintopropersemanticlabels.Finally,theoverallframeworkisoptimizedusing
theadversariallosstomixtwosourcesandtheclassificationlosstoobtainthegroundtruthlabels
using a classic minimax mechanism. The input of the network is image triplets, each of which
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:25
containsthreeimages.Thefirstoneisarealimagetreatedasaquery,thesecondoneisasynthetic
image created with the same label as the query image by the generator network, and the third
one is a synthetic image with different semantics. GAN provides a hashing model with strong
generalization potential from the maintaining of semantics and similarity, which improves the
quality of hash codes.
HashGAN [14]. HashGAN augments the training data with images synthesized by pair con-
ditional Wasserstein GAN (WGAN) inspired by [54], which sufficiently explores the pairwise
semantic relationships. In this module, the training samples along with the pairwise similarities
are considered as inputs and a generator and a discriminator is trained simultaneously by adding
the pairwise similarity besides the loss function of WGAN. The hash encoder produces high-
qualitybinarycodesforalloccurredpicturesusingalikelihoodobjectivesimilartoHashNet.Hash-
GAN is also capable of coping with the dataset without class labels but with pairwise similarity
information.
3.6.2 EnsembleLearning. Guoetal.[ 55]pointoutthatforthecurrentdeepsupervisedhashing
model, simply increasing the length of the hash code with a single hashing model cannot signifi-
cantlyenhancetheperformance.Thepotentialcauseisthatthelossfunctionsadoptedbyexisting
methods are prone to produce highly correlated and redundant hash codes. Inspired by this, sev-
eral methods attempt to leverage ensemble learning to increase the retrieval performance with
more hash bits.
Ensemble-basedDeepSupervisedHashing (EbDSH)[55].EbDSHleveragesanensemblelearning
strategy for better retrieval performance. Specifically, it trains a number of deep hashing models
withdifferenttrainingdatasets,trainingdata,initialization,andnetworks,thenconcatenatesthem
intothefinalhashcodes.Itisnoticedthattheensemblestrategyissuitableforparallelizationand
incremental learning.
WeightedMulti-deepRankingSupervisedhashing (WMRSH)[98].WMRSHattemptstogenerate
a high-quality hash function using multiple hash tables derived from the hashing networks. To
be specific, WMRSH adds bit-wise weights and table-wise weights for each bit in each hash table.
For each bit in a table, the similarity preservation is measured by product loss. Afterward, the bit
independence is measured by the correlation between two bits. Finally, the table-wise weight can
be derived from the mean average precision for every hash table. The final weight is the product
of the three above terms for the final hash codes (i.e., the concatenation of the hash tables with
weights). A similar strategy called Hash Boosting has been introduced in [117].
Apart from these methods, NMLayer [43] balances the importance of each bit and merges the
redundant bits together to learn more compact hash codes.
3.6.3 Training Strategy for Deep Hashing.In this subsection, we will introduce two methods
that adopt different training strategies from most other methods.
Greedy Hash[156]. Greedy Hash adopts a greedy algorithm for fast processing of hashing dis-
crete optimization by introducing a hash layer with a sign function instead of the quantization
error. To overcome the ill-posed gradient problem [177], the gradients are transmitted entirely to
thefrontlayer,whicheffectivelypreventsthevanishinggradientsofthesignfunctionandupdates
all bits together. This strategy is also adopted in recent works [134].
Gradient Attention deep Hashing(GAH) [72]. This work points out a dilemma in learning deep
hashingmodelsthroughgradientdescentthatitmakesnodifferencetothelossifthepairedhash
codeschangetheirsignstogether.Asaresult,GAHgeneratesattentiononthederivativesofeach
hashbitforeachimagebymaximizingthedecreaseoflossduringoptimization.Itleveragesagra-
dient attentionnetworkwith twofully-connectedlayersto producenormalizedweights andthen
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:26 X. Luo et al.
applies them to the derivatives in the last layer. In conclusion, this model optimizes the training
process by adopting a gradient attention network for acceleration.
4 DEEP UNSUPERVISED HASHING
4.1 Overview
Recently, unsupervised hashing methods have received widespread attention due to their suffi-
cient leverage of the unlabeled data, which facilities the practical applications in the real world.
Since deep unsupervised methods can not acquire label information, the semantic information is
obtained in deep feature space with pre-trained networks. With semantic information, the prob-
lem can be converted into a supervised problem. However, how to infer semantics information
and how to utilize semantics information for learning hash codes are two key problems here. Ac-
cording to semantics learning manners, the unsupervised methods can be mainly classified into
three categories, i.e., similarity reconstruction-based methods, pseudo-label-based methods, and
prediction-free self-supervised learning-based methods. Similarity reconstruction-based methods
usually generate pairwise semantic information, and then leverage pairwise semantic preserving
techniques in Section3.2 for hash code learning. Pseudo-label-based methods usually produce
pointwisepseudo-labelsforinputsandthenleveragepointwisesemanticpreservingtechniquesin
Section 3.4for hash code learning. Lastly, prediction-free self-supervised learning-based methods
leveragedataitselffortrainingwithoutgeneratingexplicitsemanticinformation,i.e.,similaritysig-
nalsandpseudo-labels.Specifically,theyusuallyutilizeregularizationterms,auto-encodermodels,
generative models, and contrastive learning to produce high-quality hash codes. The regulariza-
tiontermsincludebitbalancelossterm,bitindependencelosstermandatransformation-invariant
regularization term. Several approaches may combine different kinds of semantics learning man-
ners.Theoptimizationofbinarizationisstillanimportantproblemfordeepunsupervisedhashing.
Most of the methods usetanh(·) to approximatesiдn(·) and generate approximate hash codes by
thehashingnetworkforoptimization.ThesummaryofthesealgorithmsisshowninTable 3.Then,
we elaborate on these classes as below.
4.2 Similarity Reconstruction-based Methods
Similarityreconstruction-basedmethodsaimatleveragingpairwisemethodstosolvetheproblem.
However, the similarity information is unavailable without label annotation. Hence, these meth-
ods utilize a two-step framework as shown in Figure3. Firstly, they extract deep representations
zi using the pre-trained neural network and then infer the similarity information{so
ij}(i,j)∈Eby
distancemetricsindeepfeaturespace.Secondly,ahashingnetworkistrainedtocreatesimilarity-
preserving binary codes by leveraging the reconstructed similarity structure as guidance. With
similarity information, the problem can be solved with pairwise supervised methods. The key to
this kind of methods is how to generate accurate similarity information. Early methods usually
truncate pairwise distances in deep feature space [177]. Further studies utilizes the neighbour-
hoodinformation[ 120,122,179],confidencedegree[ 122],andothersimilaritymatrices[ 121,159]
to obtain a precise similarity structure for reliable guidance of subsequent optimization. Recently,
a few researchers argue that static similarity structure from the pre-trained network is not opti-
mal and propose to update it based on obtained hash codes [108, 141, 145] .N e x t ,w er e v i s et h e s e
methods in detail.
Semantic Structure-based Unsupervised Deep Hashing(SSDH) [177]. SSDH is the first study
along this line, which applies VGG-F model to extract deep features and perform hash code
learning. It studies the cosine distance for each pair in deep feature space, and finds that the
distribution of cosine distances can be approximated by two half Gaussian distributions. Hence,
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:27
Table 3. A Summary of Deep Unsupervised Hashing Approaches w.r.t the Manner of Generating
Similarity Information, Generating and Handling the Pseudo-Label, Binarization, as well as Other Skills
Approach Similarity Information Pseudo-Label Binarization Other skills
SSDH [177] Local Dist. - Tanh -
DistillHash [177] Local Dist. + Neighbour Information - Tanh -
SADH [141] Network Output + Adjacent Matrix - Drop + Alternation -
MLS3RUDH [159] Local Dist. + Manifold Dist. - Tanh -
TBH [145] Hash Codes - Bottleneck Reg. AE
GLC [120] Local Dist. + K-Means - Tanh -
MBE [104] Local Dist. - - Bit Bal. with Bi-Half Layer
CIMON [122] Local Dist. + Spectral Clu. + Conf. - Tanh Contrastive Learning
DATE [121] Local Dist. + Distribution Dist. + Conf. - Tanh Contrastive Learning
PLUDDH [67] - Kmeans + Cla. Layer Tanh + Quan. Loss -
DAVR [69] - Deep Clu. + Triplet Loss Drop -
CUDH [51] - Deep Embedding Clu. Tanh + Quan. Loss -
DVB[ 143] Adjacent Matrix Clu. Quan. Loss VAE + Bit Indep.
UDHPL [186] - Kmeans + PCA + MI - -
DU3H [191] Local Dist. + Conf. Kmeans + Hash Center Tanh GCN
UDMSH [132] Local Dist. + Conf. - Quan. Loss -
DSAH [108] Updated Local Dist. + Conf. - Quan. Loss -
UDKH [37] - Hash Code + Deep Clu. Alternative -
BDNN [36] - - Drop AE
DH [40] - - Quan. Loss Bit Bal. + Ind.
DeepBit [105] - - Quan. Loss Bit Bal. + Trans. Reg.
UTH[ 105] - - Quan. Loss Bit Bal. + Triplet Trans. Reg.
BGAN [152] Local Dist. - Tanh + Continuation GAN
BinGAN [201] - - Quan. Loss GAN + Bit independence
HashGAN [45] - - Quan. Loss Bit Bal. + Ind. + Trans. Reg. + GAN
SGH [31] - - - VAE
CIBHash [134] - - Drop Contrastive Learning
SPQ [77] - - Quantization Cross Contrastive Learning
HashSIM [119] Local Dist. - Tanh Bit Contrastive Learning
Drop = Drop the sign operator in the neural network and treat the binary code as an approximation of the network
output, Reg. = Regression, Quan. = Quantization, Dist. = Distance, Conf. = Confidence, Trans. = Transformation, Ind. =
Independence, Bal. = Balance., and Cla. = Classification, Clu. = Clustering.
through parameter estimation, SSDH sets two distance thresholddl and dr and construct a
similarity structure as follows:
so
ij =
⎧⎪⎪⎨⎪⎪⎩
1, if d(zi,zj)≤dl
0, if dl < d(zi,zj)< dr
−1, if d(zi,zj)≥dr
, (58)
where d(·,·) denotes the cosine distance of two vectors. From Equation (58), SSDH considers
sample pairs with distance smaller than dl as semantically similar while considers sample
pairs with distances large thandr as semantically dissimilar. Similar to SH-BDNN, a similarity
difference loss is adopted as follows:
LSSDH =
N∑
i=1
N∑
j=1
/barex/barex/barexso
ij
/barex/barex/barex
(
sh
ij −so
ij
)2
, (59)
wheresh
ij = hT
i hj/L,hi denotes the output of the deep network with activation functiontanh(·).
Theactivationfunction siдn(·) isutilizedinsteadduringevaluation.However,theperformanceof
SSDHislimitedduetotwoissues.Ononehand,itssimilaritystructureistypicallyunreliableusing
two coarse thresholds. On the other hand, it discards a range of signals in similarity structure.
DistillHash[179].DistillHashleveragesthesimilaritysignalsfromlocalstructurestodistillsim-
ilarity signals. Specifically, for each pair of images, it studies the similarities of their neighbors
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:28 X. Luo et al.
Fig. 3. Basic Framework of Deep Unsupervised Hashing with Similarity Reconstruction. The deep features
are extracted by a pre-trained network for similarity reconstruction. More details will be discussed in
Section 4.2.
and then removes the similarity signal if it has huge variants in local structures. The distillation
processcanbeimplementedwithBayesoptimalclassifier.Finally,DistillHashleverageslikelihood
loss minimization to train the hashing network with the similarity structure
LDistillHash =−
N∑
i=1
N∑
j=1
(
1so
ij =1σ
(
sb
ij
)
+ 1so
ij =−1
(
1−σ
(
sb
ij
)))
. (60)
The improvement of DistillHash over SSDH is mainly the introduction of local structures to
distill confident signals, which releases the first issue in the last paragraph.
SimilarityAdaptiveDeepHashing (SADH)[141].SADHtrainsthemodelalternativelyoverthree
parts. In part one, it trains the hashing network under the guidance of binary codes. In part two,
it leverages the network output to update the similarity structure. In part three, the hash codes
are optimized with network output following the ADMM process. The alternative optimization
improves the robustness of the model and helps achieve better hash codes for image retrieval.
Deep Unsupervised Hashing via Manifold based Local Semantic Similarity Structure Reconstruct-
ing (MLS3RUDH) [159]. MLS3RUDH incorporates the manifold structure in deep feature space to
generate an accurate similarity structure. Specifically, it leverages a random walk on the nearest
neighbor graph to measure the manifold similarity. The final similarity structure is denoted as
follows:
so
ij =
⎧⎪⎪⎨⎪⎪⎩
1, xj ∈Nc (xi)∧xj ∈Nm (xi)
−1, xj ∈Nc (xi)∧xj /nelementNm (xi)
0, otherwise
, (61)
whereNc(·) andNm(·) denotethesetoftheneighboursamplesintermsofbothcosinesimilarity
and manifold similarity, respectively. Then, the hashing network is optimized through difference
loss minimization as
LMLS 3RUDH =
N∑
i=1
N∑
j=1
log
(
cosh
(
sh
ij −so
ij
))
. (62)
MLS3RUDH leverages the manifold similarity to generate a more accurate similarity structure,
which guides the optimization of the hashing network effectively.
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:29
Auto-Encoding Twin-Bottleneck Hashing(TBH) [145]. TBH introduces an adaptive code-driven
graph to guide hash code learning. It contains a binary bottleneck to construct code-driven
similarity graph and a continuous bottleneck for reconstruction. To be specific, the similarity
structure is defined by hash codes
so
ij = 1−dh
ij/L, (63)
wheredij is the Hamming distance betweenbi andbj. The outputs of the continuous bottleneck
are fed into graph neural networks with the similarity structure as the adjacency for the final
reconstruction. Moreover, TBH involves adversarial learning to regularize the network for high-
qualityhashcodes.TBHutilizesadynamicgraphguidedwiththereconstructionlossforaccurate
similarity structures, which helps hash code preserve better similarity for reliable retrieval.
DeepUnsupervisedHashingbyGlobalandLocalConsistency (GLC)[120].GLCextractssemantic
informationfrombothlocalandglobalviews.Forlocalviews,itbuildsreliablegraphsandpenalty
graphs based on the cosine distances of image pairs. For global views, it utilizes global clustering
to derive cluster centers for different classes. During the optimization of hashing network, GLC
preservesthelocalsimilarityusingaproductlossandminimizestheHammingdistancesbetween
thehashcodesinthesamecluster.Comparedwithpreviousmethods,GLCpreservesthesimilarity
from different views in a unified manner, resulting in effective retrieval performance.
CIMON [122]. CIMON first sets a thresholddt to partition the local similarity signals based
on cosine metric into two groups. Inspired by the fact that the representations of samples with
the similar semantic information ought to be on a high-dimensionalmanifold, CIMONadopts the
resultsofspectralclusteringtoremovecontradictoryresultsforrefiningthesemanticsimilarities.
Moreover,itconstructstheconfidenceofthesimilaritysignals.Thesemanticinformationincludes
similarity signals{so
ij}(i,j)∈Eand their confidence{wij}(i,j)∈E. In formulation,
so
ij =
⎧⎪⎪⎪⎪⎨⎪⎪⎪⎪⎩
1 ci =cj&d(zi,zj)< dt
−1 ci /nequalcj&d(zi,zj)< dt
0 otherwise
, (64)
where{ci}N
i=1 is the cluster label of clustering. The confidence is built based on the cumulative
distribution function
wij =
⎧⎪⎪⎪⎪⎪⎨⎪⎪⎪⎪⎪⎩
Φ1(dt )−Φ1(d(zi,zj))
Φ1(dt )−Φ1(0) d(zi,zj)≤dt&so
ij /nequal0
Φ2(d(zi,zj))−Φ2(dt )
Φ2(2)−Φ2(dt ) dt < d(zi,zj)&so
ij /nequal0
0 ˆSij = 0
, (65)
whereΦ·(·) iscumulativedistributionfunctionofestimatedGaussiandistribution.CIMONgener-
ates two groups of semantic information by data augmentation and matches the hash code simi-
larity with similarity information in a parallel and cross manner. Moreover, contrastive learning
is also introduced to improve the quality of hash codes. To our knowledge, CIMON is the first
method using contrastive learning for hash code learning and achieves impressive performance
due to both reliable similarity information and contrastive learning.
Maximizing Bit Entropy(MBE) [104]. MBE utilizes the continuous cosine similarity signals to
guide hash code learning. More importantly, it introduces a bi-half layer for better quantization.
Specifically, for the continuous network outputs, MBE sorts the elements of each dimension over
alltheminibatchsamples,andthenassignsthetophalfelementsto1andtheremainingelements
to−1. In this manner, MBE can achieve absolute bit balance. The optimization of the bi-half layer
is based on a straight-through estimator similar to [156].
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:30 X. Luo et al.
DATE[121].DATEcharacterizeseachimagebyasetofitsaugmentedviews,whichcanbecon-
sideredasexamplesfromitslatentdistributions.Then,itcalculatesthesemanticdistancesbetween
sample pairs by computing the distribution divergence using a non-parametric way. Specifically,
we define the smoothed ball divergence statistic written as
BD
({zr
i
}R
r =1,
{
zr
j
}R
m=1
)
= 1
R
R∑
r =1
/parenlefttpA/parenleftexA
/parenleftbtA
/parenlefttpA
/parenleftbtA
1
R
M∑
r =1
d
(
zm
i ,zr
j
)
−d
(
zm
i ,zr
i
)/parenrighttpA
/parenrightbtA
2
+ /parenlefttpA
/parenleftbtA
1
R
M∑
r =1
d
(
zm
j ,zr
j
)
−d
(
zm
j ,zr
i
)/parenrighttpA
/parenrightbtA
2
/parenrighttpA/parenrightexA
/parenrightbtA
,
(66)
where{zr
i}R
r =1 and{zr
j}R
r =1 denote the features of augmented views of imagesxi andxj through a
pre-trainednetwork.Then,thedistributiondistanceiscombinedwithcosinedistancetogenerate
reliable semantic information. Contrastive learning is also utilized for high-quality hash codes.
Throughaccuratesemanticinformationenhancedbyaugmentations,DATEcanachievepromising
performance for image retrieval.
4.3 Pseudo-label-based Methods
The second class of deep unsupervised methods generates pseudo-labels. These methods treat
pseudo-labels as semantic information and convert this problem into supervised hashing. Most
of them first leverage clustering (e.g., K-means and spectral clustering) to generate pseudo-
labels [67,69,143,186,191]. Then, these pseudo-labels guide hash code learning with deep super-
visedhashingmethods.Furtherstudiesutilizeadeepclusteringframeworktocombineclustering
with the hashing network to adaptively update pseudo-labels [37, 51].
PseudoLabel-basedUnsupervisedDeepDiscriminativeHashing (PLUDDH)[67].PLUDDHutilizes
the pre-trained network to extract deep features and then generates pseudo-labels via clustering.
Thenthehashingnetworkissupervisedbypseudo-labels.Ithasthesameneuralnetworkarchitec-
ture as DBN and trains it with the classification loss and the quantization loss. PLUDDH explores
deep feature space with coarse clustering, which may generate false pseudo-labels. Hence, its re-
trieval performance is limited when the dataset is complicated.
Unsupervised Learning of Discriminative Attributes and Visual Representations (DAVR) [69].
DAVR adopts a two-step framework. In the first stage, a CNN is trained coupled with unsuper-
viseddiscriminativeclustering[ 150]togeneratetheclustermembership.Inthesecondstage,clus-
termembershipisutilizedassupervisiontouncovercommonclusterpropertieswhileoptimizing
their separability using a triplet objective. In general, the unsupervised hashing is converted into
a supervised problem by the obtained pseudo labels.
UnsupervisedDeepHashingwithPseudoLabels (UDHPL)[186].UDHPLfirstextractsfeaturesand
reduces their dimension withPrinciple Component Analysis (PCA) to release the noise. Then
itgeneratesthepseudo-labelsthroughtheBayes’rule.UDHPLmaximizesthecorrelationbetween
the projection vectors of pseudo-labels and deep features, and the features can be projected into
theHammingspace.Witharotationmatrix,thehashcodecanbegenerated,whichwillguidethe
optimizationofthehashingnetwork.UDHPLimprovesthepseudo-labelsthroughPCAandguides
the network training with mutual information maximization, which helps to preserve similarity
information for effective retrieval.
Clustering-driven Unsupervised Deep Hashing(CUDH) [51]. CUDH first extracts deep features
from the pre-trained network. Inspired by the deep clustering model DEC [174], which performs
clusteringintheembeddingspace,itmodifiesthemodeltoiterativelylearndiscriminativeclusters
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:31
intheHammingspacewithextraquantizationloss.CUDHiscapableofgeneratingdiscriminative
hash code in virtue of the deep clustering model.
Deep Unsupervised Hybrid-similarity Hadamard Hashing(DU3H) [191]. DU3H first generates
pseudo-labels through K-means clustering. Instead of adding a classification layer, DU3H uti-
lizes Hadamard matrix to project pseudo-labels into Hamming space. This strategy is similar to
CSQ[185]butinunsupervisedscenarios.Moreover,itgeneratesasimilaritystructureforpreserv-
ing pairwise similarity, which considers the confidence of different signals. This consideration of
different confidence can also be seen in UDMSH [132] and DSAH [108]. Lastly, a two-layer GCN
isintroducedtoamplifythediscrepancyofsimilaritysignalstofurtherguidethehashcodelearn-
ing.DU3Hcombinespointwisemethodsandpairwisemethodsinrecentdeepsupervisedlearning
domains, which helps to achieve significant improvement.
UnsupervisedDeepK-meansHashing (UDKH)[37].UDKHisajointframeworkwhichcombines
deep clustering with traditional clustering, i.e., K-means. It first uses K-means clustering results
to initialize the cluster labels. UDKH learns both hash codes and cluster labels in an alternative
manner. Specifically, it first fixes clustering results and optimizes the hash codes as well as the
hashingnetworkundersupervision.Then,itfixesthehashcodesandleveragesDiscreteProximal
LinearizedMinimization[ 142]toderivetheupdatedpseudo-labels.UDKHrepeatstheabovesteps
until convergence. UDKH improves the quality of hash codes along with the pseudo-labels with
progressive learning, achieving better performance compared with unsupervised methods using
fixed pseudo-labels.
4.4 Prediction-Free Self-Supervised Learning-based Methods
The last class of deep unsupervised methods is prediction-free self-supervised learning-based
methods. The early methods often impose several constraints on hash codes by minimizing
regularization terms (i.e., the bit balance loss, the bit independence loss the quantization loss, and
transformation-invariant loss) [40, 105]. To extract more information through deep neural net-
works, several researchers introduce popular self-supervised techniques into deep unsupervised
hashing, such as auto-encoder [31, 36, 145] and generative adversarial network [45, 152, 201],
and so on. Recently, contrastive learning has shown promising performance in producing
discriminativerepresentationsinvariousdomains.Inspiredbythefactthathashcodeisaspecific
form of representation, several methods involve contrastive learning into recent unsupervised
hashing, which helps to get high-quality hash codes [77, 121, 122, 134]. Following the scheme
of contrastive learning in [60], these methods usually first transform each inputxi into two
views x(1)
i and x(2)
i . Then, the hashing network projects them into two hash codesb(1)
i and b(2)
i .
Given theα⋆βdenotes the cosine similarity of two vectorsαand β, the network is trained by
minimizing the loss for each batch as follows:
LCL =−1
2NB
NB∑
i=1
/parenlefttpA
/parenleftbtA
log eb(1)
i ⋆b(2)
i /τ
Z(1)
i
+log eb(1)
i ⋆b(2)
i /τ
Z(2)
i
/parenrighttpA
/parenrightbtA
, (67)
whereτisatemperatureparameter, NB isthebatchsize,and Z(r)
i = ∑
j/nequali(eb(r)⋆b(1)
j /τ+eb(r)
i ⋆b(2)
j /τ),
r = 1 or 2. This term can also be illustrated using mutual information [134]. Minimizing Equa-
tion (67) has three potential benefits. First, since the numerator penalizes the difference in binary
codes of samples under different views, it assists in the production of transformation-invariant
binary codes. Second, since the denominator promotes to amply the distances between binary
codes of different examples which facilities the binary codes to approach a uniform distribution
in the Hamming space [166], it assists in optimizing the capacity of hash bits [141], preserving
the most semantic information. Third, because contrastive learning demonstrates promising
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:32 X. Luo et al.
performanceinvarioustasksincludinglinearclassification,clusteringaswellastransferlearning
[60,102], it aids in developing high-quality binary codes for effective retrieval [121].
Deep Hashing(DH) [40]. DH utilizes a deep hashing network and optimizes the parameters of
thenetworkwiththreecriteriaforthehashcodes.Firstly,itminimizesaquantizationlossbymin-
imizing the gap between the network output and the learnt hash codes. Secondly, it minimizes
the bit balance loss in Equation (14) so that generated binary codes distribute evenly on each bit.
Thirdly,itregularizestheweightsofhashingnetworkforindependenthashcodes.Theparameters
of the hashing network are updated by back-propagation based on the composite objective func-
tion. DH only imposes several constraints on hash codes without inferring similarity information
from the training data, which results in limited performance.
DeepBit [105]. DeepBit utilizes a deep convolutional neural network as the backbone. It also
minimizes the quantization loss as well as the bit balance loss. Differently, DeepBit enforces the
hash codes invariant to image rotation. The rotation invariant loss is formulated as
L(RI) =
N∑
i=1
R∑
θ=−R
exp
(
−θ2
2
)
/bardblex/bardblexhi−hi,θ/bardblex/bardblex
2, (68)
whereθis the rotation angle andhi,θdenotes the network output fromxi with rotationθ.T h i s
loss acts as a regularization term to enforce the hash codes invariant to certain transformations,
which improves the performance compared with DH.
UnsupervisedTripletHashing (UTH)[73].UTHbuildsthetripletsfromthedataset,eachofwhich
contains an anchor example, a rotated example along with a random example. Afterward, the
hashingnetworkisoptimizedusingthetripletinputs.Thequantizationlossandbitbalancelossare
also adopted for high-quality hash codes. The triplet loss compares the hash codes from different
hashcodes,whichhelpsgeneratediscriminativehashcodescomparedwiththeregularizationloss
in DeepBit. Hence, UTH performs better than DeepBit in various experiments.
HashGAN [45].HashGANcontainsthreenetworks,i.e.,agenerator,adiscriminator,andahash-
ingnetwork.Thehashingnetworkutilizes Lsigmoidfunctionforfinalactivation.Itsobjectivefor
real data contains four losses. It first minimizes the entropy of each bit, which is equivalent to a
quantization loss. The other three terms enforce the bit balance, invariance to different transfor-
mations,andbitindependence.SimilartoDSH-GAN,thediscriminatoristrainedinanadversarial
form.Italsoleveragesthesynthesizedimagesbyminimizingthedistancesbetweenoutputsofthe
hashing network and the inputs of the generator, which acts like an auto-encoder. Moreover, it
encouragesthegeneratortoproducesyntheticsampleswithsimilarstatisticstorealsampleswith
L2-norm loss. With GAN, HashGAN achieves better performance on both information retrieval
and clustering tasks.
Stochastic Generative Hashing(SGH) [31]. SGH proposes to utilize a generative manner to train
the hashing network through the Minimum Description Length principle. In this manner, the ob-
tained binary codes compress the whole dataset as much as possible. Specifically, it contains a
generative network and an encoding network to build the mapping between inputs and binary
codes from adverse directions. During optimization, it trains a variational auto-encoder to recon-
struct the input using the least information in binary codes. SGH is a general framework, which
can be degraded into ITQ [48] as well as Binary Autoencoder [21].
UnsupervisedHashingwithContrastiveInformationBottleneck (CIBHash)[134].CIBHashadapts
contrastive learning in deep unsupervised hashing. It considers the outputs of the hashing net-
work as a form of representation and minimizes the contrastive loss on the outputs. Specifically,
CIBHash generates two views for each input, and minimizes the contrastive learning objective,
i.e., Equation (67). To estimate the gradient of hashing network with discrete stochastic variables,
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:33
CIBHash leverages the straight-through gradient estimator [2] and the gradients are transmitted
entirelytothefrontlayersimilarto[ 156].CIBHashalsoillustratestheobjectivewithInformation
bottleneck theory with an improved model variant. From that moment on, contrastive learning
has been shown an effective tool for deep unsupervised learning since then.
Self-supervised Product Quantization(SPQ) [77]. SPQ combines contrastive learning with deep
quantization. The codewords and deep continuous representations are simultaneously optimized
bycontrastingindividuallyaugmentedviewsinacrossmanner.Specifically,fortwoviewsofeach
sample, i.e.,x(1)
i and x(2)
i , SPQ generates deep featuresz(1)
i and z(2)
i and employs codebooks in
the quantization head to generate quantized featuresˆz(1)
i and ˆz(2)
i . Instead of comparing similar-
ity between two visual descriptors or two quantized features, SPQ attempts to maximizes cross-
similaritybetweenthecontinuousrepresentationfromoneperspectiveandthefeatureafterprod-
uct quantization from the other perspective. In formulation,
LSPQ =−1
2NB
NB∑
i=1
/parenlefttpA
/parenleftbtA
log ez(1)
i ⋆ˆz(1)
i /τ
Z(2)
i
+log eˆz(1)
i ⋆z(2)
i /τ
Z(2)
i
/parenrighttpA
/parenrightbtA
, (69)
whereZ(1)
i = ∑
j/nequali ez(1)
i ⋆ˆz(2)
j /τandZ(2)
i = ∑
j/nequali eˆz(2)
i ⋆z(1)
j /τ.Withthecrosscontrastivelearningstrat-
egy, both codewords and continuous representationsare concurrentlyoptimized to produce high-
quality outputs for effective image retrieval.
Hashing via Structural and Intrinsic Similarity Learning(HashSIM) [119]. HashSIM utilizes con-
trastivelearningfordeepunsupervisedhashingfromadifferentview.Foreachbatch,itstackstwo
views of binary codes into two distinct matricesB(1) and B(2) ∈RNB×L, and takes their column
vectors as bit vectors{c(r)
l }L
l =1,r = 1 or 2. Then, HashSIM develops a intrinsic similarity learning
objective as follows:
LHashSIM =−1
2L
L∑
l =1
/parenlefttpA/parenleftexA
/parenleftbtA
log ec(1)
l ⋆c(2)
l /τ
Z(1)
i
+log ec(1)
l ⋆c(2)
l /τ
Z(2)
l
/parenrighttpA/parenrightexA
/parenrightbtA
, (70)
whereZ(r)
i = ∑
l′/nequall(ec(r)
l ⋆c(1)
l′ /τ+ec(r)
l ⋆c(2)
l′ /τ).Due to the fact the numerator attemptsto reduce the
gapbetweeneachhashbitunderdistinctaugmentationsandthedenominatorattemptstoenlarge
the distance between distinct bits, minimizing this self-supervised objective helps produce robust
and independent hash codes for effective image retrieval.
5 RELATED IMPORTANT TOPICS
5.1 Semi-Supervised Deep Hashing
Semi-supervised deep hashing simultaneously leverages the semantic information from both la-
beled samples and unlabeled samples, and a range of semi-supervised deep hashing models
have been developed recently. Compared with supervised methods and unsupervised methods,
these methods can typically overcome label scarcity in practical with limited performance degra-
dation. These methods usually incorporate semi-supervised techniques (e.g., pairwise pseudo-
labeling[148,176,187],GAN[ 161,162],andtransductivelearning[ 147])intodeepsemi-supervised
hashing. Then, the retrieval performance can benefit from abundant unlabeled images in the real
world. Generally, semi-supervised deep hashing provides a cost-effective solution to practical ap-
plications with promising performance, which desires further study in large-scale scenarios. We
then review these methods in detail.
Semi-Supervised Deep Hashing(SSDH) [187]. SSDH minimizes the semi-supervised loss func-
tioncontainingthreeterms,i.e.,arankingterm,aembeddingterm,aswellasapseudo-labelterm.
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:34 X. Luo et al.
Supervised ranking term leverages a triplet loss for labeled data. Then, SSDH generates an on-
line k-NN graph for all data, which guides pairwise similarity preserving of the hashing network.
Moreover,in semi-supervisedsettings,it generatespseudo-labels whichfurtherguide the similar-
ity preserving process. SSDH is the first to perform deep hashing in a semi-supervised fashion.
Deep Hashing with a Bipartite Graph(BGDH) [176]. BGDH builds a bipartite graph to uncover
the latent semantic structure for unlabeled data. Different from the similarity graph in unsuper-
vised hashing, its similarity structure is based on the relationships between labeled examples and
unlabeledexamples,resultinginabipartitegraph.Then,BGDHutilizesthebipartitegraphtoguide
hash code learning by pairwise similarity preserving. It also adopts the loss term in DPSH [101]
for supervised learning. Through mining the relationship in deep feature space, BGDH utilizes
unlabeled data in an appropriate manner and improves the performance.
Semi-SupervisedGenerativeAdversarialHashing (SSGAH)[161].SSGAHcombinesaGenerative
Adversarial Network with deep semi-supervised hashing. It contains a generative network, a dis-
criminator and a deep hashing network. The generative network produces two synthetic images
xp
synand xn
synfor each real imagex and the similarity betweenx and xp
synis larger than the
similarity betweenx and xn
syn. In this way, SSGAH learns the distribution of triplet-wise seman-
tic message from both labeled samples as well as unlabeled samples. The discriminator estimates
the likelihood that each input is synthetic. The hashing network is optimized using a triplet loss
withtheincorporationofsyntheticpositiveandnegativeimages.SSGAHcanproducehashcodes,
which could sufficiently explore semantics in the datasets by training the framework using an
adversarial manner.
Semi-supervised Deep Pairwise Hashing(SSDPH) [148]. SSDPH chooses a variety of labeled an-
chors in the training set, and then uses the pairwise objective for preserving similarities between
labeled samples. More importantly, it leverages the technique of temporal ensembling from semi-
supervised learning for learning similarity information from unlabeled data. Specifically, it con-
tains a teacher model and a student model. The teacher model provides supervised information
to guide the similarity learning, which is then updated in an ensemble manner. SSDPH first com-
bines deep hashing with semi-supervised techniques, which improves the retrieval performance
in real-world applications.
Transductive Semi-supervised Deep Hashing(TSDH) [147]. TSDH extends the traditional trans-
ductive learning principle into deep semi-supervised hashing, which treats pseudo-labels of unla-
beleddataasvariablesandoptimizesthemalternativelywiththehashingnetwork.Toaccomplish
this,itaddsaclassificationlayerafterproducinghashcodes.Moreover,itinvolvesapairwiseloss
forsimilaritypreservation.Lastly,TSDHestimatestheconfidenceofpseudo-labelsbytheproxim-
ity distance
vi =
∑
zj∈N(zi)
∥zi−zj∥2, (71)
ri = 1−vi
vmax
, vmax = max{v1,..., vN}, (72)
wherezi is the extracted features ofxi and N(zi) denotes the k-nearest neighbor set ofzi.I nt h i s
manner, samples that reside in densely populated regions are assigned a high confidence level.
In summary, TSDH utilizes the popular transductive learning technique to improve the retrieval
performance of semi-supervised hashing.
AdversarialBinaryMutualLearning (ACML)[162].ACMLalsointegratesaGenerativeAdversar-
ial Network into semi-supervised deep hashing. Specifically, it contains a discriminative network
and a generation network to mould the relationships between inputs and binary codes from op-
posite views. Then, an adversarial network is trained to differentiate between real and fake pairs
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:35
of samples and their hash codes. In this way, it can leverage unlabeled data to make the discrimi-
native network and generation network mutually learn from each other. Moreover, it introduces
a Weibull distribution for better similarity preserving. ACML combines a Generative Adversarial
Network with deep semi-supervised hashing and shows promising retrieval performance.
5.2 Domain Adaptation Deep Hashing
The data in the domain of interest is likely to be insufficient in practice, while the labeled sam-
ples from a separate but correlated domain is usually accessible. To sufficiently utilize the labeled
samples from source domains, several domain adaptive hashing methods have been developed
in recent years. These hashing methods usually combine similarity preserving techniques (e.g.,
pairwise [160, 199] and ranking-based similarity preserving [118]) in deep supervised hashing
with domain adaptation techniques (e.g., discrepancy minimization [70, 188], adversarial learn-
ing [62, 118], and centroid alignment [62, 160]). Hence, their methods are quite flexible. However,
thecross-domainretrievalperformanceofcurrenthashingmethodsisstillnotsatisfactory,which
desires further exploration in the future. We then review these methods as follows.
Domain Adaptive Hashing(DAH) [160]. DAH contains three parts, i.e., a supervised hashing
module for source data, an unsupervised hashing module for target data, and a domain disparity
reduction module. For source data, it minimizes the likelihood loss along with the quantization
loss. For source data, it leverages the source output to generate the label distributions and then
minimizes the entropy to ensure that the target outputs approximate source outputs from each
category. Furthermore, DAH reduces the domain difference between the source and target rep-
resentations through the minimization of multi-kernel Maximum Mean Discrepancy. This work
is the first to combine unsupervised domain adaptation with deep hashing and improves the effi-
ciency for cross-domain image retrieval.
DomainAdaptiveHashingwithIntersectantGenerativeAdversarialNetworks (IGAN)[62].Differ-
entfromDAH,IGANgeneratesthepseudo-labelsfortargetdomainsandthenalignsthesemantic
centroid for all categories. Moreover, it leverages two generators to reconstruct images in two
domains and the generators and the discriminators are updated using a GAN objective. IGAN im-
provestheretrievalperformanceusingGANaswellascentroidalignment,whicharetwocommon
techniques in domain adaption.
Deep Domain Adaptation Hashing with Adversarial Learning(DeDAHA) [118]. DeDAHA con-
tains two different CNNs for learning image representations. An adversarial loss is enforced to
explore the knowledge robust to different domains. Then DeDAHA utilizes a standard triplet loss
tolearnthehashingencoder.Whenthelabelannotationsintargetdataareunavailable,DeDAHA
leverages a multi-stage framework for unsupervised domain adaptation hashing.
Deep Transfer Hashing(DPH) [199]. DPH first uses a neural network to extract deep features
and then incorporates a deep transformation mapping network for domain adaptation. Then, for
effectivetransferlearning,DPHgeneratesthesimilarityinformationbasedonthecosinesimilarity
of deep features as well as the hash codes in source domains to guide hash code learning. DPH
shows great generality utilizing the powerful representative capacity of deep learning.
Optimal Projection Guided Transfer Hashing(GTH) [188]. GTH seeks for the maximum likeli-
hood estimation solution to minimize the error matrix between two hash projections of target
and source domains. In this way, GTH can produce domain-invariant hash projections for effec-
tivecross-domainimageretrieval.However,GTHassumesthatsimilardomainsshouldhavesmall
discrepancies between hash projections, which may be not promised in most scenarios.
Domain Adaptation Preconceived Hashing(DAPH) [70]. DAPH first reduces the distribution
discrepancy across two domains through learning a transformation matrix to project the samples
from different domains into a common space. Moreover, it involves a reconstruction constraint
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:36 X. Luo et al.
to release the information loss from the transformation. For effective hash code learning, it
adds a quantization loss to project features into hash codes. The whole learning process is in an
alternative manner for updating the transformation matrix, projection, and binary codes. DAPH
improves the performance for challenging cross-domain retrieval.
5.3 Multi-Modal Deep Hashing
Multimediadatahaveexplodedinmultiplemodalitiesincludingtext,audio,image,andvideosince
thedawnoftheinformationeraandthefastexpansionoftheInternet.Multi-modeldeephashing
has arisen much interest in the field of deep hashing recently. These methods typically project
multiplemodalitiesofdataintoasharedHammingspaceusingdeepneuralnetworksforeffective
cross-modal retrieval. The framework of multi-modal deep hashing methods is similar to general
deep hashing methods except that the similarity information includes the intra-modal and inter-
modal forms. However, each loss term characterizing the similarity information is similar to that
in deep supervised hashing discussed above. Existing methods [170] can also be categorized into
supervised methods [13, 83, 178] and unsupervised methods [66, 168, 184]. Cao et al. [11]g i v ea
detailedreviewforthemulti-modalhashingmethodsthatincludes[ 12,13,28,35,50,65,68,81,83,
95,97,178, 183,192].
6 EVALUATION PROTOCOLS
6.1 Evaluation Metrics
For deep hashing algorithms, the space cost only depends on the length of the hash codes, so the
length is usually kept the same when comparing the performance of different algorithms. The
search efficiency is measured by the average search time for a query, which mainly depends on
the architecture of the neural networks. Besides, if the weighted Hamming distance is used, we
cannot take advantage of bit operation for efficiency.
Asdiscussedabove,weusuallyusesearchaccuracytomeasureperformance.Themostpopular
matrices include Mean Average Precision, Recall, Precision, as well as the precision-recall curve.
Precision: Precision is defined by the proportion of returned samples that share the common label
with the query. The formula can be formulated as
precision = TP
TP +FP, (73)
whereTP denotes the number of returned samples that have a common label with the query and
FP denotes the number of returned samples that do not have a common label with the query.
precision@k means the total number of returned sample isk, i.e.,TP +FP =k.
Recall: Recall is defined by the proportion of samples in the database that have a common label
with the query that is retrieved. The formula can be formulated as
recall = TP
TP +FN, (74)
whereFN isthetotalnumberofsamplesinthedatabasethathaveacommonlabelwiththequery,
including samples not retrieved.recall @k means the number of returned examples isk.
Precision-recall curve: The precision rate and recall rate in image retrieval are both influenced
by k. The precision and recall rates of an approach are inversely proportional. As a result, we
could create the precision-recall curve by alteringk and using the precision rate and recall rate,
respectively.
Meanaverageprecision (MAP):Whentherecallratevariesbetween0and1,theaverageaccuracy
can be computed by varying the precision rate. The sequence summation approach is used to
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:37
compute the average accuracy in practical applications with discretion
AP = 1
F
N∑
k=1
precision@kΔ{T@k}, (75)
in whichΔ{T@k} denotes the change in recall from itemk−1t ok.T h es u mo fΔ{T@k} isF and
the core idea of AP is to evaluate a ranked list through averaging the precision at every position.
Afterward, MAP can be derived by taking the mean of the average precision of every query. In
severalworks,MAPiscalculatedintermsoftop K rankedretrievalresults.Someresearchersalso
calculate the MAP with Hamming Radius r, when only samples with distances not bigger thanr
are considered.
Alexandre Sablayrolles et al. [135] show that the above popular evaluation protocols for super-
vised hashing are not satisfactory because a trivial solution that encodes the output of a classifier
significantly outperforms existing methods. Furthermore, they provide a novel evaluation proto-
col based on retrieval of unseen classes and transfer learning. However, if the design of hashing
methods avoids using the encoding of the classifier, the above popular evaluation protocols are
still effective generally.
6.2 Datasets
The scales of regularly used assessment datasets range from small to large to extremely large.
Single-label datasets and multi-label datasets are two types of datasets.
MNIST [94] is comprised 60,000 training samples and 10,000 testing samples. It is a single-
labeleddataset,wherethe10differentclassesrepresentdifferentdigits.Eachimageisrepresented
by 784-dimensional raw features.
CIFAR-10[89]iscomprised60,000real-worldimagesin10distinctcategorizations.Itisasingle-
labeled dataset, where 10 different categorizations imply airplanes, cars, birds, cats, deer, dogs,
frogs, horses, ships, as well as trucks. These examples are identified with semantic labels utilized
to assess the performance of various hashing approaches.
ImageNet [34] is a large-scale dataset that consists of over 1.2 million images hand-annotated
by the huge project to find out what objects are included. It is a single-labeled dataset, consisting
of 1,000 categories such as “balloon” or “strawberry”.
NUS-WIDE[29] is a well-known multi-labeled image dataset collected by a team from NUS. It
consists of 269,648 examples with 5,018 unique tags. These samples are manually associated with
someofthe81concepts.Becauseimageshavetypicallyoveronelabel,twosamplesaretreatedas
semantic similar if they share one common semantic label.
MS COCO[109]isapopularmulti-labeleddatasets,consistingof82,783trainingexamplesalong
with 40,504 validation examples, each of which is associated with part of the 80 categories. After
removingexampleswithoutanyclassinformation,122,218samplescanbeobtainedforevaluating
the performance of hashing methods.
6.3 Performance Analysis
6.3.1 Performance Comparison of Deep Supervised Methods.We present the results of some
representative deep supervised hashing and quantization algorithms over CIFAR-10, NUS-WIDE,
ImageNet, and MS COCO. For CIFAR-10, 100 images are selected randomly per class (resulting in
1,000 images totally) as queries and the rest of samples are adopted as the database. A total of 500
samples per class (resulting in 5,000 samples totally) make up the training set. For NUS-WIDE, a
subset of 195,834 samples that correspond to the 21 most frequent labels are picked. Afterward,
100 samples per class (resulting in 2,100 samples totally) are picked as queries and the remaining
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
15:38 X. Luo et al.
samplesmakeuptheretrievalset.Atotalof500imagesperclass(resultingin10,500imagestotally)
aresampledasthetrainingset.ForImageNet,100categoriesarerandomlyselectedasin[ 19].The
samples associated with the chosen categories in the training set make up the database, and the
samplesinthevalidationsetareutilizedasqueries.Atotalof100examplesofeachcategoryarese-
lectedfromthedatabasefortraining.ForMSCOCO,5,000samplesareusedasqueriesandtherest
are used as the database. A total of 10,000 samples from the database are selected for training.
Notethatforthevariousexperimentalsettings,mostoftheexperimentalresultsarenotshown
in this summary in detail. The representative compared results of hashing methods are shown in
Tables4 and5. From the results, there are several observations as follows:
—Deep supervised hashing greatly outperforms traditional hashing methods (SDH and KSH)
overall, validating the strong representation-learning capacity of deep learning.
—Similarityinformationisnecessaryfordeephashing.Fordeepsupervisedhashingmethods
in the early period (i.e., before 2016), hash codes are mostly obtained by transferring classi-
ficationmodelswithoutsupervisedsimilarityinformationwhilethemethodswithpairwise
and ranking information outperform them.
—Label information helps to increase the performance of deep hashing. This point can be
shown from the fact that DSDH outperforms DPSH evidently and the superiority of Lab-
Net. Moreover, several pointwise methods (CSQ, OrthHash, and PSLDH) show comparable
performancerecentlybymappingthelabelsintoHammingspace,achievingimpressiveper-
formance on large-scale datasets.
—Severalskillsincludingregularizationterm,bitbalance,ensemblelearning,andbitindepen-
dence help to obtain accurate and robust performance, which can be seen from ablation
studies in some papers [36].
—Althoughsupervisedhashingmethodshaveachievedremarkableperformance,theyarediffi-
culttobeappliedinpracticesincelarge-scaledataannotationsareunaffordable.Toaddress
this problem, deep learning-based unsupervised methods provide a cost-effective solution
for more practical applications.
6.3.2 PerformanceComparisonofDeepUnsupervisedMethods. Thispartpresentstheresultsof
representativedeepunsupervisedhashingapproachesoverCIFAR-10,NUS-WIDE,andMSCOCO.
We follows the setting in prior works [121, 134, 145]. The dataset splits for training, testing, and
database are the same as Section6.3.1. Part of records are quoted from [134,145,146].
The compared results are shown in Table 6. From the results, we have the following
observations:
—Deep unsupervised hashing methods generally perform better than the traditional ap-
proaches (ITQ, BRE, SDH, KSH, and LSH), which suggests that the powerful representation
learning capacity of deep learning is beneficial to the retrieval performance of generated
binary codes in most cases.
—The methods that only adopt regularization terms (DeepBits and UTH) obtain poor results
among compared methods, demonstrating that the exploration of semantic information is
indispensable for discriminative hash codes.
—The methods that explore more accurate similarity structures (DATE and TBH) outperform
earlyapproachesthatobtainsimilaritystructureinacoarsemanner(SSDHandDistillHash).
The potential reason is that false similarity signals will result in error propagation during
subsequent hash code learning, implying suboptimal performance.
—The methods utilizing contrastive learning (CIBHash and DATE) achieve superb perfor-
manceamongcomparedmethods,whichimpliesthatcontrastivelearningisaneffectivetool
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.
A Survey on Deep Hashing Methods 15:39
Table 4. MAP for Different Hashing Methods on CIFAR-10 and NUS-WIDE
CIFAR-10 NUS-WIDE
Method 12bits 24bits 32bits 48bits 12bits 24bits 32bits 48bits
CNNH [173] 0.439 0.511 0.509 0.522 0.611 0.618 0.625 0.608
DNNH [92] 0.552 0.566 0.558 0.581 0.674 0.697 0.713 0.715
DHN [200] 0.555 0.594 0.603 0.621 0.708 0.735 0.748 0.758
DRSCH [189] 0.614 0.621 0.628 0.630 0.618 0.622 0.622 0.627
DSCH [189] 0.608 0.613 0.617 0.619 0.591 0.597 0.610 0.608
DSRH [197] 0.608 0.610 0.617 0.617 0.609 0.617 0.621 0.630
DSH-GAN [133] 0.735 0.781 0.787 0.802 0.838 0.856 0.861 0.863
DTSH [169] 0.710 0.750 0.765 0.774 0.773 0.808 0.812 0.824
DPSH [101] 0.713 0.727 0.744 0.757 0.752 0.790 0.794 0.812
DSDH [100] 0.740 0.786 0.801 0.820 0.776 0.808 0.820 0.829
DQN [17] 0.554 0.558 0.564 0.580 0.768 0.776 0.783 0.792
DSH [111] 0.644 0.742 0.770 0.799 0.712 0.731 0.740 0.748
DCEH [172] 0.745 0.788 0.802 0.806 0.781 0.816 0.827 0.839
DDSH [82] 0.753 0.776 0.803 0.811 0.776 0.803 0.810 0.817
DFH [103] 0.803 0.825 0.831 0.844 0.795 0.823 0.833 0.842
Greedy Hash [156] 0.774 0.795 0.810 0.822 - - - -
MIHash [8] 0.738 0.775 0.791 0.816 0.773 0.820 0.831 0.843
HBMP [9] 0.799 0.804 0.830 0.831 0.757 0.805 0.822 0.840
VDSH [195] 0.538 0.541 0.545 0.548 0.769 0.796 0.803 0.807
NMLayer [43] 0.786 0.813 0.821 0.828 0.801 0.824 0.832 0.840
HashNet [19] 0.685 0.707 0.705 0.705 0.770 0.802 0.806 0.816
AnDSH [198] 0.754 0.780 0.786 0.795 0.780 0.808 0.815 0.823
DISH [193] 0.738 0.792 0.822 0.841 0.781 0.823 0.837 0.840
SRE [194] 0.771 0.817 0.839 0.858 0.801 0.833 0.849 0.861
MLDH [124] 0.805 0.825 0.829 0.832 0.800 0.828 0.832 0.835
SDH [140] 0.285 0.329 0.341 0.356 0.568 0.600 0.608 0.638
KSH [113] 0.303 0.337 0.346 0.356 0.556 0.572 0.581 0.588
ITQ[ 48] 0.127 0.128 0.126 0.129 0.454 0.406 0.405 0.400
fordiscriminativehashcodelearning.Astheresearchmovesalong,deepunsupervisedlearn-
ingmethodscanevenoutperformpartofdeepsupervisedmethods,whichisreallyinspiring.
6.4 Training Time Cost
In this part, we investigate the training efficiency of different deep hashing methods. A total of
10 representative methods are selected. These methods are parameterized by different network
ACM Transactions on Knowledge Discovery from Data, Vol. 17, No. 1, Article 15. Publication date: February 2023.


*(Extracción truncada en la página 40 de 51 totales)*