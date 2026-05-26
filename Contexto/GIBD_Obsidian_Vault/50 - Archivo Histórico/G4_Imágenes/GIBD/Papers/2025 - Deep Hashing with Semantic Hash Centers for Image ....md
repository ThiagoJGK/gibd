---
aliases: [2025 - Deep Hashing with Semantic Hash Centers for Image ...]
tags:
  - grupo/g4_imágenes
  - estado/archivo
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2025-07-22
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Papers/Índices/2025 - Deep Hashing with Semantic Hash Centers for Image Retrieval.pdf"
tamanio_bytes: 5320274
---

# 2025 - Deep Hashing with Semantic Hash Centers for Image Retrieval

Ruta interna: `GIBD/Papers/Índices/2025 - Deep Hashing with Semantic Hash Centers for Image Retrieval.pdf`

---

Deep Hashing with Semantic Hash Centers for Image Retrieval
LI CHEN, Beihang University, China
RUI LIU, Beihang University, China
YUXIANG ZHOU, Beijing University of Posts and Telecommunications, China
XUDONG MA, Beihang University, China
YONG CHEN∗, Beijing University of Posts and Telecommunications, China
DELL ZHANG† , TeleAI, China
Deep hashing presents an effective strategy for large-scale image retrieval. Current hashing methods are generally categorized by
their supervision types: point-wise, pair-wise, and list-wise. Recent advancements in point-wise methods (e.g., CSQ, MDS) have
significantly enhanced retrieval performance across diverse datasets by pre-assigning a hash center to each class, thereby improving
the discriminability of the resultant hash codes. However, these methods employ purely data-independent algorithms for generating
hash centers, overlooking the semantic connections between different classes, which, we argue, could degrade retrieval performance.
To tackle this problem, this paper expands on the newly emerged concept of “hash centers” to introduce “semantic hash centers”, which
posits that hash centers of semantically related classes should exhibit closer Hamming distances, while those of unrelated classes
should be more distant. Based on this hypothesis, we propose a three-stage framework, termed SHC, to produce hash codes that
preserve semantics. First, we build a classification network to detect semantic similarities between classes, and utilize a data-dependent
approach to similarity calculation that can adapt to varied data distributions. Next, we develop a new optimization algorithm to
generate semantic hash centers. This algorithm not only maintains semantic relatedness among hash centers but also integrates
a constraint to ensure a minimum distance between them, addressing the issue of excessively proximate hash centers potentially
impairing retrieval performance. Finally, we train a deep hashing network with the above generated semantic hash centers to convert
each image into a binary hash code. Experiments on large-scale image retrieval across several public datasets demonstrate that
SHC generates more discriminative hash codes, markedly enhancing retrieval performance. Specifically, in terms of the MAP@100,
MAP@1000, and MAP@ALL metrics, SHC records average improvements of +7.26%, +7.62%, and +11.71%, respectively, over the most
competitive existing methods.
CCS Concepts: •Information systems → Top-k retrieval in databases.
Additional Key Words and Phrases: Learning to Hash, Hash Center, Quantization, Representation Learning, Image Retrieval.
∗Yong Chen is the Corresponding author (yong.chen@bupt.edu.cn).
†Dell Zhang is also the Corresponding author (dell.z@ieee.org).
Authors’ Contact Information: Li Chen, cc752424640@buaa.edu.cn, Beihang University, Beijing, China; Rui Liu, Beihang University, Beijing, China;
Yuxiang Zhou, Beijing University of Posts and Telecommunications, Beijing, China; Xudong Ma, macaronlin@buaa.edu.cn, Beihang University, Beijing,
China; Yong Chen, alphawolf.chen@gmail.com, Beijing University of Posts and Telecommunications, Beijing, China; Dell Zhang, dell.z@ieee.org, TeleAI,
Shanghai, China.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1
arXiv:2507.08404v1  [cs.CV]  11 Jul 2025
2 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
ACM Reference Format:
Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, and Dell Zhang. 2025. Deep Hashing with Semantic Hash Centers for
Image Retrieval. In Proceedings of ACM Transactions on Information Systems (TOIS 2025). ACM, New York, NY, USA, 28 pages.
https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
With the onset of the big data era, images have become increasingly vital in social media, e-commerce, surveillance,
and various other domains [39, 40]. This shift poses substantial challenges in managing the growing volume of images,
necessitating the development of efficient and effective methods for indexing and searching large-scale image data
repositories. Image hashing, as a quantization technique [ 13, 18, 35], has emerged as a practical solution to these
challenges. By converting images into fixed-length binary codes, it not only cuts down on storage needs significantly
but also enables rapid comparisons and retrievals via hardware-level XOR operations, and thus benefits a wide array of
applications.
cat dog car
far
far
close (≥ 𝒅) 
Gilbert-Varshamov bound + 
Semantic Constraint
Gilbert-Varshamov bound 
   →  𝒅 can be calculated
≥ 𝒅
≥ 𝒅
≥ 𝒅
Overlapping
Bern(0.5)→ 
𝒅 could be very small
Hadamard Matrix → 
𝒅 = 𝒒/𝟐 
𝒅
𝒅
𝒅
Semantic Mismatch
𝟏  𝐂𝐒𝐐(𝑪 ≤ 𝟐 × 𝒒) 𝟐  𝐂𝐒𝐐(𝑪 > 𝟐 × 𝒒) 𝟑  𝐌𝐃𝐒 𝟒  𝐒𝐇𝐂
Hash Centers: cat dog carImages:
Fig. 1. Illustration of hash center based image hashing approaches: CSQ (1&2), MDS (3), and our SHC (4).
Image hashing has undergone a series of advancements along with the evolution of technology [26, 36]. Initially,
this research field focused on data-independent, shallow feature-based hashing methods, such as Locality Sensitive
Hashing (LSH) [10, 17, 22]; but recently, it has transitioned to data-dependent, deep semantic feature-based deep hashing
approaches, such as DPSH [24, 31], HashNet [5], ADSH [19], and CSQ [42]. This transformation has vastly improved
the capability of image hashing, from processing small datasets in its early stages to efficiently managing large-scale
data sets, underscoring its practical utility.
Existing deep hashing methods can be broadly categorized into three types: pair-wise similarity-based methods [4, 5,
24, 43, 44, 46], list-wise similarity-based methods [23, 27, 38], and point-wise methods [6, 16, 30, 37, 42]. Both pair-wise
and list-wise approaches primarily focus on the relationships among local images, which limits their effectiveness. To
address this problem, CSQ [42] introduces the novel concept of “hash centers”, in the category of point-wise methods.
This approach utilizes the Hadamard matrix to predefine hash centers in the Hamming space for each image class,
aiming to align the hash codes of similar images with their respective hash centers. CSQ shows impressive performance
when the number of categories (i.e., 𝐶) is less than twice the length of hash codes (i.e., 𝑞), as illustrated in Fig. 1(1).
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 3
However, challenges arise when the number of categories exceeds twice the length of hash codes. In these cases, the
Hadamard matrix cannot produce a hash center for each individual category. Consequently, some hash centers are
randomly generated according to the Bernoulli distribution. This can lead to hash centers being too close or even
overlapping, which would result in complete misclassification of certain categories, as depicted in Fig. 1(2).
MDS [37] proposes an novel solution to the above mentioned issue. Unlike CSQ, MDS makes use of the Gilbert-
Varshamov bound [33] to determine the feasible minimum distance between hash centers, taking both the number
of image categories and the length of hash codes into account to ensure optimal spacing between hash centers. MDS
strives to eliminate the problem of overly close hash centers, particularly when dealing with a vast number of categories
and a limited hash code length. As a result, MDS achieves superior performance compared to CSQ, as shown clearly in
Fig. 1(3).
Nevertheless, both CSQ and MDS rely solely on mathematical optimality to generate hash centers, which results in a
lack of semantic interpretability (see Fig. 1). These methods are agnostic to the actual data, and assume that a larger
Hamming distance between hash centers translates straightforwardly into better retrieval accuracy. This presumption
overlooks the crucial similarity relationships between different categories. Therefore, the hash centers generated by
these methods are short of discriminative power and are arbitrarily assigned to categories, which undermines the
effectiveness of image retrieval. We argue that this approach is flawed and substantially impairs the potential for optimal
retrieval performance.
Therefore, in this paper, we introduce SHC, which posits that in a confined Hamming space, the quality of hash
centers should not merely depend on maximizing the distances between them; it is imperative to consider the semantic
relationships among categories as well. This new approach ensures that while maintaining proper distance between hash
centers, they are also endowed with semantic discriminatory capabilities. Specifically, semantically similar categories
ought to possess hash centers with shorter Hamming distances, whereas semantically dissimilar categories should
exhibit longer distances. We designate these as “semantic hash centers”, as illustrated in Fig. 1(4).
Our main contributions are threefold.
• We advance the novel concept of “hash centers” to the more explainable “semantic hash centers”, and introduce
a data-dependent similarity calculation method that utilizes the inherent semantic content of images, moving
beyond the conventional use of categorical labels. This approach enhances adaptability across varied data
distributions.
• We develop a comprehensive optimization framework tailored for generating semantic hash centers. This frame-
work aims to preserve semantic relationships while enforcing minimum distance constraints. By incorporating
proxy variables and harnessing the ALM optimization scheme, we have successfully broken down the NP-hard
problem into smaller, more manageable sub-optimization tasks, yielding impressive optimization results.
• Extensive experiments on multiple datasets underscore the superiority of our SHC approach over other cutting-
edge deep hashing methods, defining the new state of the art. Specifically, in terms of the evaluation metrics
MAP@100, MAP@1000, and MAP@ALL, SHC demonstrates significant performance improvements, with average
increases of +7.26%, +7.62%, and +11.71%, respectively. Ablation studies further validate the efficacy of our
semantic and minimum distance constraints, as well as the benefits of utilizing image semantics over categorical
labels.
Manuscript submitted to ACM
4 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
2 Related Work
Supervised deep hashing approaches use deep learning techniques to learn, from labeled data, hashing functions that
map original high-dimensional data points to a low-dimensional hash code space, enabling efficient and accurate data
retrieval. They can be categorized into pair-wise, list-wise, and point-wise methods, according to the way of semantic
measurement.
Pair-wise methods establish the similarities among data points by using pairs of images. If two images share at
least one common label, they are deemed similar and assigned a value of 1 in the pairwise similarity matrix; otherwise,
they are considered dissimilar and assigned a value of 0. Some prominent pairwise approaches include DPSH [ 24],
HashNet [5], DCH [4], BNNH [43], and IDHN [44]. DPSH marks the early efforts in this field, while HashNet adjusts
the loss function to deal with the challenge of semantic imbalance. Besides, DCH introduces the concept of “Hamming
radius” to transform the linear retrieval problem into an index lookup problem, thereby enhancing efficiency. In addition,
BNNH replaces CNN with BNN to minimize the parameter overhead, making it much more computationally efficient.
Lastly, IDHN quantifies the similarity of data pairs based on normalized semantic labels, significantly improving image
retrieval performance on multi-label datasets.
List-wise methods rank or order data pairs based on their labels, where a set of samples (or a list) is considered
simultaneously during training, rather than just pairs or individual points. DTSH [38] is a typical list-wise method,
which, unlike DPSH, uses triplet similarity to process images. In a triplet, the first image is similar to the second, but
differs from the third. The training objective not only constrains the Hamming distances of similar image pairs to be
close but also enforces a large distance between dissimilar image pairs, resulting in a superior performance compared to
DPSH. DSRH [25, 41, 45] is another representative ranking-based method that directly learns the ranking of multi-label
similarities and proposes a novel loss function to optimize the multi-label ranking metric.
Point-wise methods rely on image categories as supervised signals. Prominent examples of such methods include
GreedyHash [30], LTH [6], CSQ [42], MDS [37], and OrthoHash [16]. GreedyHash uses a point-wise loss function
that combines classification loss with penalty terms. During training, the sign function is seamlessly integrated, and a
greedy backpropagation algorithm is employed to fine-tune network parameters. LTH adopts the familiar cross-entropy
loss function, used in standard classification tasks. It reconnects image hash codes to labels and computes cross-entropy
loss with the original labels, thus preserving semantic congruity between hash codes and image labels. CSQ introduces
a global similarity metric and the concept of “hash centers”. Each image category is assigned a unique hash center,
maintaining a constant Hamming distance between them. CSQ encourages images of the same category to cluster around
their respective centers, resulting in compact hash codes. MDS builds on CSQ, addressing the challenge of minimizing the
distance between hash centers when the number of categories exceeds the capacity of Hadamard matrices. Orthohash
introduces a deep hashing model with a unified learning objective, demonstrating that maximizing cosine similarity
between continuous codes and their corresponding binary orthogonal codes ensures both discriminative hash codes
and minimal quantization errors.
Our SHC method draws inspiration from the point-wise CSQ [ 42] and MDS [ 37] approaches, particularly their
innovative concept of “Hash Center”. SHC expands upon this idea by introducing a novel concept called “Semantic
Hash Center”, which not only takes into account the distance constraints imposed on hash centers but also endows
them with the ability to maintain semantic consistency.
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 5
Fig. 2. The three-stage architecture of our SHC method: similarity matrix construction, hash centers generation, and hashing network
training.
3 Method
Image hashing aims to find a hash function F : x → {− 1, +1}𝑞 that converts an image x into a 𝑞-bit vector, so that the
more similar two images are, the smaller their Hamming distance will be.
The recently appeared SOTA methods, e.g., CSQ [42] and MDS [37], are hash centers based optimization algorithms,
which involve two stages, i.e., hash centers generation and hashing network training. They assume that the further
their generated hash centers are, the better their performances will be; however, they neglect the semantic correlations
between their generated hash centers.
Hence, we propose an enhanced deep hashing approach that employs a 3-stage pipeline, i.e., similarity matrix
generation, hash centers generation, and hashing network training, as shown in Fig. 2, which not only tries to keep
hash centers as far away from each other as possible, but also maintains semantic consistencies between them.
3.1 Stage 1: Construct the Data-dependent Pairwise Similarity Matrix
Commonly, there are 2 methods for constructing similarity matrix S = {s𝑖 𝑗 }𝑁 ×𝑁 , of which s𝑖 𝑗 represents the similarity
score between the 𝑖-th and 𝑗-th image, and 𝑁 denotes the total number of images. For the first method, if image x𝑖 and
image x𝑗 share the same class label, then s𝑖 𝑗 equals 1; or else s𝑖 𝑗 equals 0. For the second method, s𝑖 𝑗 is usually defined
as the cosine value of image x𝑖 and image x𝑗 , i.e.,
s𝑖 𝑗 = cosine < x𝑖, x𝑗 >=
x𝑇
𝑖 x𝑗
||x𝑖 || · || x𝑗 || , (1)
Manuscript submitted to ACM
6 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
where x𝑖 (𝑖 = 1, 2, · · ·, 𝑁) can be computed as the semantic vector of its textual labels from pre-trained large language
models such as BERT [11] or GPT [2]. These two methods have been testified to be useful in semantic preservations [5, 24];
however, the first approach is coarse-grained with only {0, 1} values while the second one is not adaptive to various
data distributions w.r.t. the same classes.
Here, we devise a data-dependent similarity matrix construction method via conducting a pre-classification task on
concrete image datasets. More specifically, we utilize a pre-trained ResNet34 [14] network followed by a softmax layer
for image classifications, i.e.,
p0𝑖 = ResNet34(x𝑖 ), (2)
p𝑖 = softmax(p0𝑖 ), (3)
where p𝑖 is a predicted probability vector over the 𝐶 classes, and traditionally the image x𝑖 is categorized to the 𝑗-th
class with
𝑗 = arg max
𝑘
{p𝑖𝑘 }, 𝑘 = 1, 2, · · ·, 𝐶. (4)
However, other values of p𝑖, i.e., p𝑖𝑘 (k≠j), also covering rich semantics, are not used to the fullest. In fact, the larger
p𝑖𝑘 is, the higher the similarity between the 𝑖-th and the 𝑘-th class can be treated. Usually, p𝑖 𝑗 is the largest, and much
larger than the other values p𝑖𝑘 , as shown in Fig. 2; thus, we first mask the maximum valuep0𝑖 𝑗 of p0𝑖, and then softmax
it to p′
𝑖 , formulated as:
ˆp𝑖 ≜ mask𝑗 (p0𝑖 ) ∈ R𝐶, (5)
and
p′
𝑖 = softmax( ˆp𝑖 ), (6)
where ˆp𝑖𝑘 = p0𝑖𝑘 and ˆp𝑖 𝑗 = −INF.
For all the images in the 𝑚-th category, we average their similarity vectors, i.e.,
s𝑚 = 1
𝑁𝑚
𝑁𝑚∑︁
𝑙=1
p′
𝑙 , (7)
where 𝑁𝑚 represents the number of images in category 𝑚, and s𝑚 represents the overall similarity between catetory 𝑚
and other categories.
Next, we normalize s𝑚 to (−1, +1)𝐶-vector via:
s𝑚 = s𝑚 − s𝑚0
max{| max(s𝑚) − s𝑚0|, | min(s𝑚) − s𝑚0|} , (8)
where
s𝑚0 = 1
𝐶
𝐶∑︁
𝑙=1
s𝑚𝑙 . (9)
Since s𝑚 denotes the overall similarity between the 𝑚-th class and the other classes, we can concatenate them as the
whole pairwise similarity matrix S:
S = [s1, s2, · · ·, s𝐶 ]𝑇 ∈ R𝐶 ×𝐶, (10)
of which s𝑖 𝑗 denotes the similarity between the𝑖-th and the 𝑗-th class. Here,S is not symmetric and its diagonal elements
does not equal 1 yet, then we further conduct the following operators for the final similarity matrix:
S = S + ST
2 , (11)
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 7
and
s𝑖𝑖 = 1, (12)
which deserves most notice that it’s data-dependent instead of labels-based, thus adaptive to various data distributions.
3.2 Stage 2: Generate the Semantic Hash Centers
In this part, we develop an optimization procedure to yield a set of hash centers, which are not only under the guidance
of similarity matrix S, but also mutually separated at least by a minimal distance 𝑑 that can be computed by the
Gibert-Varshamov bound (see Theorem 3.1).
3.2.1 The Minimal Distance Calculation. The Gilbert-Varshamov bound lets us know that for 𝐶 hash centers, the
Hamming distance between any two of them is larger than or at least equal to 𝑑.
Theorem 3.1 (Gilbert-V arshamov Bound). For 𝐶 𝑞 -bit binary vectors h𝑖 ∈ {− 1, +1}𝑞 (1 ≤ 𝑖 ≤ 𝐶), set the minimal
Hamming distance of any two vectors as 𝑑, there is the Gilbert-Varshamov bound:
2𝑞
𝐶 ≤
𝑑 −1∑︁
𝑖=0
𝑞
𝑖

. (13)
Proof. This theorem has been proven by Ref. [34], which in fact provides a method to calculate the minimal distance
𝑑 given 𝐶 and 𝑞. □
To be specific, since 𝑑 is an integer in {1, 2, ..., 𝑞}, and according to Eq. (13), we can compute it easily by exhaustive
search [37].
3.2.2 Optimization Objective. For all images of 𝐶 classes, we consider to generate 𝐶 hash centers {h1, h2, · · ·, h𝐶 } with
h𝑖 corresponding to the 𝑖-th class. Clearly, if the 𝑖-th and 𝑗-th class are more semantically similar, then their cosine
score between h𝑖 and h𝑗 is much closer to s𝑖 𝑗, which can be formulated as:
min
h𝑖,𝑖=1,···,𝐶
L𝑠𝑒𝑚𝑎𝑛𝑡𝑖𝑐 =
𝐶∑︁
𝑖=1
𝐶∑︁
𝑗=1
||s𝑖 𝑗 − 1
𝑞 h𝑇
𝑖 h𝑗 || 2
2
s.t. h𝑖 ∈ {− 1, +1}𝑞;
(14)
in addition, we want the distances between hash centers are as large as possible, and meanwhile the Hamming distance
between any two hash centers is larger than or at least equal to the minimal distance 𝑑, i.e.,
max
h𝑖,𝑖=1,···,𝐶
L𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 𝑜 =
𝐶∑︁
𝑖=1
∑︁
𝑗≠𝑖
||h𝑖 − h𝑗 || 𝐻𝑎𝑚𝑚𝑖𝑛𝑔
s.t. ||h𝑖 − h𝑗 || 𝐻𝑎𝑚𝑚𝑖𝑛𝑔 ≥ 𝑑, (𝑖 ≠ 𝑗).
(15)
By the following Theorem 3.2, the optimizaiton (15) is then converted to:
min
h𝑖,𝑖=1,···,𝐶
L𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 =
𝐶∑︁
𝑖=1
∑︁
𝑗≠𝑖
h𝑇
𝑖 h𝑗
s.t. h𝑇
𝑖 h𝑗 ≤ 𝑞 − 2𝑑, (1 ≤ 𝑖, 𝑗 ≤ 𝐶, 𝑗 ≠ 𝑖).
(16)
Manuscript submitted to ACM
8 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
Theorem 3.2 (Hamming-Euclid). For any two binary vectors h𝑖, h𝑗 ∈ {− 1, +1}𝑞, there is:
D(h𝑖, h𝑗 ) ≜ ||h𝑖 − h𝑗 || 𝐻𝑎𝑚𝑚𝑖𝑛𝑔 = 1
2 (𝑞 − h𝑇
𝑖 h𝑗 ), (17)
where || · ||𝐻𝑎𝑚𝑚𝑖𝑛𝑔 represents the number of non-zero elements for a given vector; for example, if x = [0, 0, −2, −2, +2 + 2]𝑇 ,
then ||x|| 𝐻𝑎𝑚𝑚𝑖𝑛𝑔 = 4.
Proof. Let’s first assume that:
x =h𝑖 − h𝑗
=[𝑥1, · · ·, 𝑥𝑠, · · ·, 𝑥𝑞]𝑇 ,
(18)
where 𝑥𝑠 (𝑠 = 1, · · ·, 𝑞) equals to 0 or -2 or +2. Denote the number of 0 and -2 and +2 in x as 𝑛0, 𝑛−2, 𝑛+2, respectively.
Then, there exists:
𝑞 = 𝑛0 + 𝑛−2 + 𝑛+2. (19)
Next, let’s do some derivations:
||h𝑖 − h𝑗 || 𝐻𝑎𝑚𝑚𝑖𝑛𝑔 = 𝑛−2 + 𝑛+2, (20)
and
1
2 (𝑞 − h𝑇
𝑖 h𝑗 ) = 1
2 (𝑞 − (𝑛0 − (𝑛−2 + 𝑛+2)))
= 1
2 ((𝑞 − 𝑛0) + ( 𝑛−2 + 𝑛+2))
= 1
2 ((𝑛−2 + 𝑛+2) + ( 𝑛−2 + 𝑛+2))
=𝑛−2 + 𝑛+2,
(21)
which finishes the proof of Eq. (17), building the connection between the Hamming space and the Euclid space. □
Taking the optimization (14) and (16) into considerations, we could obtain the overall optimization:
min
H
L = L𝑠𝑒𝑚𝑎𝑛𝑡𝑖𝑐 + 𝜇 × L𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒
= ||S − 1
𝑞 H𝑇 H|| 2
𝐹 + 𝜇
𝐶∑︁
𝑖=1
∑︁
𝑗≠𝑖
h𝑇
𝑖 h𝑗
s.t.
(
h𝑇
𝑖 h𝑗 ≤ 𝑞 − 2𝑑, (1 ≤ 𝑖, 𝑗 ≤ 𝐶, 𝑗 ≠ 𝑖),
h𝑖 ∈ {− 1, +1}𝑞,
(22)
where S=(s𝑖 𝑗)𝐶 ×𝐶 represents the data-dependent similarity matrix, H = [h1, h2, · · ·, h𝐶 ] denotes 𝐶 hash centers, || · || 𝐹
is the matrix’sF -norm, and 𝜇 is a hyper-parameter to balance the importance between the semantics and distances.
3.2.3 Alternating Optimization Procedure. The optimization (22) is a NP-hard problem because of the binary constraints
H ∈ {− 1, +1}𝑞×𝐶. A feasible optimization method is to relax the discrete constraints; however, it will result in sub-
optimal performance. Then, we devise a procedure that can optimize its variables in a mixed discrete-and-continuous
manner.
First, we introduce an auxiliary variable M that satisfies:
M = [m1, m2, · · ·, m𝐶 ] = H, (23)
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 9
where M ∈ R𝑞×𝐶. Then let k𝑖 𝑗 equals to 𝑞 − h𝑇
𝑖 h𝑗 − 2𝑑, i.e.,
k𝑖 𝑗 = 𝑞 − h𝑇
𝑖 h𝑗 − 2𝑑. (24)
Next, we replace one H with M in Eq. (22), and transform the optimization (22) into:
min
H,M
L = ||S − 1
𝑞 H𝑇 M|| 2
𝐹 + 𝜇
𝐶∑︁
𝑖=1
∑︁
𝑗≠𝑖
h𝑇
𝑖 h𝑗
s.t.
 

h𝑖 ∈ {− 1, +1}𝑞,
m𝑖 ∈ R𝑞,
h𝑖 = m𝑖, (𝑖 = 1, 2, ..., 𝐶),
𝑘𝑖 𝑗 = 𝑞 − h𝑇
𝑖 h𝑗 − 2𝑑,
𝑘𝑖 𝑗 ≥ 0,
(25)
which has both equality and inequality constraints; hence, we leverage the Augmented Lagrangian method [1] (ALM)
and then re-formulate the optimization (25) as:
min
H,M,K,Λ ,𝛼
L = ||S − 1
𝑞 H𝑇 M|| 2
𝐹 + 𝜇
𝐶∑︁
𝑖=1
∑︁
𝑗≠𝑖
h𝑇
𝑖 h𝑗
+
𝐶∑︁
𝑖=1
𝜆𝑇
𝑖 (h𝑖 − m𝑖 ) +
𝐶∑︁
𝑖=1
𝜌𝑖
2 ||h𝑖 − m𝑖 || 2
2
+
∑︁
𝑗≠𝑖
𝛼𝑖 𝑗 (𝑞 − 2𝑑 − h𝑇
𝑖 h𝑗 − k𝑖 𝑗)
+
∑︁
𝑗≠𝑖
𝛽𝑖 𝑗
2 ||𝑞 − 2𝑑 − h𝑇
𝑖 h𝑗 − k𝑖 𝑗 || 2
s.t.
(
h𝑖 ∈ {− 1, +1}𝑞,
k𝑖 𝑗 ≥ 0,
(26)
where K=(k𝑖 𝑗)𝐶 ×𝐶, Λ = [𝜆1, 𝜆2, · · ·, 𝜆𝐶 ], and 𝛼=(𝛼𝑖 𝑗)𝐶 ×𝐶 are learnable parameters, while 𝜇, 𝜌 = (𝜌1, 𝜌2, · · ·, 𝜌𝐶 ), and
𝛽 = (𝛽𝑖 𝑗)𝐶 ×𝐶 are non-negative hyper-parameters.
To solve the optimization problem (26), we adopt the alternating optimization strategy: we iteratively optimize the
variables {H, M, K, Λ , 𝛼} one by one, each time with the other variables fixed, until the objective function L converges.
Update m𝑖. To update m𝑖 with the other variables fixed, the sub-problem of L in Eq. (26) is an un-constrained
optimization. The gradient of L w.r.t. m𝑖 can be easily calculated as:
𝜕𝐿 (m𝑖 )
𝜕m𝑖
=
 2
𝑞2 HH𝑇 m𝑖 − 2
𝑞 Hs𝑖

+ (−𝜆𝑖 ) + ( 𝜌𝑖m𝑖 − 𝜌𝑖h𝑖 ).
(27)
By setting this gradient to zero, we can update m𝑖 by:
m𝑖 =
 2
𝑞2 HH𝑇 + 𝜌𝑖I
 −1  2
𝑞 Hs𝑖 + 𝜆𝑖 + 𝜌𝑖h𝑖

, (28)
Manuscript submitted to ACM
10 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
which can be written in matrix forms, i.e.,
M =
 2
𝑞2 HH𝑇 + 𝜌I
 −1  2
𝑞 HS + Λ + 𝜌hi

, (29)
where I denotes the identity matrix and 𝜌𝑖 = 𝜌.
Update k𝑖 𝑗. To update k𝑖 𝑗 with the other variables fixed, the sub-problem of L in Eq. (26) can be re-written as:
min
K
L = 𝛼𝑖 𝑗 (𝑞 − 2𝑑 − h𝑇
𝑖 h𝑗 − k𝑖 𝑗)
+ 𝛽𝑖 𝑗
2 ||𝑞 − 2𝑑 − h𝑇
𝑖 h𝑗 − k𝑖 𝑗 || 2
2
s.t. k𝑖 𝑗 ≥ 0, (𝑖 ≠ 𝑗).
(30)
The gradient of L w.r.t. k𝑖 𝑗 can be computed as:
𝜕L(k𝑖 𝑗)
𝜕k𝑖 𝑗
= 𝛽𝑖 𝑗 [(k𝑖 𝑗 − (𝑞 − 2𝑑 − hi𝑇 h𝑗 )] − 𝛼𝑖 𝑗 . (31)
By setting this gradient to zero, and taking its inequality constraint into consideration, we can update k𝑖 𝑗 by:
k𝑖 𝑗 = max {𝑞 − 2𝑑 − h𝑇
𝑖 h𝑗 + 𝛼𝑖 𝑗
𝛽𝑖 𝑗
, 0}, (32)
which can also be written in matrix forms, i.e,
K = max {𝑞 − 2𝑑 − H𝑇 H + 𝛼
𝛽 , 0}. (33)
Update h𝑖. To update h𝑖 with the other variables fixed, the sub-problem of L in Eq. (26) can be re-formulated as:
min
h𝑖
L =
𝐶∑︁
𝑖=1
||s𝑖 − 1
𝑞 M𝑇 h𝑖 || 2
2 + 𝜇
𝐶∑︁
𝑖=1
∑︁
𝑗≠𝑖
h𝑇
𝑖 h𝑗
+
𝐶∑︁
𝑖=1
𝜆𝑇
𝑖 (h𝑖 − m𝑖 ) +
𝐶∑︁
𝑖=1
𝜌𝑖
2 ||h𝑖 − m𝑖 || 2
2
+
𝐶∑︁
𝑖=1
∑︁
𝑗≠𝑖
𝛼𝑖 𝑗 (𝑞 − 2𝑑 − h𝑇
𝑖 h𝑗 − k𝑖 𝑗)
+
𝐶∑︁
𝑖=1
∑︁
𝑗≠𝑖
𝛽𝑖 𝑗
2 ||𝑞 − 2𝑑 − h𝑇
𝑖 h𝑗 − k𝑖 𝑗 || 2
2
s.t. h𝑖 ∈ {− 1, +1}𝑞 .
(34)
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 11
The gradient of L w.r.t. h𝑖 can be achieved by:
𝜕L(h𝑖 )
𝜕h𝑖
=
 2
𝑞2 MM𝑇 h𝑖 − 2
𝑞 Ms𝑖

+ ( 2𝜇
𝐶∑︁
𝑗=1,𝑗 ≠𝑖
hj)
+ (𝜆𝑇
𝑖 ) + 𝜌𝑖 (h𝑖 − m𝑖 ) + (− 2𝜇
𝐶∑︁
𝑗=1,𝑗 ≠𝑖
𝛼𝑖 𝑗h𝑗 )
+
𝐶∑︁
𝑗=1,𝑗 ≠𝑖
𝛽𝑖 𝑗 [2h𝑗 h𝑇
𝑗 h𝑖 − 2(𝑞 − 2𝑑 − k𝑖 𝑗h𝑗 )]
s.t. h𝑖 ∈ {− 1, +1}𝑞 .
(35)
Here, we utilize projected gradient descent (PGD) [3, 9] to update h𝑖 via:
h𝑖 = sign

h𝑖 − 1
𝜂
𝜕𝐿 (h𝑖 )
𝜕h𝑖

. (36)
where 𝜂 is a hyper-parameter.
Update 𝜆𝑖. To update 𝜆𝑖 with the other variables fixed, we can easily update 𝜆𝑖 via:
𝜆𝑖 = 𝜆𝑖 + 𝜌𝑖 (h𝑖 − m𝑖 ). (37)
Update 𝛼𝑖 𝑗. To update 𝛼𝑖 𝑗 with the other variables fixed, we can easily update 𝛼𝑖 𝑗 with:
𝛼𝑖 𝑗 = 𝛼𝑖 𝑗 + 𝛽𝑖 𝑗 (𝑞 − 2𝑑 − h𝑇
𝑖 h𝑗 − k𝑖 𝑗). (38)
The above designed optimization procedure is summarized in Algorithm 1.
3.3 Stage 3: Train the Deep Hashing Network
With the generated 𝐶 semantic hash centers {h𝑐 }𝐶
𝑐=1 as the supervised binary codes for images, we train a deep hashing
network that can convert image x𝑖 to its hash code b𝑖 (𝑖 = 1, 2, · · ·, 𝑁). As shown in Fig. 2 (Stage 3), the deep hashing
network has three components.
The first component is the CNN backbone, which is used to capture the deep representations of the images. Here, we
use ResNet34 [14] as the backbone, which is composed of a series of convolution layers and residual connections.
The second component is the hash layer, which is a fully-connected layer with the tanh(·) activation function that
can map the image representation of x𝑖 extracted by ResNet34 into a 𝑞-bit-like hash code b0𝑖, and its final hash code b𝑖
can be achieved via:
b𝑖 = sign(b0𝑖 ) ∈ {− 1, +1}𝑞 . (39)
Here, in the hash layer, we replace thesign(·) function with thetanh(·) function because the former is non-differentiable,
making it unsuitable for training neural networks via backpropagation [ 29]; while the latter is differentiable and
meanwhile can well approximate the former. Nevertheless, such an approximation can lead to quantization errors;
hence, in the subsequent loss function, we will incorporate a quantization loss to compensate for these errors.
The third component is the loss function, and we utilize CSQ’s [ 42] loss function, which consists of the central
similarity loss and the quantization loss. Since hash centers are binary vectors, we employ Binary Cross Entropy
Manuscript submitted to ACM
12 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
Algorithm 1: Semantic Hash Centers Generation
Input: Similarity matrix: S; Code length: 𝑞; Number of classes: 𝐶; Minimal distance: 𝑑; Number of Training
cycles: 𝑇 ; Hyper-parameters: 𝜇, 𝜌, 𝜂, 𝛽.
Output: Semantic Hash Centers: H = [h1, · · ·, h𝐶 ].
1 /*H has a minimum distance 𝑑*/;
2 Initialize hash centers {h1, · · ·, h𝐶 } by MDS [37];
3 while 𝑡 ≤ 𝑇 do
4 /*M is a proxy of H*/;
5 update M via Eq. (29);
6 /*K is used to constrain the minimal distance*/;
7 update K via Eq. (33);
8 /*column by column*/;
9 while 𝑖 ≤ 𝐶 do
10 /*—– h𝑖—–*/;
11 while 𝑙 ≤ 3 do
12 update h𝑖 via Eq. (36);
13 𝑙 ← 𝑙 + 1;
14 end
15 /*—– 𝜆𝑖—–*/;
16 update 𝜆𝑖 via Eq. (37);
17 /*—– 𝛼𝑖 𝑗—–*/;
18 update 𝛼𝑖 𝑗 via Eq. (38);
19 𝑖 ← 𝑖 + 1;
20 end
21 𝑡 ← 𝑡 + 1;
22 end
23 Return semantic hash centers H = [h1, · · ·, h𝐶 ].
(BCE) [28] to replace the Hamming distance between hash code and its corresponding hash center, i.e.,
D𝐻𝑎𝑚𝑚𝑖𝑛𝑔 (h𝑖, b0𝑖 ) ∝ BCE
 h𝑖 + 1
2 , b0𝑖 + 1
2

, (40)
which indicates that the much closer the Hamming distance between a hash code and its corresponding hash center,
the much smaller the value of the BCE function. Therefore, we can obtain the optimization objective of the central
similarity loss L𝐶:
L𝐶 = 1
𝑁
𝑁∑︁
𝑖=1
 1 + h𝑖
2 log 1 + b0𝑖
2 + 1 − h𝑖
2 log 1 − b0𝑖
2

. (41)
Besides, the output binary-like vectors should be as close as possible to binary codes, thus we introduce a quantization
loss L𝑄 to refine the generated approximate hash codes b0𝑖. Similar to DHN [46], we define:
L𝑄 =
𝑁∑︁
𝑖=1
||| b0𝑖 | − 1|| 2
2, (42)
where 1 ∈ R𝑞 is an all-one vector.
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 13
Table 1. Dataset Statistics.
Datasets #Training #Query #RetrievalDB
CIFAR-100 10,000 5,000 45,000
Stanford Cars-A 8,184 8,041 8,184
Stanford Cars-B 5,880 1,958 8,347
NABirds-A 24,633 23,929 24,633
NABirds-B 16,384 8,171 25,612
Finally, we can summarize the overall central similarity optimization problem as:
min
Θ
L = L𝐶 + 𝛾L𝑄, (43)
where Θ denotes all the trainable parameters of deep hashing network, and 𝛾 is the hyper-parameter.
4 Experiment
This section will display the effectiveness of our proposed SHC approach via a series of experiments and ablation
studies.
4.1 Datasets
We conduct experiments on three widely-used datasets, i.e., CIFAR-1001 [21], Stanford Cars2 [20], and NABirds3 [32],
based on which we curate 5 versions with different settings, described as below.
CIFAR-100. It contains 100 classes, and each class has 600 images. Based on this dataset, we randomly sample 10,000
images as the training set, 5,000 images as the query set, and the remaining 45,000 images as the retrieval database.
Stanford Cars. It holds 196 classes with 16,185 images in total. In its official website, it is divided almost evenly into
two parts, i.e., 8,144 training images and 8,041 query images, and the 8,144 training images are also used as the retrieval
databse. This version is marked as Stanford Cars-A.
By contrast, we randomly select 5,880, 1,958, and 8347 images as the training set, the query set, and the retrieval
database, respectively. This version is marked as Stanford Cars-B.
NABirds. It holds 555 classes, and has been partitioned into 24,633 training images and 23,929 query images; similar
to the Stanford Cars-A, the 24,633 training images are also treated as the retrieval database. This version is marked as
NABirds-A.
While we randomly choose 16,384 images for training, 8,171 images for queries, and 25,612 images as the retrieval
databse. This version is marked as NABirds-B.
It’s noteworthy that the version-B’s training samples are much smaller than the version-A’s, which implies that the
learning on version-B’s training set is much more challenging than that on version-A’s; besides, the training set, the
query set, and the retrieval database of version-B do not share any images, which is much more general and practical in
real-world scenarios. The specific statistics of these five curated datasets are collected in Table 1.
1https://www.cs.toronto.edu/~kriz/cifar.html
2https://datasets.activeloop.ai/docs/ml/datasets/stanford-cars-dataset/
3https://dl.allaboutbirds.org/nabirds
Manuscript submitted to ACM
14 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
4.2 Evaluation Measures
To evaluate different methods’ retrieval performances, we use four widely-used measures, i.e., Mean Average Precision
(MAP@topK), Precision@topK curves, Recall@topK curves, and Precision-Recall curves [5, 7, 8, 16, 42]. For MAPs, we
set topK to 100, 1000, and ALL (i.e., the total number of images in the retrieval database). In terms of Precision/Recall
curves, its topK is pre-defined as a large range from 1, 5, 10, · · ·, to 500.
Precision@topK refers to the proportion of images relevant to the query in the topK retrieved results. MAP@topK is
a metric based on Precision@topK over all query images, also employed to evaluate a method’s retrieval precisions.
Recall@topK refers to the ratio of successfully retrieved relevant images in the topK results among all the relevant
images in the database.
These two types of measures are often adopted to evaluate the performance of a retrieval system. However, there
is a trade-off effect between the two kinds of metrics. Generally, as topK increases, precision tends to decrease while
recall tends to rise. Thus, the precision-recall curves present an overall illustration, taking both precision and recall into
considerations, to assess a method’s retrieval performance.
4.3 Baseline Approaches
We compare our SHC approach with several state-of-the-art image hashing methods, including:
• DPSH4 [24] is the first deep network that can perform simultaneous feature and hash-code learning for retrievals
with pairwise labels.
• DTSH5 [38] puts forward a triplet based deep hashing method that aims to maximize the likelihood of the given
triplet labels.
• HashNet6 [5] learns binary hash codes by minimizing a novel weighted pairwise cross-entropy loss in deep
CNN architectures.
• GreedyHash7 [30] adopts a greedy principle to tackle the discrete constrained models by iteratively updating
the network toward the probable optimal discrete solution in each iteration.
• IDHN8 [44] develops a quantified similarity formula to assess the fine-grained pairwise similarity for supervising
hash-code generations.
• LTH9 [6] addresses the problem of learning to hash for more realistic datasets where the labels of given datasets
roughly exhibit a long-tail distribution.
• CSQ10 [42] proposes a new concept “Hash Center”, and then learns hash codes by optimizing the Hamming
distance between hash codes and corresponding hash centers.
• MDS11 [37] is an improved variant of CSQ, addressing the issue of how to enforce a minimum distance between
hash centers when it is not feasible to utilize Hadamard matrices to generate hash centers due to a large number
of categories.
4https://github.com/TreezzZ/DPSH_PyTorch
5https://github.com/Minione/DTSH
6https://github.com/thuml/HashNet
7https://github.com/ssppp/GreedyHash
8https://github.com/pectinid16/IDHN
9https://github.com/butterfly-chinese/long-tail-hashing
10https://github.com/yuanli2333/Hadamard-Matrix-for-hashing
11https://github.com/Wangld5/Center-Hashing
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 15
Table 2. Accuracies of the pre-trained classification network on various datasets.
Datasets Accuracy
CIFAR-100 66.72%
Stanford Cars-A 73.49%
Stanford Cars-B 62.56%
NABirds-A 52.14%
NABirds-B 48.44%
• OrthoHash12 [16] unifies training objectives of deep hashing by maximizing the cosine similarity between the
continuous codes and binary orthogonal target under a cross entropy loss.
Among the 9 deep hashing baselines, there are 3 methods using hash centers (i.e., OrthoHash[ 16], CSQ[42], and
MDS[37]), and 6 other deep hashing methods (i.e., LTH[6], IDHN[44], GreedyHash[30], HashNet[5], DTSH[38] and
DPSH[24]) using pointwise classification, pairwise or listwise semantic supervisions.
4.4 Experimental Settings
In stage 1, we utilize a pre-trained ResNet3413 [15] to extract image features, followed by a fully-connected layer to
project the image features to the class space. The softmax function is then applied to get the probability vectors for
classification. During training, we adopt the RMS optimizer with a cosine annealing schedule with an initial learning
rate of 7e-5 for CIFAR-100 and 1e-4 for other datasets, and set the batch size and epochs to 64 and 300, respectively.
In stage 2, we first adopt MDS [37] for initializing hash centers and then execute Algorithm 1 for generating the
semantic hash centers. With respect to parameters 𝛼 and 𝜆, we set their initial values to 0𝐶 ×𝐶 and 0.1𝑞×𝐶, respectively.
For other parameters, we set 𝜌 = 0.2, 𝜇 = 0.625, 𝛽 = (1e − 6)𝐶 ×𝐶, and 𝜂 = 0.5. Besides, we set the training cycles 𝑇 to
20, and perform 3 inner iterations to ensure convergence when optimizing the parameter H.
In stage 3, we leverage a pre-trained ResNet34 to extract image features, followed by a FC layer with the tanh(·)
activation function to approximate the binary output. During training, we use the RMS optimizer with a cosine annealing
learning rate schedule. For the CIFAR-100 dataset, the initial learning rate is set to 7e-5, while for other datasets, it is set
to 1e-4. Besides, we set the batch size, epochs, and 𝛾 (Eq. (43)) to 64, 300, and 1e-4, respectively.
To ensure a fair comparison across different approaches in the experiments, we employ the pre-trained ResNet-34 as
the backbone for all competitors. For the selected baseline approaches, their parameters are all set in accordance with
their corresponding papers.
All the curated datasets and codes will be released on Github.
4.5 Results and Analysis
This part will exhibit experimental results and some analysis stage by stage.
4.5.1 Accuracies of Pre-trained Classification Network. To yield semantic hash centers, we first need to pre-train
a classification network, which is a ResNet34 based classifier trained via the cross-entropy loss; after training, its
accuracies on the testing sets of various datasets are collected in Table 2.
12https://github.com/kamwoh/orthohash
13https://pytorch.org/vision/0.12/generated/torchvision.models.resnet34.html
Manuscript submitted to ACM
16 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
From Table 2, we can see that the classification accuracies on all datasets are not very high, but it does not affect our
method to achieve good retrieval performance (see below). Although many images are classified into the wrong classes,
we only focus on the mean (center) of all images within each class, and hence these classfication errors can be diluted
out under the most correctly-classified images.
4.5.2 Effectiveness of Generated Semantic Hash Centers. By conducting the optimization Algorithm 1 in Section 3.2, we
can obtain semantic hash centers; then we compare our method with three other approaches, i.e., random-generated
hash centers [42], hash centers generated using Hadamard matrices [42], and hash centers generated using the Gilbert-
Varshamov bound [37]. We compare these methods in terms of the minimal Hamming distance (𝑑𝑚𝑖𝑛) between any two
hash centers, and the pairwise similarity losses (𝑆𝑙𝑜𝑠𝑠 ) of the generated hash centers, which are formulated as follows:
𝑑𝑚𝑖𝑛 = min{ 𝑞 − H𝑇 H ⊙ ( 1 − I)
2 }, (44)
𝑆𝑙𝑜𝑠𝑠 = L𝑠𝑒𝑚𝑎𝑛𝑡𝑖𝑐 = ||S − 1
𝑞 H𝑇 H|| 2
𝐹 ; (45)
and obviously a smaller 𝑆𝑙𝑜𝑠𝑠 indicates a better preservation of semantics between classes.
Table 3 displays the minimal distances and the pairwise similarity losses of the generated hash centers w.r.t. different
methods. Evidently, our method achieves the largest 𝑑𝑚𝑖𝑛 and the smallest 𝑆𝑙𝑜𝑠𝑠 across all datasets, which validates the
effectiveness of our hash centers generation algorithm, i.e., not only well keeping semantics preserved but also setting
hash centers apart as far as possible.
Table 3. Comparisons of different hash center generation methods w.r.t. the minimal distance 𝑑𝑚𝑖𝑛 and the similarity loss 𝑆𝑙𝑜𝑠𝑠 .
Datasets Methods 16 bits 32 bits 64 bits
𝑑𝑚𝑖𝑛 𝑆𝑙𝑜𝑠𝑠 𝑑𝑚𝑖𝑛 𝑆𝑙𝑜𝑠𝑠 𝑑𝑚𝑖𝑛 𝑆𝑙𝑜𝑠𝑠
CIFAR-100
random 0 0.1094 5 0.0767 17 0.0612
CSQ 0 0.1074 4 0.0697 32 0.0528
MDS 4 0.0993 10 0.0698 32 0.0528
SHC 4 0.0545 10 0.0268 24 0.0321
Stanford Cars-A
random 0 0.0876 3 0.0569 14 0.0413
CSQ 0 0.0907 4 0.0558 14 0.0379
MDS 4 0.0831 10 0.0534 23 0.0375
SHC 4 0.0623 10 0.0308 23 0.0205
Stanford Cars-B
random 0 0.0909 3 0.0592 14 0.0444
CSQ 0 0.0932 4 0.0583 14 0.0410
MDS 4 0.0861 10 0.0557 23 0.0410
SHC 4 0.0706 10 0.0301 23 0.0237
NABirds-A
random 0 0.0720 3 0.0406 13 0.0252
CSQ 0 0.0754 4 0.0410 16 0.0249
MDS 3 0.0702 9 0.0393 21 0.0242
SHC 3 0.0664 9 0.0316 21 0.0140
NABirds-B
random 0 0.0725 3 0.0413 13 0.0258
CSQ 0 0.0761 4 0.0417 16 0.0255
MDS 3 0.0710 9 0.0398 21 0.0248
SHC 3 0.0668 9 0.0315 21 0.0144
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 17
Table 4. Different methods’ mAP@topK=100 values on CIFAR-100, Stanford Cars, and NABirds datasets (the code length is set to 16,
32 and 64 bits).
Methods
CIFAR-100 Stanford Cars-A Stanford Cars-B NABirds-A NABirds-B
16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits
DPSH 0.1572 0.3541 0.5054 0.0308 0.0533 0.1421 0.0574 0.0989 0.1264 0.0107 0.0150 0.0157 0.0117 0.0191 0.0403
DTSH 0.4939 0.5416 0.5898 0.4236 0.5425 0.6589 0.3533 0.4781 0.5339 0.1237 0.1645 0.2422 0.0145 0.1487 0.1811
HashNet 0.2085 0.4115 0.3914 0.1216 0.2641 0.2554 0.0644 0.2134 0.2134 0.0566 0.1737 0.1902 0.0385 0.1232 0.1661
GreedyHash 0.5690 0.6013 0.5941 0.5998 0.6686 0.6962 0.3966 0.4942 0.5375 0.4687 0.5545 0.5886 0.3264 0.4115 0.4541
IDHN 0.1127 0.2839 0.4883 0.0293 0.0433 0.3322 0.0444 0.1326 0.2728 0.0139 0.0137 0.0151 0.0148 0.0247 0.0401
CSQ 0.6123 0.6559 0.6620 0.5401 0.6827 0.7342 0.3219 0.4895 0.5853 0.4540 0.5699 0.6243 0.2956 0.4309 0.4984
MDS 0.6199 0.6519 0.6649 0.5866 0.6801 0.7396 0.3238 0.4804 0.5697 0.4812 0.5791 0.6248 0.3247 0.4359 0.4990
LTH 0.5687 0.6171 0.6405 0.5753 0.6612 0.7157 0.3768 0.4915 0.5406 0.4461 0.5572 0.5926 0.2861 0.4004 0.4521
OrthoHash 0.5709 0.5871 0.5866 0.4824 0.5726 0.6064 0.2762 0.3888 0.4496 0.4583 0.5152 0.5242 0.2721 0.3420 0.3501
SHC 0.6360 0.6629 0.6672 0.6674 0.7448 0.7776 0.4440 0.5859 0.6288 0.4986 0.6117 0.6586 0.3552 0.4753 0.5289
Table 5. Different methods’ mAP@topK=1000 values on CIFAR-100, Stanford Cars, and NABirds datasets (the code length is set to 16,
32 and 64 bits).
Methods
CIFAR-100 Stanford Cars-A Stanford Cars-B NABirds-A NABirds-B
16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits
DPSH 0.1423 0.2991 0.4332 0.0262 0.0255 0.0809 0.0329 0.0483 0.0606 0.0124 0.0148 0.0143 0.0132 0.0179 0.0242
DTSH 0.4260 0.4688 0.5420 0.3898 0.5158 0.6468 0.2789 0.4002 0.4509 0.0704 0.1032 0.1748 0.0519 0.0812 0.1057
HashNet 0.2032 0.3582 0.3395 0.1039 0.2404 0.2186 0.0586 0.1572 0.1626 0.0582 0.1590 0.1357 0.0424 0.0950 0.1023
GreedyHash 0.5230 0.5619 0.5574 0.5858 0.6598 0.6895 0.3112 0.4104 0.4659 0.4528 0.5463 0.5843 0.2589 0.3361 0.3914
IDHN 0.0983 0.2378 0.4208 0.0201 0.0224 0.2990 0.0257 0.0884 0.2065 0.0145 0.0134 0.0143 0.0164 0.0197 0.0226
CSQ 0.5686 0.6113 0.6117 0.5200 0.6742 0.7295 0.2406 0.4026 0.5064 0.4427 0.5633 0.6213 0.2279 0.3562 0.4297
MDS 0.5775 0.6074 0.6167 0.5732 0.6722 0.7358 0.2426 0.3961 0.4959 0.4736 0.5743 0.6234 0.2536 0.3640 0.4372
LTH 0.5255 0.5718 0.5859 0.5568 0.6485 0.7093 0.2882 0.4012 0.4589 0.4248 0.5464 0.5884 0.2166 0.3195 0.3715
OrthoHash 0.4869 0.4955 0.5028 0.4327 0.5241 0.5563 0.1908 0.2885 0.3426 0.4215 0.4785 0.4840 0.1865 0.2494 0.2543
SHC 0.5936 0.6166 0.6170 0.6525 0.7379 0.7762 0.3541 0.5033 0.5573 0.4878 0.6072 0.6608 0.2813 0.3991 0.4575
4.5.3 Retrieval Performances of Different Approaches. Table 4, 5, and 6 present the MAP@topK (topK=100, 1000, and
ALL) values of our SHC approach and 9 baselines on 5 datasets with 3 different hash code lengths (i.e., 16, 32, and 64
bits). In all the tables, the best results are highlighted in bold, while the second-best are underlined. Obviously, SHC
performs better than other methods, achieving an improvement of (+0.13 % ∼ +11.53%), (+5.14 % ∼ +13.77%), (+7.43 %
Manuscript submitted to ACM
18 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
Table 6. Different methods’ mAP@topK=ALL values on CIFAR-100, Stanford Cars, and NABirds datasets (the code length is set to 16,
32 and 64 bits).
Methods
CIFAR-100 Stanford cars-A Stanford cars-B NABirds-A NABirds-B
16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits
DPSH 0.0833 0.2403 0.3697 0.0128 0.0130 0.0587 0.0199 0.0300 0.0383 0.0059 0.0062 0.0059 0.0056 0.0061 0.0060
DTSH 0.3497 0.3914 0.4513 0.3850 0.5120 0.6418 0.2573 0.3792 0.4281 0.0547 0.0918 0.1653 0.0318 0.0603 0.0843
HashNet 0.1487 0.3041 0.2930 0.0996 0.2383 0.2174 0.0470 0.1441 0.1519 0.0562 0.1557 0.1347 0.0342 0.0754 0.0872
GreedyHash 0.3809 0.4204 0.4238 0.5764 0.6497 0.6802 0.2406 0.3320 0.3929 0.4407 0.5330 0.5721 0.1646 0.2259 0.2824
IDHN 0.0642 0.1797 0.3652 0.0094 0.0112 0.2907 0.0137 0.0698 0.1787 0.0077 0.0068 0.0062 0.0072 0.0077 0.0066
CSQ 0.4272 0.4875 0.5185 0.5087 0.6651 0.7233 0.1785 0.3301 0.4462 0.4311 0.5535 0.6146 0.1387 0.2462 0.3244
MDS 0.4388 0.4856 0.5205 0.5613 0.6597 0.7273 0.1837 0.3251 0.4313 0.4627 0.5650 0.6155 0.1597 0.2542 0.3313
LTH 0.3890 0.4544 0.4980 0.5449 0.6395 0.7032 0.2221 0.3332 0.4065 0.4066 0.5350 0.5811 0.1334 0.2380 0.3135
OrthoHash 0.3813 0.3948 0.4062 0.4225 0.5154 0.5491 0.1572 0.2537 0.3147 0.4121 0.4702 0.4753 0.1403 0.2069 0.2175
SHC 0.4838 0.5186 0.5262 0.6433 0.7300 0.7720 0.2976 0.4547 0.5139 0.4755 0.6001 0.6566 0.1926 0.3185 0.3988
∼ +22.64%), (+2.77 % ∼ +6.68%), (+4.64 % ∼ +25.30%) over the second-best competitor on CIFAR-100, Stanford Cars-A,
Stanford Cars-B, NABirds-A, and NABirds-B datasets, respectively.
To be specific, the improvements brought by our SHC on CIFAR-100 are not as significant as those on the other two
datasets. This is attributed to the fact that the CIFAR-100 dataset has fewer categories, allowing for larger Hamming
distances between hash centers. In other words, the Hamming space is still relatively “spacious”. By the way, as the hash
code length becomes larger, the Hamming space becomes much more “spacious”, leading to a diminishing improvement.
On the Stanford Cars dataset, the largest improvement is observed with a hash code length of 32 bits. This could be
because, at 16 bits, the Hamming space becomes too “crowded”, leaving insufficient room for semantic adjustments
by the hash centers, resulting in a decrease in improvement. On the other hand, when the hash code length is 64
bits, the Hamming space becomes too “spacious”, similar to the CIFAR-100 dataset, which also leads to a decrease
in improvement. However, a 32-bit Hamming space strikes the right balance for the Stanford Cars dataset with 196
categories, achieving the best optimization results.
In our experiments on the NABirds dataset, which has the largest number of categories, the improvements increase
with the hash code length. This is consistent with the above analysis, where the 16-bit and 32-bit Hamming spaces are
too “crowded” (555 classes); however, a 64-bit Hamming space allows for better semantic optimization by the hash
centers without being overly “spacious”.
Fig. 3 shows the retrieval performance w.r.t. Precision-Recall curves on five datasets. Our method exhibits the optimal
PR curves on almost all datasets. On the CIFAR-100 dataset, as the length of the hash codes increases, the advantage of
our method in terms of the PR curve gradually diminishes. On the Stanford Cars dataset, our method demonstrates the
most significant advantage in the PR curve when the hash code length is 32 bits. On the NABirds dataset, the advantage
of our method in the PR curve becomes more pronounced with the increase in hash code length. These observations
align with the above analysis presented in Section 4.5.3.
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 19
Fig. 4 shows the retrieval performance w.r.t. Precision-topK curves on five datasets. The Precision-topK curves of our
method are optimal on almost all datasets, and the trend of the curves aligns with the theoretical expectations described
in Section 4.2, where precision gradually decreases as topK increases. However, in the case of the NABirds dataset with
a hash code length of 16 bits, our method’s precision curve falls slightly behind. This could be attributed to the fact that
for the NABirds dataset with a large number of categories, a lower hash code length may not provide sufficient distance
between hash centers, resulting in a decrease in the precision curve.
Fig. 5 shows the retrieval performance w.r.t. Recall-topK curves on five datasets. On most datasets, the Recall-topK
curves of our method are the best, and the trend of the curves aligns with the theoretical expectations described in
Section 4.2, where recall gradually increases as topK increases. However, on the Stanford Cars dataset, our method’s
Recall curves are not always optimal. This could be attributed to the composition of the dataset, where the training data
we used is relatively small, and the differences between categories in the dataset may not be significant, which could
have a negative impact on Recall.
In the case of the NABirds-B dataset with a hash code length of 16 bits, our method’s Recall curve slightly falls
behind. The reason could be similar to the precision curve, where a lower hash code length may not provide sufficient
distance between hash centers for a dataset with a large number of categories like NABirds, resulting in a decrease in
the Recall curve.
Manuscript submitted to ACM
20 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
(a) 16 bits, CIFAR-100
 (b) 32 bits, CIFAR-100
 (c) 64 bits, CIFAR-100
(d) 16 bits, Stanford Cars-A
 (e) 32 bits, Stanford Cars-A
 (f) 64 bits, Stanford Cars-A
(g) 16 bits, Stanford Cars-B
 (h) 32 bits, Stanford Cars-B
 (i) 64 bits, Stanford Cars-B
(j) 16 bits, NABirds-A
 (k) 32 bits, NABirds-A
 (l) 64 bits, NABirds-A
(m) 16 bits, NABirds-B
 (n) 32 bits, NABirds-B
 (o) 64 bits, NABirds-B
Fig. 3. The Precision-Recall curves of different methods on CIFAR100, Stanford Cars, and NABirds datasets (16, 32, and 64 bits).
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 21
(a) 16 bits, CIFAR-100
 (b) 32 bits, CIFAR-100
 (c) 64 bits, CIFAR-100
(d) 16 bits, Stanford Cars-A
 (e) 32 bits, Stanford Cars-A
 (f) 64 bits, Stanford Cars-A
(g) 16 bits, Stanford Cars-B
 (h) 32 bits, Stanford Cars-B
 (i) 64 bits, Stanford Cars-B
(j) 16 bits, NABirds-A
 (k) 32 bits, NABirds-A
 (l) 64 bits, NABirds-A
(m) 16 bits, NABirds-B
 (n) 32 bits, NABirds-B
 (o) 64 bits, NABirds-B
Fig. 4. The Precision@topK of different methods on CIFAR100, Stanford Cars, and NABirds datasets (16, 32, and 64 bits).
Manuscript submitted to ACM
22 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
(a) 16 bits, CIFAR-100
 (b) 32 bits, CIFAR-100
 (c) 64 bits, CIFAR-100
(d) 16 bits, Stanford Cars-A
 (e) 32 bits, Stanford Cars-A
 (f) 64 bits, Stanford Cars-A
(g) 16 bits, Stanford Cars-B
 (h) 32 bits, Stanford Cars-B
 (i) 64 bits, Stanford Cars-B
(j) 16 bits, NABirds-A
 (k) 32 bits, NABirds-A
 (l) 64 bits, NABirds-A
(m) 16 bits, NABirds-B
 (n) 32 bits, NABirds-B
 (o) 64 bits, NABirds-B
Fig. 5. The Recall@topK of different methods on CIFAR100, Stanford Cars, and NABirds datasets (16, 32, and 64 bits).
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 23
4.6 Ablation Studies
We conduct ablation experiments by removing each component individually, i.e., the semantic and distance constraints,
and also explore how the widely-used BERT-based or our data-dependent similarity matrices affect the final retrieval
performances of the proposed SHC method.
4.6.1 Effectiveness of the Minimal Distance Constraint. To demonstrate the effectiveness of the distance constraint in
optimization (22), we remove L𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 and then there comes:
min
H
L = ||S − 1
𝑞 H𝑇 H|| 2
𝐹
s.t. H ∈ {− 1, +1}𝑞×𝐶 .
(46)
Similar to the optimization method used in Section 3.2, we first rewrite the optimization (46) as follows:
min
H
L = ||S − 1
𝑞 H𝑇 M|| 2
𝐹 + 𝜇||H − M|| 2
𝐹
s.t.
(
H ∈ {− 1, +1}𝑞×𝐶,
M ∈ R𝑞×𝐶 .
(47)
We can tackle the above problem by iteratively fixing H and optimizing M, and then fixing M and optimizing H.
Update M:
M =
 1
𝑞 S + 𝜇I

H
 1
𝑞2 H𝑇 H + 𝜇I

; (48)
Update H:
H = sign

( 1
𝑞 S + 𝜇I)M(M𝑇 M) −1

. (49)
Now we obtained the hash centers without the distance constraint, i.e., SHC without L𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 . The experimental
results are collected in Table 7.
By comparing the MAP values of SHC and “SHC without L𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 ”, it is evident that SHC, with both semantic and
distance constraints, outputs larger than in most cases or at least comparable MAP values with “SHC withoutL𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒 ”,
only with semantic constraint, which validates the effectiveness of the distance constraint.
4.6.2 Effectiveness of the Semantic Constraint. To demonstrate the effectiveness of the semantic constraint in opti-
mization (22), we remove L𝑠𝑒𝑚𝑎𝑛𝑡𝑖𝑐 , which is equivalent to MDS [37]. The experimental results are collected in Table
7.
By comparing the MAP values of SHC and MDS, it is evident that SHC, with both semantic and distance constraints,
outputs larger MAP values than MDS, only with distance constraint, which validates the effectiveness of the semantic
constraint.
4.6.3 Effectiveness of Different Similarity Matrices. We also compare the MAP results of the method using the BERT [12]-
based similarity matrix and our SHC using the data-dependent similarity matrix.
The process of generating the similarity matrix using BERT is as below. For each class, its label is denoted as 𝑙𝑐
(𝑐 = 1, 2, · · ·, 𝐶), then we can get its semantic embedding via the pre-trained BERT model via e𝑐 = BERT(𝑙𝑐 ), based on
which, the similarity between any two classes s𝑖 𝑗 can be computed by their cosine values, i.e.,
s𝑖 𝑗 = cos(e𝑖, e𝑗 ) =
e𝑇
𝑖 e𝑗
||e𝑖 || × || e𝑗 || ; (50)
Manuscript submitted to ACM
24 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
then the similarity matrix S = (s𝑖 𝑗)𝐶 ×𝐶 based on BERT is prepared (i.e., each s𝑖 𝑗 is normalized to (-1,+1)) for subsequent
semantic hash centers generation, after which it follows the same process as SHC. These two methods are marked as
“BERT” v.s. “SHC”. Their experimental results are collected in Table 8.
Obviously, the BERT-based similarity matrix calculated with class labels is not that effective as our data-dependent
similarity matrix using the inherent distributions of the images. This is because that class labels are relatively static
compared with specific image datasets, and they can not adapt well to dynamic image datasets (e.g., for fixed class
labels, we can curate various distributed image datasets); in addition, for each class, it can be labeled with various texts,
this also can lead to different similarities between classes but with the same image datasets.
In sharp contrast, our designed data-dependent similarity matrix is just based on all the image samples, free from
class labels, and hence can well automatically adapt to the distribution of various datasets.
Table 7. The MAP@topK (topK=100, 1000, and ALL) values of our SHC approach w/o semantic L𝑠𝑒𝑚𝑎𝑛𝑡𝑖𝑐 or distance L𝑑𝑖𝑠𝑡𝑎𝑛𝑐𝑒
components. Note that d with ✓means the distance constraint; likewise, S with ✓means the semantic constraint.
MAP@topK d S
CIFAR-100 Stanford Cars-A Stanford Cars-B NABirds-A NABirds-B
16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits
MAP@100
✓ 0.6128 0.6495 0.6579 0.5804 0.6787 0.7393 0.3434 0.4828 0.5907 0.4837 0.5809 0.6234 0.3187 0.4401 0.4903
✓ 0.6337 0.6580 0.6726 0.6268 0.7280 0.7647 0.3999 0.5712 0.6025 0.4926 0.5921 0.6563 0.3429 0.4410 0.5437
✓ ✓ 0.6360 0.6629 0.6672 0.6674 0.7448 0.7776 0.4440 0.5859 0.6288 0.4986 0.6117 0.6586 0.3552 0.4753 0.5289
MAP@1000
✓ 0.5692 0.6064 0.6087 0.5654 0.6698 0.7355 0.2678 0.4026 0.5194 0.4756 0.5761 0.6218 0.2466 0.3678 0.4290
✓ 0.5830 0.6117 0.6208 0.6100 0.7199 0.7632 0.3129 0.4818 0.5274 0.4827 0.5869 0.6567 0.2669 0.3655 0.4646
✓ ✓ 0.5936 0.6166 0.6175 0.6525 0.7379 0.7762 0.3541 0.5033 0.5573 0.4878 0.6072 0.6608 0.2813 0.3991 0.4575
MAP@ALL
✓ 0.4308 0.4826 0.5158 0.5509 0.6566 0.7280 0.2012 0.3260 0.4518 0.4646 0.5670 0.6135 0.1568 0.2590 0.3215
✓ 0.4602 0.5055 0.5209 0.5989 0.7127 0.7565 0.2501 0.4280 0.4792 0.4720 0.5789 0.6521 0.1774 0.2577 0.3950
✓ ✓ 0.4838 0.5186 0.5262 0.6433 0.7300 0.7720 0.2976 0.4547 0.5139 0.4755 0.6001 0.6566 0.1926 0.3185 0.3988
4.7 Convergence Analysis
Fig. 6 exhibits the changes of losses and MAP values w.r.t. the training epochs on the curated datasets. Obviously, we
could observe that across all datasets, the losses on the whole decrease as the number of training epochs increases.
Owing to the utilization of cosine annealing learning rate14 in our experiments, the initial learning rate is relatively
large. As the training epochs progressively rise up, the learning rate gradually decreases; therefore, the losses exhibit
fluctuations in the early epochs, but as the learning rate becomes much smaller, the losses become more stable.
The trends of MAP values align with these losses, showing a gradual increase overall. During the training when the
losses fluctuate, the MAPs also exhibit fluctuations. As the learning rate decreases, the MAP values tend to stabilize.
14https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 25
(a) 16 bits, CIFAR-100
 (b) 32 bits, CIFAR-100
 (c) 64 bits, CIFAR-100
(d) 16 bits, Stanford Cars-A
 (e) 32 bits, Stanford Cars-A
 (f) 64 bits, Stanford Cars-A
(g) 16 bits, Stanford Cars-B
 (h) 32 bits, Stanford Cars-B
 (i) 64 bits, Stanford Cars-B
(j) 16 bits, NABirds-A
 (k) 32 bits, NABirds-A
 (l) 64 bits, NABirds-A
(m) 16 bits, NABirds-B
 (n) 32 bits, NABirds-B
 (o) 64 bits, NABirds-B
Fig. 6. The mAP@topK (topK=100, 1000, ALL) and loss curves w.r.t. training iterations on CIFAR100, Stanford Cars, and NABirds
datasets. Manuscript submitted to ACM
26 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
Table 8. Comparisons of the retrieval performance between the BERT-based and data-dependent similarity matrices for semantic
hash centers generation.
MAP@topK S type
CIFAR-100 Stanford Cars-A Stanford Cars-B NABirds-A NABirds-B
16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits 16 bits 32 bits 64 bits
MAP@100
BERT 0.6154 0.6497 0.6620 0.5911 0.6798 0.7348 0.3384 0.4830 0.5673 0.4509 0.5279 0.6222 0.3189 0.4427 0.4976
SHC 0.6360 0.6629 0.6672 0.6674 0.7448 0.7776 0.4440 0.5859 0.6280 0.4986 0.6117 0.6586 0.3552 0.4753 0.5289
MAP@1000
BERT 0.5664 0.6025 0.6151 0.5757 0.6712 0.7297 0.2557 0.3950 0.4851 0.4445 0.5699 0.6225 0.2498 0.3720 0.4307
SHC 0.5936 0.6166 0.6175 0.6525 0.7379 0.7762 0.3541 0.5033 0.5573 0.4878 0.6072 0.6608 0.2813 0.3991 0.4575
MAP@ALL
BERT 0.4263 0.4772 0.5096 0.5625 0.6593 0.7212 0.1965 0.3227 0.4196 0.4315 0.5509 0.6151 0.1613 0.2659 0.3370
SHC 0.4838 0.5186 0.5262 0.6433 0.7300 0.7720 0.2976 0.4547 0.5139 0.4755 0.6001 0.6566 0.1926 0.3185 0.3988
5 Conclusion
In this paper, we expand upon the newly emerged concept of “hash centers” to develop the more explainable “semantic
hash centers”, proposing a novel deep hashing approach named SHC, structured around a three-stage learning procedure.
First, we develop a data-dependent, rather than label-based, pairwise similarity calculation method that can well adapt
to various data distributions. Second, we not only impose semantic constraints on the hash centers, but also set them
apart as far as possible, and meanwhile compute a lower bound for the minimal distance between any two hash centers.
Through the strategic use of proxy variables and by utilizing the ALM optimization framework, we are able to gradually
decompose the complex optimization problem into several manageable sub-problems and solve them efficiently. Third,
we employ a classic deep hashing network that is supervised using the semantic hash centers derived above. Extensive
experiments on multiple datasets demonstrate that our SHC method outperforms several current state-of-the-art deep
hashing techniques in image retrieval tasks. Furthermore, ablation studies confirm the effectiveness of our devised
data-dependent similarity matrix and the utilized optimization algorithm for generating semantic hash centers.
Acknowledgments
This work is supported in part by National Natural Science Foundation of China (Grant No. 62372054, 62006005) and
National Key Research and Development Program of China (Grant No. 2022YFC3302200).
References
[1] Ernesto G Birgin and José Mario Martínez. 2014. Practical augmented Lagrangian methods for constrained optimization . SIAM.
[2] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry,
Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey
Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam
McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. In NeurIPS.
[3] Jian-Feng Cai, Tianming Wang, and Ke Wei. 2018. Spectral Compressed Sensing via Projected Gradient Descent. SIAM J. Optim. 28, 3 (2018),
2625–2653.
[4] Yue Cao, Mingsheng Long, Bin Liu, and Jianmin Wang. 2018. Deep cauchy hashing for hamming space retrieval. In CVPR. 1229–1237.
[5] Zhangjie Cao, Mingsheng Long, Jianmin Wang, and Philip S Yu. 2017. Hashnet: Deep learning to hash by continuation. In ICCV. 5608–5617.
[6] Yong Chen, Yuqing Hou, Shu Leng, Qing Zhang, Zhouchen Lin, and Dell Zhang. 2021. Long-tail hashing. In SIGIR. 1328–1338.
Manuscript submitted to ACM
Deep Hashing with Semantic Hash Centers for Image Retrieval 27
[7] Yong Chen, Zhibao Tian, Hui Zhang, Jun Wang, and Dell Zhang. 2020. Strongly Constrained Discrete Hashing. IEEE Trans. Image Process. 29 (2020),
3596–3611.
[8] Yong Chen, Hui Zhang, Zhibao Tian, Jun Wang, Dell Zhang, and Xuelong Li. 2022. Enhanced Discrete Multi-Modal Hashing: More Constraints Yet
Less Time to Learn. IEEE Trans. Knowl. Data Eng. 34, 3 (2022), 1177–1190.
[9] Woocheol Choi and Jimyeong Kim. 2023. On the convergence analysis of the decentralized projected gradient descent. CoRR abs/2303.08412 (2023).
[10] Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S Mirrokni. 2004. Locality-sensitive hashing scheme based on p-stable distributions. In
Proceedings of the twentieth annual symposium on Computational geometry . 253–262.
[11] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding. In NAACL-HLT. 4171–4186.
[12] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirectional transformers for language
understanding. arXiv preprint arXiv:1810.04805 (2018).
[13] Yunchao Gong, Svetlana Lazebnik, Albert Gordo, and Florent Perronnin. 2013. Iterative Quantization: A Procrustean Approach to Learning Binary
Codes for Large-Scale Image Retrieval. IEEE Trans. Pattern Anal. Mach. Intell. 35, 12 (2013), 2916–2929.
[14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep Residual Learning for Image Recognition. In CVPR. 770–778.
[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In CVPR. 770–778.
[16] Jiun Tian Hoe, Kam Woh Ng, Tianyu Zhang, Chee Seng Chan, Yi-Zhe Song, and Tao Xiang. 2021. One loss for all: Deep hashing with a single cosine
similarity based learning objective. In NeurIPS. 24286–24298.
[17] Piotr Indyk and Rajeev Motwani. 1998. Approximate nearest neighbors: towards removing the curse of dimensionality. In Proceedings of the thirtieth
annual ACM symposium on Theory of computing . 604–613.
[18] Hervé Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization for Nearest Neighbor Search. IEEE Trans. Pattern Anal. Mach. Intell.
33, 1 (2011), 117–128.
[19] Qing-Yuan Jiang and Wu-Jun Li. 2018. Asymmetric Deep Supervised Hashing. In AAAI. 3342–3349.
[20] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 2013. 3d object representations for fine-grained categorization. In ICCV. 554–561.
[21] Alex Krizhevsky, Geoffrey Hinton, et al. 2009. Learning multiple layers of features from tiny images. (2009).
[22] Brian Kulis and Kristen Grauman. 2009. Kernelized locality-sensitive hashing for scalable image search. In ICCV. 2130–2137.
[23] Hanjiang Lai, Yan Pan, Ye Liu, and Shuicheng Yan. 2015. Simultaneous feature learning and hash coding with deep neural networks. In CVPR.
3270–3278.
[24] Wu-Jun Li, Sheng Wang, and Wang-Cheng Kang. 2015. Feature learning based deep supervised hashing with pairwise labels. arXiv preprint
arXiv:1511.03855 (2015).
[25] Xiaoqing Liu, Huanqiang Zeng, Yifan Shi, Jianqing Zhu, Chih-Hsien Hsia, and Kai-Kuang Ma. 2023. Deep Cross-Modal Hashing Based on Semantic
Consistent Ranking. IEEE Trans. Multim. 25 (2023), 9530–9542.
[26] Xiao Luo, Haixin Wang, Daqing Wu, Chong Chen, Minghua Deng, Jianqiang Huang, and Xian-Sheng Hua. 2023. A Survey on Deep Hashing
Methods. ACM Trans. Knowl. Discov. Data 17, 1 (2023), 15:1–15:50.
[27] Mohammad Norouzi, David J Fleet, and Russ R Salakhutdinov. 2012. Hamming distance metric learning. NeurIPS, 1070–1078.
[28] Usha Ruby and Vamsidhar Yendapalli. 2020. Binary cross entropy with deep learning technique for image classification. Int. J. Adv. Trends Comput.
Sci. Eng 9, 10 (2020).
[29] David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. 1986. Learning representations by back-propagating errors. Nature 323 (1986),
533–536.
[30] Shupeng Su, Chao Zhang, Kai Han, and Yonghong Tian. 2018. Greedy hash: Towards fast optimization for accurate hash coding in cnn. (2018),
806–815.
[31] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew
Rabinovich. 2015. Going deeper with convolutions. In CVPR. 1–9.
[32] Grant Van Horn, Steve Branson, Ryan Farrell, Scott Haber, Jessie Barry, Panos Ipeirotis, Pietro Perona, and Serge Belongie. 2015. Building a bird
recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection. In CVPR. 595–604.
[33] Rom Rubenovich Varshamov. 1957. Estimate of the number of signals in error correcting codes. Docklady Akad. Nauk, SSSR 117 (1957), 739–741.
[34] Rom Rubenovich Varshamov. 1957. Estimate of the Number of Signals in Error Correcting Codes. Docklady Akad. Nauk, S.S.S.R. 177 (1957), 739–741.
[35] Jun Wang, Wei Liu, Sanjiv Kumar, and Shih-Fu Chang. 2015. Learning to hash for indexing big data—A survey. Proc. IEEE 104, 1 (2015), 34–57.
[36] Jingdong Wang, Ting Zhang, Jingkuan Song, Nicu Sebe, and Heng Tao Shen. 2018. A Survey on Learning to Hash. IEEE Trans. Pattern Anal. Mach.
Intell. 40, 4 (2018), 769–790.
[37] Liangdao Wang, Yan Pan, Cong Liu, Hanjiang Lai, Jian Yin, and Ye Liu. 2023. Deep hashing with minimal-distance-separated hash centers. In CVPR.
23455–23464.
[38] Xiaofang Wang, Yi Shi, and Kris M Kitani. 2017. Deep supervised hashing with triplet labels. In ACCV. 70–84.
[39] Rongkai Xia, Yan Pan, Hanjiang Lai, Cong Liu, and Shuicheng Yan. 2014. Supervised hashing for image retrieval via image representation learning.
In AAAI, Vol. 28.
[40] Erkun Yang, Tongliang Liu, Cheng Deng, Wei Liu, and Dacheng Tao. 2019. Distillhash: Unsupervised deep hashing by distilling data pairs. In CVPR.
2946–2955.
Manuscript submitted to ACM
28 Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang
[41] Ting Yao, Fuchen Long, Tao Mei, and Yong Rui. 2016. Deep Semantic-Preserving and Ranking-Based Hashing for Image Retrieval. In IJCAI.
3931–3937.
[42] Li Yuan, Tao Wang, Xiaopeng Zhang, Francis EH Tay, Zequn Jie, Wei Liu, and Jiashi Feng. 2020. Central similarity quantization for efficient image
and video retrieval. In CVPR. 3083–3092.
[43] Wanqian Zhang, Dayan Wu, Yu Zhou, Bo Li, Weiping Wang, and Dan Meng. 2021. Binary neural network hashing for image retrieval. InSIGIR.
1318–1327.
[44] Zheng Zhang, Qin Zou, Yuewei Lin, Long Chen, and Song Wang. 2019. Improved deep hashing with soft pairwise similarity for multi-label image
retrieval. IEEE Transactions on Multimedia 22, 2 (2019), 540–553.
[45] Fang Zhao, Yongzhen Huang, Liang Wang, and Tieniu Tan. 2015. Deep semantic ranking based hashing for multi-label image retrieval. In CVPR.
1556–1564.
[46] Han Zhu, Mingsheng Long, Jianmin Wang, and Yue Cao. 2016. Deep hashing network for efficient similarity retrieval. In AAAI, Vol. 30.
Manuscript submitted to ACM