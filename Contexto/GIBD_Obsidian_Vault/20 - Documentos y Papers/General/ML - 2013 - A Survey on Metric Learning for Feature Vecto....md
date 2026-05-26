---
aliases: [ML - 2013 - A Survey on Metric Learning for Feature Vecto...]
tags:
  - grupo/general
  - estado/util
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2022-08-30
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Papers/6 - Metric Learning/ML - 2013 - A Survey on Metric Learning for Feature Vectors and Structured Data.pdf"
tamanio_bytes: 816779
---

# ML - 2013 - A Survey on Metric Learning for Feature Vectors and Structured Data

Ruta interna: `GIBD/Papers/6 - Metric Learning/ML - 2013 - A Survey on Metric Learning for Feature Vectors and Structured Data.pdf`

---

arXiv:1306.6709v4  [cs.LG]  12 Feb 2014
Technical report
A Survey on Metric Learning for Feature Vectors and
Structured Data
Aur´ elien Bellet∗ bellet@usc.edu
Department of Computer Science
University of Southern California
Los Angeles, CA 90089, USA
Amaury Habrard amaury.habrard@univ-st-etienne.fr
Marc Sebban marc.sebban@univ-st-etienne.fr
Laboratoire Hubert Curien UMR 5516
Universit´ e de Saint-Etienne
18 rue Benoit Lauras, 42000 St-Etienne, France
Abstract
The need for appropriate ways to measure the distance or similarity between data is ubiq-
uitous in machine learning, pattern recognition and data mining, but h andcrafting such
good metrics for speciﬁc problems is generally diﬃcult. This has led to t he emergence of
metric learning, which aims at automatically learning a metric from data and has attracted
a lot of interest in machine learning and related ﬁelds for the past ten years. This survey
paper proposes a systematic review of the metric learning literatur e, highlighting the pros
and cons of each approach. We pay particular attention to Mahalan obis distance metric
learning, a well-studied and successful framework, but additionally present a wide range of
methods that have recently emerged as powerful alternatives, in cluding nonlinear metric
learning, similarity learning and local metric learning. Recent trends a nd extensions, such
as semi-supervised metric learning, metric learning for histogram da ta and the derivation of
generalization guarantees, are also covered. Finally, this survey a ddresses metric learning
for structured data, in particular edit distance learning, and atte mpts to give an overview
of the remaining challenges in metric learning for the years to come.
Keywords: Metric Learning, Similarity Learning, Mahalanobis Distance, Edit Dista nce
1. Introduction
The notion of pairwise metric—used throughout this survey as a generic term for distance,
similarity or dissimilarity function—between data points plays an important role in many
machine learning, pattern recognition and data mining tech niques.
1 For instance, in classi-
ﬁcation, the k-Nearest Neighbor classiﬁer ( Cover and Hart , 1967) uses a metric to identify
the nearest neighbors; many clustering algorithms, such as the prominent K-Means (Lloyd,
1982), rely on distance measurements between data points; in inf ormation retrieval, doc-
∗. Most of the work in this paper was carried out while the autho r was aﬃliated with Laboratoire Hubert
Curien UMR 5516, Universit´ e de Saint-Etienne, France.
1. Metric-based learning methods were the focus of the recen t SIMBAD European project (ICT 2008-FET
2008-2011). Website: http://simbad-fp7.eu/
c⃝ Aur´ elien Bellet, Amaury Habrard and Marc Sebban.
Bellet, Habrard and Sebban
uments are often ranked according to their relevance to a giv en query based on similarity
scores. Clearly, the performance of these methods depends o n the quality of the metric:
as in the saying “birds of a feather ﬂock together”, we hope th at it identiﬁes as similar
(resp. dissimilar) the pairs of instances that are indeed se mantically close (resp. diﬀerent).
General-purpose metrics exist (e.g., the Euclidean distan ce and the cosine similarity for
feature vectors or the Levenshtein distance for strings) bu t they often fail to capture the
idiosyncrasies of the data of interest. Improved results ar e expected when the metric is
designed speciﬁcally for the task at hand. Since manual tuni ng is diﬃcult and tedious, a lot
of eﬀort has gone into metric learning, the research topic devoted to automatically learning
metrics from data.
1.1 Metric Learning in a Nutshell
Although its origins can be traced back to some earlier work ( e.g.,
Short and Fukunaga ,
1981; Fukunaga, 1990; Friedman, 1994; Hastie and Tibshirani , 1996; Baxter and Bartlett ,
1997), metric learning really emerged in 2002 with the pioneerin g work of Xing et al. (2002)
that formulates it as a convex optimization problem. It has s ince been a hot research topic,
being the subject of tutorials at ICML 2010 2 and ECCV 2010 3 and workshops at ICCV
2011,4 NIPS 2011 5 and ICML 2013. 6
The goal of metric learning is to adapt some pairwise real-va lued metric function, say
the Mahalanobis distance dM (x, x′) =
√
(x − x′)T M (x − x′), to the problem of interest
using the information brought by training examples. Most me thods learn the metric (here,
the positive semi-deﬁnite matrix M indM ) in a weakly-supervised way from pair or triplet-
based constraints of the following form:
• Must-link / cannot-link constraints (sometimes called pos itive / negative pairs):
S = {(xi,x j) : xi and xj should be similar },
D = {(xi,x j) : xi and xj should be dissimilar }.
• Relative constraints (sometimes called training triplets ):
R = {(xi,x j,x k) : xi should be more similar to xj than to xk}.
A metric learning algorithm basically aims at ﬁnding the par ameters of the metric such
that it best agrees with these constraints (see Figure 1 for an illustration), in an eﬀort to
approximate the underlying semantic metric. This is typica lly formulated as an optimization
problem that has the following general form:
min
M
ℓ(M, S, D, R) + λR(M )
where ℓ(M, S, D, R) is a loss function that incurs a penalty when training const raints
are violated, R(M ) is some regularizer on the parameters M of the learned metric and
2. http://www.icml2010.org/tutorials.html
3. http://www.ics.forth.gr/eccv2010/tutorials.php
4. http://www.iccv2011.org/authors/workshops/
5. http://nips.cc/Conferences/2011/Program/schedule.php?Session=Workshops
6. http://icml.cc/2013/?page_id=41
2
A Survey on Metric Learning for Feature Vectors and Structur ed Data
Metric Learning
Figure 1: Illustration of metric learning applied to a face r ecognition task. For simplicity,
images are represented as points in 2 dimensions. Pairwise c onstraints, shown
in the left pane, are composed of images representing the sam e person (must-
link, shown in green) or diﬀerent persons (cannot-link, show n in red). We wish
to adapt the metric so that there are fewer constraint violat ions (right pane).
Images are taken from the Caltech Faces dataset. 8
Underlying
distribution
Metric learning
algorithm
Metric-based
algorithm
Data
sample
Learned
metric
Learned
predictor
Prediction
Figure 2: The common process in metric learning. A metric is l earned from training data
and plugged into an algorithm that outputs a predictor (e.g. , a classiﬁer, a regres-
sor, a recommender system...) which hopefully performs bet ter than a predictor
induced by a standard (non-learned) metric.
λ ≥ 0 is the regularization parameter. As we will see in this surv ey, state-of-the-art metric
learning formulations essentially diﬀer by their choice of m etric, constraints, loss function
and regularizer.
After the metric learning phase, the resulting function is u sed to improve the perfor-
mance of a metric-based algorithm, which is most often k-Nearest Neighbors ( k-NN), but
may also be a clustering algorithm such as K-Means, a ranking algorithm, etc. The common
process in metric learning is summarized in Figure 2.
1.2 Applications
Metric learning can potentially be beneﬁcial whenever the n otion of metric between in-
stances plays an important role. Recently, it has been appli ed to problems as diverse as
link prediction in networks (
Shaw et al. , 2011), state representation in reinforcement learn-
ing (Taylor et al. , 2011), music recommendation ( McFee et al. , 2012), partitioning problems
8. http://www.vision.caltech.edu/html-files/archive.html
3
Bellet, Habrard and Sebban
(Lajugie et al. , 2014), identity veriﬁcation (Ben et al. , 2012), webpage archiving (Law et al. ,
2012), cartoon synthesis ( Yu et al. , 2012) and even assessing the eﬃcacy of acupuncture
(Liang et al. , 2012), to name a few. In the following, we list three large ﬁelds of application
where metric learning has been shown to be very useful.
Computer vision There is a great need of appropriate metrics in computer visi on, not
only to compare images or videos in ad-hoc representations— such as bags-of-visual-words
(
Li and Perona , 2005)—but also in the pre-processing step consisting in buildin g this very
representation (for instance, visual words are usually obt ained by means of clustering). For
this reason, there exists a large body of metric learning lit erature dealing speciﬁcally with
computer vision problems, such as image classiﬁcation ( Mensink et al. , 2012), object recog-
nition ( Frome et al. , 2007; Verma et al. , 2012), face recognition ( Guillaumin et al. , 2009b;
Lu et al. , 2012), visual tracking ( Li et al. , 2012; Jiang et al. , 2012) or image annotation
(Guillaumin et al. , 2009a).
Information retrieval The objective of many information retrieval systems, such a s
search engines, is to provide the user with the most relevant documents according to his/her
query. This ranking is often achieved by using a metric betwe en two documents or between
a document and a query. Applications of metric learning to th ese settings include the work
of Lebanon (2006); Lee et al. (2008); McFee and Lanckriet (2010); Lim et al. (2013).
Bioinformatics Many problems in bioinformatics involve comparing sequenc es such as
DNA, protein or temporal series. These comparisons are base d on structured metrics such
as edit distance measures (or related string alignment scor es) for strings or Dynamic Time
Warping distance for temporal series. Learning these metri cs to adapt them to the task
of interest can greatly improve the results. Examples inclu de the work of Xiong and Chen
(2006); Saigo et al. (2006); Kato and Nagano (2010); Wang et al. (2012a).
1.3 Related T opics
We mention here three research topics that are related to met ric learning but outside the
scope of this survey.
Kernel learning While metric learning is parametric (one learns the paramet ers of a
given form of metric, such as a Mahalanobis distance), kerne l learning is usually nonpara-
metric: one learns the kernel matrix without any assumption on the form of the kernel
that implicitly generated it. These approaches are thus ver y powerful but limited to the
transductive setting and can hardly be applied to new data. T he interested reader may
refer to the recent survey on kernel learning by
Abbasnejad et al. (2012).
Multiple kernel learning Unlike kernel learning, Multiple Kernel Learning (MKL) is
parametric: it learns a combination of predeﬁned base kerne ls. In this regard, it can be seen
as more restrictive than metric or kernel learning, but as op posed to kernel learning, MKL
has very eﬃcient formulations and can be applied in the induc tive setting. The interested
reader may refer to the recent survey on MKL by G¨ onen and Alpaydin(2011).
Dimensionality reduction Supervised dimensionality reduction aims at ﬁnding a low-
dimensional representation that maximizes the separation of labeled data and in this respect
4
A Survey on Metric Learning for Feature Vectors and Structur ed Data
has connections with metric learning, 9 although the primary objective is quite diﬀerent.
Unsupervised dimensionality reduction, or manifold learn ing, usually assume that the (un-
labeled) data lie on an embedded low-dimensional manifold w ithin the higher-dimensional
space and aim at “unfolding” it. These methods aim at capturi ng or preserving some prop-
erties of the original data (such as the variance or local dis tance measurements) in the
low-dimensional representation.10 The interested reader may refer to the surveys by Fodor
(2002) and van der Maaten et al. (2009).
1.4 Why this Survey?
As pointed out above, metric learning has been a hot topic of r esearch in machine learning
for a few years and has now reached a considerable level of mat urity both practically and
theoretically. The early review due to
Yang and Jin (2006) is now largely outdated as it
misses out on important recent advances: more than 75% of the work referenced in the
present survey is post 2006. A more recent survey, written in dependently and in parallel to
our work, is due to Kulis (2012). Despite some overlap, it should be noted that both surveys
have their own strengths and complement each other well. Ind eed, the survey of Kulis takes
a more general approach, attempting to provide a uniﬁed view of a few core metric learning
methods. It also goes into depth about topics that are only br ieﬂy reviewed here, such
as kernelization, optimization methods and applications. On the other hand, the present
survey is a detailed and comprehensive review of the existin g literature, covering more than
50 approaches (including many recent works that are missing from Kulis’ paper) with their
relative merits and drawbacks. Furthermore, we give partic ular attention to topics that
are not covered by Kulis, such as metric learning for structu red data and the derivation of
generalization guarantees.
We think that the present survey may foster novel research in metric learning and be
useful to a variety of audiences, in particular: (i) machine learners wanting to get introduced
to or update their knowledge of metric learning will be able t o quickly grasp the pros and
cons of each method as well as the current strengths and limit ations of the research area
as a whole, and (ii) machine learning practitioners interes ted in applying metric learning to
their own problem will ﬁnd information to help them choose th e methods most appropriate
to their needs, along with links to source codes whenever ava ilable.
Note that we focus on general-purpose methods, i.e., that ar e applicable to a wide range
of application domains. The abundant literature on metric l earning designed speciﬁcally for
computer vision is not addressed because the understanding of these approaches requires a
signiﬁcant amount of background in that area. For this reaso n, we think that they deserve
a separate survey, targeted at the computer vision audience .
1.5 Prerequisites
This survey is almost self-contained and has few prerequisi tes. For metric learning from
feature vectors, we assume that the reader has some basic kno wledge of linear algebra
9. Some metric learning methods can be seen as ﬁnding a new fea ture space, and a few of them actually
have the additional goal of making this feature space low-di mensional.
10. These approaches are sometimes referred to as “unsuperv ised metric learning”, which is somewhat mis-
leading because they do not optimize a notion of metric.
5
Bellet, Habrard and Sebban
Notation Description
R Set of real numbers
Rd Set of d-dimensional real-valued vectors
Rc× d Set of c × d real-valued matrices
Sd
+ Cone of symmetric PSD d × d real-valued matrices
X Input (instance) space
Y Output (label) space
S Set of must-link constraints
D Set of cannot-link constraints
R Set of relative constraints
z = (x, y) ∈ X × Y An arbitrary labeled instance
x An arbitrary vector
M An arbitrary matrix
I Identity matrix
M ⪰ 0 PSD matrix M
∥ · ∥p p-norm
∥ · ∥F Frobenius norm
∥ · ∥∗ Nuclear norm
tr(M ) Trace of matrix M
[t]+ = max(0, 1 − t) Hinge loss function
ξ Slack variable
Σ Finite alphabet
x String of ﬁnite size
Table 1: Summary of the main notations.
and convex optimization (if needed, see Boyd and Vandenberghe , 2004, for a brush-up).
For metric learning from structured data, we assume that the reader has some familiarity
with basic probability theory, statistics and likelihood m aximization. The notations used
throughout this survey are summarized in Table 1.
1.6 Outline
The rest of this paper is organized as follows. We ﬁrst assume that data consist of vectors
lying in some feature space X ⊆ Rd. Section
2 describes key properties that we will use
to provide a taxonomy of metric learning algorithms. In Sect ion 3, we review the large
body of work dealing with supervised Mahalanobis distance l earning. Section 4 deals with
recent advances and trends in the ﬁeld, such as linear simila rity learning, nonlinear and
local methods, histogram distance learning, the derivatio n of generalization guarantees and
semi-supervised metric learning methods. We cover metric l earning for structured data
in Section 5, with a focus on edit distance learning. Lastly, we conclude this survey in
Section 6 with a discussion on the current limitations of the existing literature and promising
directions for future research.
2. Key Properties of Metric Learning Algorithms
Except for a few early methods, most metric learning algorit hms are essentially “com-
petitive” in the sense that they are able to achieve state-of -the-art performance on some
problems. However, each algorithm has its intrinsic proper ties (e.g., type of metric, ability
to leverage unsupervised data, good scalability with dimen sionality, generalization guaran-
6
A Survey on Metric Learning for Feature Vectors and Structur ed Data
Metric Learning 
Fully
supervised
Weakly
supervised
Semi
supervised
Learning 
paradigm
Form of 
metric
Linear
Nonlinear
Local
Optimality of 
the solution
Local
Global
Scalability
w.r.t. 
dimension
w.r.t. number 
of examples
Dimensionality
reduction
Yes
No
Figure 3: Five key properties of metric learning algorithms .
tees, etc) and emphasis should be placed on those when decidi ng which method to apply
to a given problem. In this section, we identify and describe ﬁve key properties of metric
learning algorithms, summarized in Figure 3. We use them to provide a taxonomy of the
existing literature: the main features of each method are gi ven in Table 2.11
Learning Paradigm We will consider three learning paradigms:
• Fully supervised: the metric learning algorithm has access to a set of labeled training
instances {zi = (xi,y i)}n
i=1, where each training example zi ∈ Z = X × Y is composed
of an instance xi ∈ X and a label (or class) yi ∈ Y . Y is a discrete and ﬁnite set of
|Y|labels (unless stated otherwise). In practice, the label in formation is often used
to generate speciﬁc sets of pair/triplet constraints S, D, R, for instance based on a
notion of neighborhood.
12
• Weakly supervised: the algorithm has no access to the labels of individual trai ning
instances: it is only provided with side information in the f orm of sets of constraints
S, D, R. This is a meaningful setting in a variety of applications wh ere labeled data is
costly to obtain while such side information is cheap: examp les include users’ implicit
feedback (e.g., clicks on search engine results), citation s among articles or links in a
network. This can be seen as having label information only at the pair/triplet level.
• Semi-supervised: besides the (full or weak) supervision, the algorithm has a ccess to
a (typically large) sample of unlabeled instances for which no side information is
available. This is useful to avoid overﬁtting when the label ed data or side information
are scarce.
F orm of Metric Clearly, the form of the learned metric is a key choice. One ma y identify
three main families of metrics:
11. Whenever possible, we use the acronyms provided by the au thors of the studied methods. When there
is no known acronym, we take the liberty of choosing one.
12. These constraints are usually derived from the labels pr ior to learning the metric and never challenged.
Note that Wang et al. (2012b) propose a more reﬁned (but costly) approach to the problem o f building
the constraints from labels. Their method alternates betwe en selecting the most relevant constraints
given the current metric and learning a new metric based on th e current constraints.
7
Bellet, Habrard and Sebban
• Linear metrics , such as the Mahalanobis distance. Their expressive power i s limited
but they are easier to optimize (they usually lead to convex f ormulations, and thus
global optimality of the solution) and less prone to overﬁtt ing.
• Nonlinear metrics, such as the χ 2 histogram distance. They often give rise to noncon-
vex formulations (subject to local optimality) and may over ﬁt, but they can capture
nonlinear variations in the data.
• Local metrics, where multiple (linear or nonlinear) local metrics are lea rned (typically
simultaneously) to better deal with complex problems, such as heterogeneous data.
They are however more prone to overﬁtting than global method s since the number of
parameters they learn can be very large.
Scalability With the amount of available data growing fast, the problem o f scalability
arises in all areas of machine learning. First, it is desirab le for a metric learning algorithm to
scale well with the number of training examples n (or constraints). As we will see, learning
the metric in an online way is one of the solutions. Second, me tric learning methods
should also scale reasonably well with the dimensionality d of the data. However, since
metric learning is often phrased as learning a d × d matrix, designing algorithms that scale
reasonably well with this quantity is a considerable challe nge.
Optimality of the Solution This property refers to the ability of the algorithm to ﬁnd
the parameters of the metric that satisfy best the criterion of interest. Ideally, the solution
is guaranteed to be the global optimum—this is essentially the case for convex formulations
of metric learning. On the contrary, for nonconvex formulat ions, the solution may only be
a local optimum.
Dimensionality Reduction As noted earlier, metric learning is sometimes formulated
as ﬁnding a projection of the data into a new feature space. An interesting byproduct in
this case is to look for a low-dimensional projected space, a llowing faster computations as
well as more compact representations. This is typically ach ieved by forcing or regularizing
the learned metric matrix to be low-rank.
3. Supervised Mahalanobis Distance Learning
This section deals with (fully or weakly) supervised Malaha nobis distance learning (some-
times simply referred to as distance metric learning), whic h has attracted a lot of interest
due to its simplicity and nice interpretation in terms of a li near projection. We start by
presenting the Mahalanobis distance and two important chal lenges associated with learning
this form of metric.
The Mahalanobis distance This term comes from
Mahalanobis (1936) and originally
refers to a distance measure that incorporates the correlat ion between features:
dmaha(x, x′) =
√
(x − x′)T Ω− 1(x − x′),
where x and x′ are random vectors from the same distribution with covarian ce matrix Ω.
By an abuse of terminology common in the metric learning lite rature, we will in fact use
8
A Survey on Metric Learning for Feature Vectors and Structur ed Data
Page Name Y ear Source Supervision F orm of Scalability Optimum Dimension Regularizer Additional
Code Metric w.r.t. n w.r.t. d Reduction Information
11 MMC 2002 Yes W eak Linear ★✩✩ ✩✩✩ Global No None —
11 S&J 2003 No W eak Linear ★★✩ ★★★ Global No Frobenius norm —
12 NCA 2004 Yes Full Linear ★✩✩ ★★✩ Local Yes None For k-NN
12 MCML 2005 Yes Full Linear ★✩✩ ✩✩✩ Global No None For k-NN
13 LMNN 2005 Yes Full Linear ★★✩ ★✩✩ Global No None For k-NN
13 RCA 2003 Yes W eak Linear ★★✩ ★★✩ Global No None —
14 ITML 2007 Yes W eak Linear ★✩✩ ★★✩ Global No LogDet Online version
15 SDML 2009 No W eak Linear ★✩✩ ★★✩ Global No LogDet+L 1 n ≪ d
15 POLA 2004 No W eak Linear ★★★ ★✩✩ Global No None Online
15 LEGO 2008 No W eak Linear ★★★ ★★✩ Global No LogDet Online
16 RDML 2009 No W eak Linear ★★★ ★★✩ Global No Frobenius norm Online
16 MDML 2012 No W eak Linear ★★★ ★✩✩ Global Yes Nuclear norm Online
16 mt-LMNN 2010 Yes Full Linear ★★✩ ✩✩✩ Global No Frobenius norm Multi-task
17 MLCS 2011 No W eak Linear ★✩✩ ★★✩ Local Yes N/A Multi-task
17 GPML 2012 No W eak Linear ★✩✩ ★★✩ Global Yes von Neumann Multi-task
18 TML 2010 Yes W eak Linear ★★✩ ★★✩ Global No Frobenius norm Transfer learning
19 LPML 2006 No W eak Linear ★★✩ ★★✩ Global Yes L1 norm —
19 SML 2009 No W eak Linear ★✩✩ ✩✩✩ Global Yes L2,1 norm —
19 BoostMetric 2009 Yes W eak Linear ★✩✩ ★★✩ Global Yes None —
20 DML-p 2012 No W eak Linear ★✩✩ ★✩✩ Global No None —
20 RML 2010 No W eak Linear ★★✩ ✩✩✩ Global No Frobenius norm Noisy constraints
21 MLR 2010 Yes Full Linear ★★✩ ✩✩✩ Global Yes Nuclear norm For ranking
22 SiLA 2008 No Full Linear ★★✩ ★★✩ N/A No None Online
22 gCosLA 2009 No W eak Linear ★★★ ✩✩✩ Global No None Online
23 OASIS 2009 Yes W eak Linear ★★★ ★★✩ Global No Frobenius norm Online
23 SLLC 2012 No Full Linear ★★✩ ★★✩ Global No Frobenius norm For linear classif.
24 RSL 2013 No Full Linear ★✩✩ ★★✩ Local No Frobenius norm Rectangular matrix
25 LSMD 2005 No W eak Nonlinear ★✩✩ ★★✩ Local Yes None —
25 NNCA 2007 No Full Nonlinear ★✩✩ ★★✩ Local Yes Recons. error —
26 SVML 2012 No Full Nonlinear ★✩✩ ★★✩ Local Yes Frobenius norm For SVM
26 GB-LMNN 2012 No Full Nonlinear ★★✩ ★★✩ Local Yes None —
26 HDML 2012 Yes W eak Nonlinear ★★✩ ★★✩ Local Yes L2 norm Hamming distance
27 M2-LMNN 2008 Yes Full Local ★★✩ ★✩✩ Global No None —
28 GLML 2010 No Full Local ★★★ ★★✩ Global No Diagonal Generative
28 Bk-means 2009 No W eak Local ★✩✩ ★★★ Global No RKHS norm Bregman dist.
29 PLML 2012 Yes W eak Local ★★✩ ✩✩✩ Global No Manifold+Frob —
29 RFD 2012 Yes W eak Local ★★✩ ★★★ N/A No None Random forests
30 χ2-LMNN 2012 No Full Nonlinear ★★✩ ★★✩ Local Yes None Histogram data
31 GML 2011 No W eak Linear ★✩✩ ★★✩ Local No None Histogram data
31 EMDL 2012 No W eak Linear ★✩✩ ★★✩ Local No Frobenius norm Histogram data
34 LRML 2008 Yes Semi Linear ★✩✩ ✩✩✩ Global No Laplacian —
35 M-DML 2009 No Semi Linear ★✩✩ ✩✩✩ Local No Laplacian Auxiliary metrics
35 SERAPH 2012 Yes Semi Linear ★✩✩ ✩✩✩ Local Yes Trace+entropy Probabilistic
36 CDML 2011 No Semi N/A N/A N/A N/A N/A N/A Domain adaptation
36 DAML 2011 No Semi Nonlinear ★✩✩ ✩✩✩ Global No MMD Domain adaptation
Table 2: Main features of metric learning methods for featur e vectors. Scalability levels are relative and given as a rou gh guide.
9
Bellet, Habrard and Sebban
the term Mahalanobis distance to refer to generalized quadr atic distances, deﬁned as
dM (x, x′) =
√
(x − x′)T M (x − x′)
and parameterized by M ∈ Sd
+, where Sd
+ is the cone of symmetric positive semi-deﬁnite
(PSD) d × d real-valued matrices (see Figure
4).13 M ∈ Sd
+ ensures that dM satisﬁes the
properties of a pseudo-distance: ∀x, x′, x′′ ∈ X ,
1. dM (x, x′) ≥ 0 (nonnegativity),
2. dM (x, x) = 0 (identity),
3. dM (x, x′) = d(x′, x) (symmetry),
4. dM (x, x′′) ≤ d(x, x′) +d(x′, x′′) (triangle inequality).
Interpretation Note that when M is the identity matrix, we recover the Euclidean
distance. Otherwise, one can express M as LT L, where L ∈ Rk× d where k is the rank of
M . We can then rewrite dM (x, x′) as follows:
dM (x, x′) =
√
(x − x′)T M (x − x′)
=
√
(x − x′)T LT L(x − x′)
=
√
(Lx − Lx′)T (Lx − Lx′).
Thus, a Mahalanobis distance implicitly corresponds to com puting the Euclidean distance
after the linear projection of the data deﬁned by the transfo rmation matrix L. Note that
if M is low-rank, i.e., rank( M ) = r < d, then it induces a linear projection of the data
into a space of lower dimension r. It thus allows a more compact representation of the
data and cheaper distance computations, especially when th e original feature space is high-
dimensional. These nice properties explain why learning Ma halanobis distance has attracted
a lot of interest and is a major component of metric learning.
Challenges This leads us to two important challenges associated with le arning Maha-
lanobis distances. The ﬁrst one is to maintain M ∈ Sd
+ in an eﬃcient way during the
optimization process. A simple way to do this is to use the pro jected gradient method
which consists in alternating between a gradient step and a p rojection step onto the PSD
cone by setting the negative eigenvalues to zero.
14 However this is expensive for high-
dimensional problems as eigenvalue decomposition scales i n O(d3). The second challenge
is to learn a low-rank matrix (which implies a low-dimension al projection space, as noted
earlier) instead of a full-rank one. Unfortunately, optimi zing M subject to a rank constraint
or regularization is NP-hard and thus cannot be carried out e ﬃciently.
13. Note that in practice, to get rid of the square root, the Ma halanobis distance is learned in its more
convenient squared form d2
M (x, x′) = ( x − x′)T M (x − x′).
14. Note that
Qian et al. (2013) have proposed some heuristics to avoid doing this projecti on at each itera-
tion.
10
A Survey on Metric Learning for Feature Vectors and Structur ed Data
0
0.2
0.4
0.6
0.8
1
−1
−0.5
0
0.5
1
0
0.2
0.4
0.6
0.8
1
αβ
γ
Figure 4: The cone S2
+ of positive semi-deﬁnite 2x2 matrices of the form
[
α β
β γ
]
.
The rest of this section is a comprehensive review of the supe rvised Mahalanobis distance
learning methods of the literature. We ﬁrst present two earl y approaches (Section
3.1). We
then discuss methods that are speciﬁc to k-nearest neighbors (Section 3.2), inspired from in-
formation theory (Section 3.3), online learning approaches (Section 3.4), multi-task learning
(Section 3.5) and a few more that do not ﬁt any of the previous categories (S ection 3.6).
3.1 Early Approaches
The approaches in this section deal with the PSD constraint i n a rudimentary way.
MMC (Xing et al.) The seminal work of
Xing et al. (2002) is the ﬁrst Mahalanobis
distance learning method. 15 It relies on a convex formulation with no regularization, wh ich
aims at maximizing the sum of distances between dissimilar p oints while keeping the sum
of distances between similar examples small:
max
M ∈ Sd
+
∑
(xi,xj)∈D
dM (xi, xj)
s.t.
∑
(xi,xj)∈S
d2
M (xi, xj) ≤ 1.
(1)
The algorithm used to solve (
1) is a simple projected gradient approach requiring the full
eigenvalue decomposition of M at each iteration. This is typically intractable for medium
and high-dimensional problems.
S&J (Schultz & Joachims) The method proposed by
Schultz and Joachims (2003) re-
lies on the parameterization M = AT W A, where A is ﬁxed and known and W diagonal.
We get:
d2
M (xi, xj) = ( Axi − Axj)T W (Axi − Axj).
15. Source code available at: http://www.cs.cmu.edu/~epxing/papers/
11
Bellet, Habrard and Sebban
By deﬁnition, M is PSD and thus one can optimize over the diagonal matrix W and avoid
the need for costly projections on the PSD cone. They propose a formulation based on
triplet constraints:
min
W
∥M ∥2
F + C
∑
i,j,k
ξijk
s.t. d2
M (xi, xk) − d2
M (xi, xj) ≥ 1 − ξijk ∀(xi, xj, xk) ∈ R,
(2)
where ∥M ∥2
F = ∑
i,jM 2
ij is the squared Frobenius norm of M , the ξijk ’s are “slack” vari-
ables to allow soft constraints
16 andC ≥ 0 is the trade-oﬀ parameter between regularization
and constraint satisfaction. Problem ( 2) is convex and can be solved eﬃciently. The main
drawback of this approach is that it is less general than full Mahalanobis distance learning:
one only learns a weighting W of the features. Furthermore, A must be chosen manually.
3.2 Approaches Driven by Nearest Neighbors
The objective functions of the methods presented in this sec tion are related to a nearest
neighbor prediction rule.
NCA (Goldberger et al.) The idea of Neighbourhood Component Analysis
17 (NCA),
introduced by Goldberger et al. (2004), is to optimize the expected leave-one-out error of a
stochastic nearest neighbor classiﬁer in the projection sp ace induced by dM . They use the
decomposition M = LT L and they deﬁne the probability that xi is the neighbor of xj by
pij = exp(−∥ Lxi − Lxj∥2
2)
∑
l̸=i exp(−∥ Lxi − Lxl∥2
2), p ii = 0.
Then, the probability that xi is correctly classiﬁed is:
pi =
∑
j:yj=yi
pij.
They learn the distance by solving:
max
L
∑
i
pi. (3)
Note that the matrix L can be chosen to be rectangular, inducing a low-rank M . The main
limitation of (
3) is that it is nonconvex and thus subject to local maxima. Hong et al. (2011)
later proposed to learn a mixture of NCA metrics, while Tarlow et al. (2013) generalize NCA
to k-NN with k >1.
MCML (Globerson & Roweis) Shortly after Goldberger et al., Globerson and Roweis
(2005) proposed MCML (Maximally Collapsing Metric Learning), an alternative convex
formulation based on minimizing a KL divergence between pij and an ideal distribution,
16. This is a classic trick used for instance in soft-margin S VM (Cortes and Vapnik , 1995). Throughout this
survey, we will consistently use the symbol ξ to denote slack variables.
17. Source code available at: http://www.ics.uci.edu/~fowlkes/software/nca/
12
A Survey on Metric Learning for Feature Vectors and Structur ed Data
which can be seen as attempting to collapse each class to a sin gle point. 18 Unlike NCA,
the optimization is done with respect to the matrix M and the problem is thus convex.
However, like MMC, MCML requires costly projections onto th e PSD cone.
LMNN (W einberger et al.) Large Margin Nearest Neighbors 19 (LMNN), introduced by
Weinberger et al. ( 2005; 2008; 2009), is one of the most widely-used Mahalanobis distance
learning methods and has been the subject of many extensions (described in later sections).
One of the reasons for its popularity is that the constraints are deﬁned in a local way: the
k nearest neighbors (the “target neighbors”) of any training instance should belong to the
correct class while keeping away instances of other classes (the “impostors”). The Euclidean
distance is used to determine the target neighbors. Formall y, the constraints are deﬁned in
the following way:
S = {(xi, xj) : yi =yj and xj belongs to the k-neighborhood of xi},
R = {(xi, xj, xk) : ( xi, xj) ∈ S,y i ̸=yk}.
The distance is learned using the following convex program:
min
M ∈ Sd
+
(1 − µ )
∑
(xi,xj )∈S
d2
M (xi, xj) + µ
∑
i,j,k
ξijk
s.t. d2
M (xi, xk) − d2
M (xi, xj) ≥ 1 − ξijk ∀(xi, xj, xk) ∈ R,
(4)
where µ ∈ [0, 1] controls the “pull/push” trade-oﬀ. The authors developed a special-
purpose solver—based on subgradient descent and careful bo ok-keeping—that is able to
deal with billions of constraints. Alternative ways of solv ing the problem have been pro-
posed (
Torresani and Lee , 2006; Nguyen and Guo , 2008; Park et al. , 2011; Der and Saul ,
2012). LMNN generally performs very well in practice, although i t is sometimes prone to
overﬁtting due to the absence of regularization, especiall y in high dimension. It is also very
sensitive to the ability of the Euclidean distance to select relevant target neighbors. Note
that Do et al. (2012) highlighted a relation between LMNN and Support Vector Mac hines.
3.3 Information-Theoretic Approaches
The methods presented in this section frame metric learning as an optimization problem
involving an information measure.
RCA (Bar-Hillel et al.) Relevant Component Analysis
20 (Shental et al. , 2002; Bar-Hillel et al. ,
2003, 2005) makes use of positive pairs only and is based on subsets of th e training exam-
ples called “chunklets”. These are obtained from the set of p ositive pairs by applying a
transitive closure: for instance, if ( x1, x2) ∈ S and ( x2, x3) ∈ S , then x1, x2 and x3 belong
to the same chunklet. Points in a chunklet are believed to sha re the same label. Assuming
a total of n points in k chunklets, the algorithm is very eﬃcient since it simply amo unts to
18. An implementation is available within the Matlab Toolbo x for Dimensionality Reduction:
http://homepage.tudelft.nl/19j49/Matlab_Toolbox_for_Dimensionality_Reduction.html
19. Source code available at: http://www.cse.wustl.edu/~kilian/code/code.html
20. Source code available at: http://www.scharp.org/thertz/code.html
13
Bellet, Habrard and Sebban
computing the following matrix:
ˆC = 1
n
k∑
j=1
nj∑
i=1
(xji − ˆmj)(xji − ˆmj)T,
where chunkletj consists of {xji}nj
i=1 and ˆmj is its mean. Thus, RCA essentially reduces the
within-chunklet variability in an eﬀort to identify feature s that are irrelevant to the task.
The inverse of ˆC is used in a Mahalanobis distance. The authors have shown tha t (i) it is
the optimal solution to an information-theoretic criterio n involving a mutual information
measure, and (ii) it is also the optimal solution to the optim ization problem consisting in
minimizing the within-class distances. An obvious limitat ion of RCA is that it cannot make
use of the discriminative information brought by negative p airs, which explains why it is
not very competitive in practice. RCA was later extended to h andle negative pairs, at the
cost of a more expensive algorithm ( Hoi et al. , 2006; Yeung and Chang , 2006).
ITML (Davis et al.) Information-Theoretic Metric Learning 21 (ITML), proposed by
Davis et al. (2007), is an important work because it introduces LogDet diverge nce regular-
ization that will later be used in several other Mahalanobis distance learning methods (e.g.,
Jain et al. , 2008; Qi et al. , 2009). This Bregman divergence on positive deﬁnite matrices is
deﬁned as:
Dld(M, M 0) = tr( M M− 1
0 ) − log det(M M− 1
0 ) − d,
where d is the dimension of the input space and M 0 is some positive deﬁnite matrix we
want to remain close to. In practice, M 0 is often set to I (the identity matrix) and thus
the regularization aims at keeping the learned distance clo se to the Euclidean distance. The
key feature of the LogDet divergence is that it is ﬁnite if and only if M is positive deﬁnite.
Therefore, minimizing Dld(M, M 0) provides an automatic and cheap way of preserving the
positive semi-deﬁniteness of M . ITML is formulated as follows:
min
M ∈ Sd
+
Dld(M, M 0) + γ
∑
i,j
ξij
s.t. d2
M (xi, xj) ≤ u +ξij ∀(xi, xj) ∈ S
d2
M (xi, xj) ≥ v − ξij ∀(xi, xj) ∈ D,
(5)
where u,v ∈ R are threshold parameters and γ ≥ 0 the trade-oﬀ parameter. ITML thus
aims at satisfying the similarity and dissimilarity constr aints while staying as close as pos-
sible to the Euclidean distance (if M 0 = I). More precisely, the information-theoretic
interpretation behind minimizing Dld(M, M 0) is that it is equivalent to minimizing the
KL divergence between two multivariate Gaussian distribut ions parameterized by M and
M 0. The algorithm proposed to solve (
5) is eﬃcient, converges to the global minimum and
the resulting distance performs well in practice. A limitat ion of ITML is that M 0, that
must be picked by hand, can have an important inﬂuence on the q uality of the learned
distance. Note that Kulis et al. (2009) have shown how hashing can be used together with
ITML to achieve fast similarity search.
21. Source code available at: http://www.cs.utexas.edu/~pjain/itml/
14
A Survey on Metric Learning for Feature Vectors and Structur ed Data
SDML (Qi et al.) With Sparse Distance Metric Learning (SDML), Qi et al. (2009)
speciﬁcally deal with the case of high-dimensional data tog ether with few training samples,
i.e., n ≪ d. To avoid overﬁtting, they use a double regularization: the LogDet divergence
(using M 0 = I or M 0 = Ω− 1 where Ω is the covariance matrix) and L1-regularization
on the oﬀ-diagonal elements of M . The justiﬁcation for using this L1-regularization is
two-fold: (i) a practical one is that in high-dimensional sp aces, the oﬀ-diagonal elements of
Ω− 1 are often very small, and (ii) a theoretical one suggested by a consistency result from
a previous work in covariance matrix estimation ( Ravikumar et al. , 2011) that applies to
SDML. They use a fast algorithm based on block-coordinate de scent (the optimization is
done over each row of M − 1) and obtain very good performance for the speciﬁc case n ≪ d.
3.4 Online Approaches
In online learning (
Littlestone, 1988), the algorithm receives training instances one at a
time and updates at each step the current hypothesis. Althou gh the performance of online
algorithms is typically inferior to batch algorithms, they are very useful to tackle large-scale
problems that batch methods fail to address due to time and sp ace complexity issues. Online
learning methods often come with regret bounds, stating tha t the accumulated loss suﬀered
along the way is not much worse than that of the best hypothesi s chosen in hindsight. 22
POLA (Shalev-Shwartz et al.) POLA (Shalev-Shwartz et al. , 2004), for Pseudo-metric
Online Learning Algorithm, is the ﬁrst online Mahalanobis d istance learning approach and
learns the matrix M as well as a threshold b ≥ 1. At each step t, POLA receives a pair
(xi, xj,y ij), where yij = 1 if ( xi, xj) ∈ S and yij = − 1 if ( xi, xj) ∈ D , and performs two
successive orthogonal projections:
1. Projection of the current solution ( M t− 1,b t− 1) onto the set C1 = {(M,b ) ∈ Rd2+1 :
[yij(d2
M (xi, xj) − b) + 1]+ = 0}, which is done eﬃciently (closed-form solution). The
constraint basically requires that the distance between tw o instances of same (resp.
diﬀerent) labels be below (resp. above) the threshold b with a margin 1. We get an
intermediate solution ( M t− 1
2,b t− 1
2 ) that satisﬁes this constraint while staying as close
as possible to the previous solution.
2. Projection of ( M t− 1
2,b t− 1
2 ) onto the set C2 = {(M,b ) ∈ Rd2+1 : M ∈ Sd
+,b ≥ 1},
which is done rather eﬃciently (in the worst case, one only ne eds to compute the
minimal eigenvalue of M t− 1
2 ). This projects the matrix back onto the PSD cone. We
thus get a new solution ( M t,b t) that yields a valid Mahalanobis distance.
A regret bound for the algorithm is provided.
LEGO (Jain et al.) LEGO (Logdet Exact Gradient Online), developed by
Jain et al.
(2008), is an improved version of POLA based on LogDet divergence r egularization. It
features tighter regret bounds, more eﬃcient updates and be tter practical performance.
22. A regret bound has the following general form: ∑T
t=1ℓ(ht,z t) − ∑T
t=1ℓ(h∗,z t) ≤ O(T ), where T is the
number of steps, ht is the hypothesis at time t and h∗ is the best batch hypothesis.
15
Bellet, Habrard and Sebban
RDML (Jin et al.) RDML ( Jin et al. , 2009) is similar to POLA in spirit but is more
ﬂexible. At each step t, instead of forcing the margin constraint to be satisﬁed, it performs
a gradient descent step of the following form (assuming Frob enius regularization):
M t =πSd
+
(
M t− 1 − λy ij(xi − xj)(xi − xj)T )
,
where πSd
+
(·) is the projection to the PSD cone. The parameter λ implements a trade-oﬀ
between satisfying the pairwise constraint and staying clo se to the previous matrix M t− 1.
Using some linear algebra, the authors show that this update can be performed by solving
a convex quadratic program instead of resorting to eigenval ue computation like POLA.
RDML is evaluated on several benchmark datasets and is shown to perform comparably to
LMNN and ITML.
MDML (Kunapuli & Shavlik) MDML (
Kunapuli and Shavlik , 2012), for Mirror De-
scent Metric Learning, is an attempt of proposing a general f ramework for online Maha-
lanobis distance learning. It is based on composite mirror d escent (Duchi et al. , 2010), which
allows online optimization of many regularized problems. I t can accommodate a large class
of loss functions and regularizers for which eﬃcient update s are derived, and the algorithm
comes with a regret bound. Their study focuses on regulariza tion with the nuclear norm
(also called trace norm) introduced by Fazel et al. (2001) and deﬁned as ∥M ∥∗ = ∑
iσi,
where the σi’s are the singular values of M .23 It is known to be the best convex relaxation
of the rank of the matrix and thus nuclear norm regularizatio n tends to induce low-rank
matrices. In practice, MDML has performance comparable to L MNN and ITML, is fast and
sometimes induces low-rank solutions, but surprisingly th e algorithm was not evaluated on
large-scale datasets.
3.5 Multi-T ask Metric Learning
This section covers Mahalanobis distance learning for the m ulti-task setting (
Caruana,
1997), where given a set of related tasks, one learns a metric for e ach in a coupled fashion
in order to improve the performance on all tasks.
mt-LMNN (Parameswaran & W einberger) Multi-Task LMNN
24 (Parameswaran and Weinberger ,
2010) is a straightforward adaptation of the ideas of Multi-Task SVM (Evgeniou and Pontil ,
2004) to metric learning. Given T related tasks, they model the problem as learning a shared
Mahalanobis metric dM 0 as well as task-speciﬁc metrics dM 1,...,d M t and deﬁne the metric
for task t as
dt(x, x′) = ( x − x′)T (M 0 + M t)(x − x′).
Note that M 0 + M t ⪰ 0, hence dt is a valid pseudo-metric. The LMNN formulation is
easily generalized to this multi-task setting so as to learn the metrics jointly, with a speciﬁc
regularization term deﬁned as follows:
γ0∥M 0 − I∥2
F +
T∑
t=1
γt∥M t∥2
F,
23. Note that when M ∈ Sd
+, ∥M ∥∗ = tr(M ) = ∑d
i=1Mii, which is much cheaper to compute.
24. Source code available at:
http://www.cse.wustl.edu/~kilian/code/code.html
16
A Survey on Metric Learning for Feature Vectors and Structur ed Data
whereγt controls the regularization of M t. When γ0 → ∞ , the shared metric dM 0 is simply
the Euclidean distance, and the formulation reduces to T independent LMNN formulations.
On the other hand, when γt>0 → ∞ , the task-speciﬁc matrices are simply zero matrices
and the formulation reduces to LMNN on the union of all data. I n-between these extreme
cases, these parameters can be used to adjust the relative im portance of each metric: γ0
to set the overall level of shared information, and γt to set the importance of M t with
respect to the shared metric. The formulation remains conve x and can be solved using the
same eﬃcient solver as LMNN. In the multi-task setting, mt-L MNN clearly outperforms
single-task metric learning methods and other multi-task c lassiﬁcation techniques such as
mt-SVM.
MLCS (Y ang et al.) MLCS (
Yang et al. , 2011) is a diﬀerent approach to the problem
of multi-task metric learning. For each task t ∈ { 1,...,T }, the authors consider learning a
Mahalanobis metric
d2
LT
t Lt
(x, x′) = ( x − x′)T LT
t Lt(x − x′) = ( Ltx − Ltx′)T (Ltx − Ltx′)
parameterized by the transformation matrix Lt ∈ Rr× d. They show that Lt can be decom-
posed into a “subspace” part Lt
0 ∈ Rr× d and a “low-dimensional metric” part Rt ∈ Rr× r
such that Lt = RtLt
0. The main assumption of MLCS is that all tasks share a common
subspace, i.e., ∀t, Lt
0 = L0. This parameterization can be used to extend most of metric
learning methods to the multi-task setting, although it bre aks the convexity of the formu-
lation and is thus subject to local optima. However, as oppos ed to mt-LMNN, it can be
made low-rank by setting r<d and thus has many less parameters to learn. In their work,
MLCS is applied to the version of LMNN solved with respect to t he transformation matrix
(
Torresani and Lee , 2006). The resulting method is evaluated on problems with very sc arce
training data and study the performance for diﬀerent values o fr. It is shown to outperform
mt-LMNN, but the setup is a bit unfair to mt-LMNN since it is fo rced to be low-rank by
eigenvalue thresholding.
GPML (Y ang et al.) The work of
Yang et al. (2012) identiﬁes two drawbacks of pre-
vious multi-task metric learning approaches: (i) MLCS’s as sumption of common subspace
is sometimes too strict and leads to a nonconvex formulation , and (ii) the Frobenius reg-
ularization of mt-LMNN does not preserve geometry. This pro perty is deﬁned as being
the ability to propagate side-information: the task-speci ﬁc metrics should be regularized so
as to preserve the relative distance between training pairs . They introduce the following
formulation, which extends any metric learning algorithm t o the multi-task setting:
min
M 0,...,M t∈ Sd
+
t∑
i=1
(ℓ(M t, St, Dt, Rt) +γdϕ(M t, M 0)) + γ0dϕ(A0, M 0), (6)
whereℓ(M t, St, Dt, Rt) is the loss function for the task t based on the training pairs/triplets
(depending on the chosen algorithm), dϕ(A, B) = ϕ (A) − ϕ (B) − tr
(
(∇ ϕ B)T (A − B)
)
is a Bregman matrix divergence (
Dhillon and Tropp , 2007) and A0 is a predeﬁned metric
(e.g., the identity matrix I). mt-LMNN can essentially be recovered from ( 6) by setting
ϕ (A) = ∥A∥2
F and additional constraints M t ⪰ M 0. The authors focus on the von
17
Bellet, Habrard and Sebban
Neumann divergence:
dV N (A, B) = tr( A log A − A log B − A + B),
where log A is the matrix logarithm of A. Like the LogDet divergence mentioned earlier in
this survey (Section 3.3), the von Neumann divergence is known to be rank-preserving and to
provide automatic enforcement of positive-semideﬁnitene ss. The authors further show that
minimizing this divergence encourages geometry preservat ion between the learned metrics.
Problem ( 6) remains convex as long as the original algorithm used for so lving each task is
convex, and can be solved eﬃciently using gradient descent m ethods. In the experiments,
the method is adapted to LMNN and outperforms single-task LM NN as well as mt-LMNN,
especially when training data is very scarce.
TML (Zhang & Y eung)
Zhang and Yeung (2010) propose a transfer metric learning
(TML) approach.25 They assume that we are given S independent source tasks with enough
labeled data and that a Mahalanobis distance M s has been learned for each task s. The goal
is to leverage the information of the source metrics to learn a distance M t for a target task,
for which we only have a scarce amount nt of labeled data. No assumption is made about
the relation between the source tasks and the target task: th ey may be positively/negatively
correlated or uncorrelated. The problem is formulated as fo llows:
min
M t∈ Sd
+,Ω⪰0
2
n2
t
∑
i<j
ℓ
(
yiyj
[
1 − d2
M t(xi, xj)
])
+ λ 1
2 ∥M t∥2
F + λ 2
2 tr( ˜M Ω− 1 ˜M
T
)
s.t. tr( Ω) = 1,
(7)
where ℓ(t) = max(0, 1 − t) is the hinge loss, ˜M = (vec(M 1),..., vec(M s), vec(M t)). The
ﬁrst two terms are classic (loss on all possible pairs and Fro benius regularization) while the
third one models the relation between tasks based on a positi ve deﬁnite covariance matrix
Ω. Assuming that the source tasks are independent and of equal importance, Ω can be
expressed as
Ω =
(
α I (m− 1)× (m− 1) ωm
ωm ω
)
,
where ωm denotes the task covariances between the target task and the source tasks, and
ω denotes the variance of the target task. Problem ( 7) is convex and is solved using an
alternating procedure that is guaranteed to converge to the global optimum: (i) ﬁxing Ω
and solving for M t, which is done online with an algorithm similar to RDML, and ( ii) ﬁxing
M t and solving for Ω, leading to a second-order cone program whose number of vari ables
and constraints is linear in the number of tasks. In practice , TML consistently outperforms
metric learning methods without transfer when training dat a is scarce.
3.6 Other Approaches
In this section, we describe a few approaches that are outsid e the scope of the previous
categories. The ﬁrst two (LPML and SML) fall into the categor y of sparse metric learning
25. Source code available at: http://www.cse.ust.hk/~dyyeung/
18
A Survey on Metric Learning for Feature Vectors and Structur ed Data
methods. BoostMetric is inspired from the theory of boostin g. DML-p revisits the original
metric learning formulation of Xing et al. RML deals with the presence of noisy constraints.
Finally, MLR learns a metric for solving a ranking task.
LPML (Rosales & F ung) The method of
Rosales and Fung (2006) aims at learning
matrices with entire columns/rows set to zero, thus making M low-rank. For this purpose,
they use L1 norm regularization and, restricting their framework to di agonal dominant
matrices, they are able to formulate the problem as a linear p rogram that can be solved
eﬃciently. However, L1 norm regularization favors sparsity at the entry level only , not
speciﬁcally at the row/column level, even though in practic e the learned matrix is sometimes
low-rank. Furthermore, the approach is less general than Ma halanobis distances due to the
restriction to diagonal dominant matrices.
SML (Ying et al.) SML
26 (Ying et al. , 2009), for Sparse Metric Learning, is a distance
learning approach that regularizes M with the mixed L2,1 norm deﬁned as
∥M ∥2,1 =
d∑
i=1
∥M i∥2,
which tends to zero out entire rows of M (as opposed to the L1 norm used in LPML), and
therefore performs feature selection. More precisely, the y set M = U T W U, where U ∈ Od
(the set of d × d orthonormal matrices) and W ∈ Sd
+, and solve the following problem:
min
U ∈ Od,W ∈ Sd
+
∥W ∥2,1 + γ
∑
i,j,k
ξijk
s.t. d2
M (xi, xk) − d2
M (xi, xj) ≥ 1 − ξijk ∀(xi, xj, xk) ∈ R,
(8)
where γ ≥ 0 is the trade-oﬀ parameter. Unfortunately, L2,1 regularized problems are
typically diﬃcult to optimize. Problem (
8) is reformulated as a min-max problem and solved
using smooth optimization ( Nesterov, 2005). Overall, the algorithm has a fast convergence
rate but each iteration has an O(d3) complexity. The method performs well in practice while
achieving better dimensionality reduction than full-rank methods such as Rosales and Fung
(2006). However, it cannot be applied to high-dimensional proble ms due to the complexity
of the algorithm. Note that the same authors proposed a uniﬁe d framework for sparse
metric learning ( Huang et al. , 2009, 2011).
BoostMetric (Shen et al.) BoostMetric27 (Shen et al. , 2009, 2012) adapts to Maha-
lanobis distance learning the ideas of boosting, where a goo d hypothesis is obtained through
a weighted combination of so-called “weak learners” (see th e recent book on this matter
by Schapire and Freund , 2012). The method is based on the property that any PSD ma-
trix can be decomposed into a positive linear combination of trace-one rank-one matrices.
This kind of matrices is thus used as weak learner and the auth ors adapt the popular
boosting algorithm Adaboost ( Freund and Schapire , 1995) to this setting. The resulting al-
gorithm is quite eﬃcient since it does not require full eigen value decomposition but only the
26. Source code is not available but is indicated as “coming s oon” by the authors. Check:
http://www.enm.bris.ac.uk/staff/xyy/software.html
27. Source code available at: http://code.google.com/p/boosting/
19
Bellet, Habrard and Sebban
computation of the largest eigenvalue. In practice, BoostM etric achieves competitive perfor-
mance but typically requires a very large number of iteratio ns for high-dimensional datasets.
Bi et al. (2011) further improve the scalability of the approach, while Liu and Vemuri (2012)
introduce regularization on the weights as well as a term to r educe redundancy among the
weak learners.
DML-p (Ying et al., Cao et al.) The work of
Ying and Li (2012); Cao et al. (2012b)
revisit MMC, the original approach of Xing et al. (2002), by investigating the following
formulation, called DML-p:
max
M ∈ Sd
+

 1
|D|
∑
(xi,xj )∈D
[dM (xi, xj)]2p


1/p
s.t.
∑
(xi,xj )∈S
d2
M (xi, xj) ≤ 1.
(9)
Note that by setting p = 0. 5 we recover MMC. The authors show that (
9) is convex for
p ∈ (−∞ , 1) and can be cast as a well-known eigenvalue optimization pr oblem called “min-
imizing the maximal eigenvalue of a symmetric matrix”. They further show that it can
be solved eﬃciently using a ﬁrst-order algorithm that only r equires the computation of the
largest eigenvalue at each iteration (instead of the costly full eigen-decomposition used by
Xing et al.). Experiments show competitive results and low c omputational complexity. A
general drawback of DML- p is that it is not clear how to accommodate a regularizer (e.g. ,
sparse or low-rank).
RML (Huang et al.) Robust Metric Learning (
Huang et al. , 2010) is a method that
can successfully deal with the presence of noisy/incorrect training constraints, a situation
that can arise when they are not derived from class labels but from side information such
as users’ implicit feedback. The approach is based on robust optimization ( Ben-Tal et al. ,
2009): assuming that a proportion 1 − η of the m training constraints (say triplets) are
incorrect, it minimizes some loss function ℓ for any η fraction of the triplets:
min
M ∈ Sd
+,t
t + λ
2 ∥M ∥F
s.t. t ≥
m∑
i=1
qiℓ
(
d2
M (xi, x′′
i ) − d2
M (xi, x′
i)
)
, ∀q ∈ Q (η),
(10)
where ℓ is taken to be the hinge loss and Q(η) is deﬁned as
Q(η) =
{
q ∈ { 0, 1}m :
m∑
i=1
qi ≤ ηm
}
.
In other words, Problem (
10) minimizes the worst-case violation over all possible sets of
correct constraints. Q(η) can be replaced by its convex hull, leading to a semi-deﬁnit e
program with an inﬁnite number of constraints. This can be fu rther simpliﬁed into a
convex minimization problem that can be solved either using subgradient descent or smooth
20
A Survey on Metric Learning for Feature Vectors and Structur ed Data
optimization ( Nesterov, 2005). However, both of these require a projection onto the PSD
cone. Experiments on standard datasets show good robustnes s for up to 30% of incorrect
triplets, while the performance of other methods such as LMN N is greatly damaged.
MLR (McF ee & Lankriet) The idea of MLR ( McFee and Lanckriet , 2010), for Metric
Learning to Rank, is to learn a metric for a ranking task, wher e given a query instance,
one aims at producing a ranked list of examples where relevan t ones are ranked higher
than irrelevant ones. 28 Let P the set of all permutations (i.e., possible rankings) over t he
training set. Given a Mahalanobis distance d2
M and a query x, the predicted ranking p ∈ P
consists in sorting the instances by ascending d2
M (x, ·). The metric learning M is based on
Structural SVM (
Tsochantaridis et al. , 2005):
min
M ∈ Sd
+
∥M ∥∗ + C
∑
i
ξi
s.t. ⟨M,ψ (xi,p i) − ψ (xi,p )⟩F ≥ ∆(pi,p ) − ξi ∀i ∈ { 1,...,n },p ∈ P,
(11)
where ∥M ∥∗ = tr( M ) is the nuclear norm, C ≥ 0 the trade-oﬀ parameter, ⟨A, B⟩F =∑
i,jAijBij the Frobenius inner product, ψ : R × P → Sd the feature encoding of an input-
output pair ( xi,p ),
29 and ∆(pi,p ) ∈ [0, 1] the “margin” representing the loss of predicting
ranking p instead of the true ranking pi. In other words, ∆( pi,p ) assesses the quality of
rankingp with respect to the best ranking pi and can be evaluated using several measures,
such as the Area Under the ROC Curve (AUC), Precision-at- k or Mean Average Precision
(MAP). Since the number of constraints is super-exponentia l in the number of training
instances, the authors solve ( 11) using a 1-slack cutting-plane approach ( Joachims et al. ,
2009) which essentially iteratively optimizes over a small set o f active constraints (adding
the most violated ones at each step) using subgradient desce nt. However, the algorithm
requires a full eigendecomposition of M at each iteration, thus MLR does not scale well
with the dimensionality of the data. In practice, it is compe titive with other metric learning
algorithms for k-NN classiﬁcation and a structural SVM algorithm for rankin g, and can
induce low-rank solutions due to the nuclear norm. Lim et al. (2013) propose R-MLR, an
extension to MLR to deal with the presence of noisy features 30 using the mixed L2,1 norm
as in SML ( Ying et al. , 2009). R-MLR is shown to be able to ignore most of the irrelevant
features and outperforms MLR in this situation.
4. Other Advances in Metric Learning
So far, we focused on (linear) Mahalanobis metric learning w hich has inspired a large amount
of work during the past ten years. In this section, we cover ot her advances and trends in
metric learning for feature vectors. Most of the section is d evoted to (fully and weakly)
supervised methods. In Section
4.1, we address linear similarity learning. Section 4.2 deals
with nonlinear metric learning (including the kernelizati on of linear methods), Section 4.3
28. Source code is available at: http://www-cse.ucsd.edu/~bmcfee/code/mlr
29. The feature map ψ is designed such that the ranking p which maximizes ⟨M,ψ (x,p )⟩F is the one given
by ascending d2
M (x, ·).
30. Notice that this is diﬀerent from noisy side information , which was investigated by the method RML
(
Huang et al. , 2010) presented earlier in this section.
21
Bellet, Habrard and Sebban
with local metric learning and Section 4.4 with metric learning for histogram data. Sec-
tion 4.5 presents the recently-developed frameworks for deriving g eneralization guarantees
for supervised metric learning. We conclude this section wi th a review of semi-supervised
metric learning (Section 4.6).
4.1 Linear Similarity Learning
Although most of the work in linear metric learning has focus ed on the Mahalanobis dis-
tance, other linear measures, in the form of similarity func tions, have recently attracted
some interest. These approaches are often motivated by the p erspective of more scalable
algorithms due to the absence of PSD constraint.
SiLA (Qamar et al.) SiLA (
Qamar et al. , 2008) is an approach for learning similarity
functions of the following form:
KM (x, x′) = xT M x′
N (x, x′),
where M ∈ Rd× d is not required to be PSD nor symmetric, and N (x, x′) is a normaliza-
tion term which depends on x and x′. This similarity function can be seen as a gener-
alization of the cosine similarity, widely used in text and i mage retrieval (see for instance
Baeza-Yates and Ribeiro-Neto , 1999; Sivic and Zisserman , 2009). The authors build on the
same idea of “target neighbors” that was introduced in LMNN, but optimize the similarity
in an online manner with an algorithm based on voted perceptr on. At each step, the algo-
rithm goes through the training set, updating the matrix whe n an example does not satisfy
a criterion of separation. The authors present theoretical results that follow from the voted
perceptron theory in the form of regret bounds for the separa ble and inseparable cases.
In subsequent work, Qamar and Gaussier (2012) study the relationship between SiLA and
RELIEF, an online feature reweighting algorithm.
gCosLA (Qamar & Gaussier) gCosLA (
Qamar and Gaussier , 2009) learns generalized
cosine similarities of the form
KM (x, x′) = xT M x′
√
xT M x
√
x′T M x′
,
where M ∈ Sd
+. It corresponds to a cosine similarity in the projection spa ce implied by
M . The algorithm itself, an online procedure, is very similar to that of POLA (presented
in Section
3.4). Indeed, they essentially use the same loss function and al so have a two-
step approach: a projection onto the set of arbitrary matric es that achieve zero loss on
the current example pair, followed by a projection back onto the PSD cone. The ﬁrst
projection is diﬀerent from POLA (since the generalized cosi ne has a normalization factor
that depends on M ) but the authors manage to derive a closed-form solution. Th e second
projection is based on a full eigenvalue decomposition of M , making the approach costly
as dimensionality grows. A regret bound for the algorithm is provided and it is shown
experimentally that gCosLA converges in fewer iterations t han SiLA and is generally more
accurate. Its performance is competitive with LMNN and ITML . Note that Nguyen and Bai
(2010) optimize the same form of similarity based on a nonconvex fo rmulation.
22
A Survey on Metric Learning for Feature Vectors and Structur ed Data
OASIS (Chechik et al.) OASIS31 (Chechik et al. , 2009, 2010) learns a bilinear similarity
with a focus on large-scale problems. The bilinear similari ty has been used for instance in
image retrieval ( Deng et al. , 2011) and has the following simple form:
KM (x, x′) = xT M x′,
where M ∈ Rd× d is not required to be PSD nor symmetric. In other words, it is r elated to
the (generalized) cosine similarity but does not include no rmalization nor PSD constraint.
Note that when M is the identity matrix,KM amounts to an unnormalized cosine similarity.
The bilinear similarity has two advantages. First, it is eﬃc iently computable for sparse
inputs: if x and x′ havek1 andk2 nonzero features, KM (x, x′) can be computed in O(k1k2)
time. Second, unlike the Mahalanobis distance, it can deﬁne a similarity measure between
instances of diﬀerent dimension (for example, a document and a query) if a rectangular
matrix M is used. Since M ∈ Rd× d is not required to be PSD, Chechik et al. are able
to optimize KM in an online manner using a simple and eﬃcient algorithm, whi ch belongs
to the family of Passive-Aggressive algorithms ( Crammer et al. , 2006). The initialization is
M = I, then at each step t, the algorithm draws a triplet ( xi, xj, xk) ∈ R and solves the
following convex problem:
M t = arg min
M ,ξ
1
2 ∥M − M t− 1∥2
F +Cξ
s.t. 1 − d2
M (xi, xj) +d2
M (xi, xk) ≤ ξ
ξ ≥ 0,
(12)
whereC ≥ 0 is the trade-oﬀ parameter between minimizing the loss and s taying close from
the matrix obtained at the previous step. Clearly, if 1 − d2
M (xi, xj) + d2
M (xi, xk) ≤ 0,
then M t = M t− 1 is the solution of (
12). Otherwise, the solution is obtained from a
simple closed-form update. In practice, OASIS achieves com petitive results on medium-
scale problems and unlike most other methods, is scalable to problems with millions of
training instances. However, it cannot incorporate comple x regularizers. Note that the
same authors derived two more algorithms for learning bilin ear similarities as applications
of more general frameworks. The ﬁrst one is based on online le arning in the manifold of
low-rank matrices ( Shalit et al. , 2010, 2012) and the second on adaptive regularization of
weight matrices ( Crammer and Chechik , 2012).
SLLC (Bellet et al.) Similarity Learning for Linear Classiﬁcation ( Bellet et al. , 2012b)
takes an original angle by focusing on metric learning for li near classiﬁcation. As opposed
to pair and triplet-based constraints used in other approac hes, the metric is optimized to
be (ǫ,γ,τ )-good ( Balcan et al. , 2008a), a property based on an average over some points
which has a deep connection with the performance of a sparse l inear classiﬁer built from
such a similarity. SLLC learns a bilinear similarity KM and is formulated as an eﬃcient
unconstrained quadratic program:
min
M ∈ Rd× d
1
n
n∑
i=1
ℓ(1 − yi
1
γ|R|
∑
xj∈R
yjKM (xi, xj)) + β ∥M ∥2
F, (13)
31. Source code available at: http://ai.stanford.edu/~gal/Research/OASIS/
23
Bellet, Habrard and Sebban
where R is a set of reference points randomly selected from the train ing sample, γ is the
margin parameter, ℓ is the hinge loss and β the regularization parameter. Problem ( 13)
essentially learns KM such that training examples are more similar on average to re ference
points of the same class than to reference points of the oppos ite class by a margin γ. In
practice, SLLC is competitive with traditional metric lear ning methods, with the additional
advantage of inducing extremely sparse classiﬁers. A drawb ack of the approach is that linear
classiﬁers (unlike k-NN) cannot naturally deal with the multi-class setting, an d thus one-
vs-all or one-vs-one strategies must be used.
RSL (Cheng) As OASIS and SLLC,
Cheng (2013) also proposes to learn a bilinear
similarity, but focuses on the setting of pair matching (pre dicting whether two pairs are
similar). Pairs are of the form ( x, x′), where x ∈ Rd and x′ ∈ Rd′
potentially have
diﬀerent dimensionality, thus one has to learn a rectangular matrix M ∈ Rd× d′
. This
is a relevant setting for matching instances from diﬀerent do mains, such as images with
diﬀerent resolutions, or queries and documents. The matrix M is set to have ﬁxed rank
r ≪ min(d,d ′). RSL (Riemannian Similarity Learning) is formulated as fo llows:
max
M ∈ Rd× d′
∑
(xi,xj )∈S∪D
ℓ(xi, xj,y ij) + ∥M ∥F
s.t. rank( M ) = r,
(14)
where ℓ is some diﬀerentiable loss function (such as the log loss or th e squared hinge loss).
The optimization is carried out eﬃciently using recent adva nces in optimization over Rie-
mannian manifolds ( Absil et al. , 2008) and based on the low-rank factorization of M . At
each iteration, the procedure ﬁnds a descent direction in th e tangent space of the current
solution, and a retractation step to project the obtained ma trix back to the low-rank man-
ifold. It outputs a local minimum of ( 14). Experiments are conducted on pair-matching
problems where RSL achieves state-of-the-art results usin g a small rank matrix.
4.2 Nonlinear Methods
As we have seen, work in supervised metric learning has focus ed on linear metrics because
they are more convenient to optimize (in particular, it is ea sier to derive convex formulations
with the guarantee of ﬁnding the global optimum) and less pro ne to overﬁtting. In some
cases, however, there is nonlinear structure in the data tha t linear metrics are unable to
capture. The kernelization of linear methods can be seen as a satisfactory solution to this
problem. This strategy is explained in Section
4.2.1. The few approaches consisting in
directly learning nonlinear forms of metrics are addressed in Section 4.2.2.
4.2.1 Kernelization of Linear Methods
The idea of kernelization is to learn a linear metric in the no nlinear feature space induced
by a kernel function and thereby combine the best of both worl ds, in the spirit of what
is done in SVM. Some metric learning approaches have been sho wn to be kernelizable
(see for instance
Schultz and Joachims , 2003; Shalev-Shwartz et al. , 2004; Hoi et al. , 2006;
Torresani and Lee , 2006; Davis et al. , 2007) using speciﬁc arguments, but in general ker-
nelizing a particular metric algorithm is not trivial: a new formulation of the problem has
24
A Survey on Metric Learning for Feature Vectors and Structur ed Data
to be derived, where interface to the data is limited to inner products, and sometimes a
diﬀerent implementation is necessary. Moreover, when kerne lization is possible, one must
learn a n× n matrix. As the number of training examples n gets large, the problem becomes
intractable.
Recently though, several authors ( Chatpatanasiri et al. , 2010; Zhang et al. , 2010) have
proposed general kernelization methods based on Kernel Pri ncipal Component Analysis
(Sch¨ olkopf et al., 1998), a nonlinear extension of PCA ( Pearson, 1901). In short, KPCA im-
plicitly projects the data into the nonlinear (potentially inﬁnite-dimensional) feature space
induced by a kernel and performs dimensionality reduction i n that space. The (unchanged)
metric learning algorithm can then be used to learn a metric i n that nonlinear space—this is
referred to as the “KPCA trick”. Chatpatanasiri et al. ( 2010) showed that the KPCA trick
is theoretically sound for unconstrained metric learning a lgorithms (they prove representer
theorems). Another trick (similar in spirit in the sense tha t it involves some nonlinear
preprocessing of the feature space) is based on kernel densi ty estimation and allows one to
deal with both numerical and categorical attributes ( He et al. , 2013). General kernelization
results can also be obtained from the equivalence between Ma halanobis distance learning
in kernel space and linear transformation kernel learning ( Jain et al. , 2010, 2012), but are
restricted to spectral regularizers. Lastly, Wang et al. (2011) address the problem of choos-
ing an appropriate kernel function by proposing a multiple k ernel framework for metric
learning.
Note that kernelizing a metric learning algorithm may drast ically improve the quality of
the learned metric on highly nonlinear problems, but may als o favor overﬁtting (because pair
or triplet-based constraints become much easier to satisfy in a nonlinear, high-dimensional
kernel space) and thereby lead to poor generalization perfo rmance.
4.2.2 Learning Nonlinear Forms of Metrics
A few approaches have tackled the direct optimization of non linear forms of metrics. These
approaches are subject to local optima and more inclined to o verﬁt the data, but have the
potential to signiﬁcantly outperform linear methods on som e problems.
LSMD (Chopra et al.)
Chopra et al. (2005) pioneered the nonlinear metric learning
literature. They learn a nonlinear projection GW (x) parameterized by a vector W such
that the L1 distance in the low-dimensional target space ∥GW (x) − GW (x′)∥1 is small for
positive pairs and large for negative pairs. No assumption i s made about the nature of
GW : the parameter W corresponds to the weights in a convolutional neural networ k and
can thus be an arbitrarily complex nonlinear mapping. These weights are learned through
back-propagation and stochastic gradient descent so as to m inimize a loss function designed
to make the distance for positive pairs smaller than the dist ance of negative pairs by a
given margin. Due to the use of neural networks, the approach suﬀers from local optimality
and needs careful tuning of the many hyperparameters, requi ring a signiﬁcant amount of
validation data in order to avoid overﬁtting. This leads to a high computational complexity.
Nevertheless, the authors demonstrate the usefulness of LS MD on face veriﬁcation tasks.
NNCA (Salakhutdinov & Hinton) Nonlinear NCA ( Salakhutdinov and Hinton , 2007)
is another distance learning approach based on deep learnin g. NNCA ﬁrst learns a nonlinear,
low-dimensional representation of the data using a deep bel ief network (stacked Restricted
25
Bellet, Habrard and Sebban
Boltzmann Machines) that is pretrained layer-by-layer in a n unsupervised way. In a second
step, the parameters of the last layer are ﬁne-tuned by optim izing the NCA objective
(Section 3.2). Additional unlabeled data can be used as a regularizer by m inimizing their
reconstruction error. Although it suﬀers from the same limit ations as LSMD due to its deep
structure, NNCA is shown to perform well when enough data is a vailable. For instance,
on a digit recognition dataset, NNCA based on a 30-dimension al nonlinear representation
signiﬁcantly outperforms k-NN in the original pixel space as well as NCA based on a linear
space of same dimension.
SVML (Xu et al.)
Xu et al. (2012) observe that learning a Mahalanobis distance with
an existing algorithm and plugging it into a RBF kernel does n ot signiﬁcantly improve SVM
classiﬁcation performance. They instead propose Support V ector Metric Learning (SVML),
an algorithm that alternates between (i) learning the SVM mo del with respect to the current
Mahalanobis distance and (ii) learning a Mahalanobis dista nce that minimizes a surrogate
of the validation error of the current SVM model. Since the la tter step is nonconvex in
any event (due to the nonconvex loss function), the authors o ptimize the distance based on
the decomposition LT L, thus there is no PSD constraint and the approach can be made
low-rank. Frobenius regularization on L may be used to avoid overﬁtting. The optimization
procedure is done using a gradient descent approach and is ra ther eﬃcient although subject
to local minima. Nevertheless, SVML signiﬁcantly improves standard SVM results.
GB-LMNN (Kedem et al.) Kedem et al. (2012) propose Gradient-Boosted LMNN, a
nonlinear method consisting in generalizing the Euclidean distance with a nonlinear trans-
formation φ as follows:
dφ(x, x′) = ∥φ(x) − φ(x′)∥2.
This nonlinear mapping takes the form of an additive functio n φ =φ 0 +α ∑T
t=1ht, where
h1,...,h T are gradient boosted regression trees (
Friedman, 2001) of limited depth p and
φ 0 corresponds to the mapping learned by linear LMNN. They once again use the same
objective function as LMNN and are able to do the optimizatio n eﬃciently, building on
gradient boosting. On an intuitive level, the tree selected by gradient descent at each
iteration divides the space into 2 p regions, and instances falling in the same region are
translated by the same vector—thus examples in diﬀerent regi ons are translated in diﬀerent
directions. Dimensionality reduction can be achieved by le arning trees with r-dimensional
output. In practice, GB-LMNN seems quite robust to overﬁtti ng and performs well, often
achieving comparable or better performance than LMNN and IT ML.
HDML (Norouzi et al.) Hamming Distance Metric Learning ( Norouzi et al. , 2012a)
proposes to learn mappings from real-valued vectors to bina ry codes on which the Hamming
distance performs well. 32 Recall that the Hamming distance dH between two binary codes
of same length is simply the number of bits on which they disag ree. A great advantage
of working with binary codes is their small storage cost and t he fact that exact neighbor
search can be done in sublinear time ( Norouzi et al. , 2012b). The goal here is to optimize a
mapping b(x) that projects a d-dimensional real-valued input x to a q-dimensional binary
32. Source code available at: https://github.com/norouzi/hdml
26
A Survey on Metric Learning for Feature Vectors and Structur ed Data
code. The mapping takes the general form:
b(x; w) = sign (f (x; w)),
where f : Rd → Rq can be any function diﬀerentiable in w, sign( ·) is the element-wise
sign function and w is a real-valued vector representing the parameters to be le arned. For
instance,f can be a nonlinear transform obtained with a multilayer neur al network. Given
a relative constraint ( xi, xj, xk) ∈ R , denote by hi, hj and hk their corresponding binary
codes given by b. The loss is then given by
ℓ(hi, hj, hk) = [1 − dH (hi, hk) +dH (hi, hj)]+.
In the other words, the loss is zero when the Hamming distance between hi and hj is a
at least one bit smaller than the distance between hi and hk. HDML is formalized as
a loss minimization problem with L2 norm regularization on w. This objective function
is nonconvex and discontinuous, but the authors propose to o ptimize a continuous upper
bound on the loss which can be computed in O(q2) time, which is eﬃcient as long as the
code length q remains small. In practice, the objective is optimized usin g a stochastic
gradient descent approach. Experiments show that relative ly short codes obtained by non-
linear mapping are suﬃcient to achieve few constraint viola tions, and that a k-NN classiﬁer
based on these codes can achieve competitive performance wi th state-of-the-art classiﬁers.
Neyshabur et al. (2013) later showed that using asymmetric codes can lead to shorte r en-
codings while maintaining similar performance.
4.3 Local Metric Learning
The methods studied so far learn a global (linear or nonlinea r) metric. However, if the data
is heterogeneous, a single metric may not well capture the co mplexity of the task and it
might be beneﬁcial to use multiple local metrics that vary ac ross the space (e.g., one for
each class or for each instance).
33 This can often be seen as approximating the geodesic
distance deﬁned by a metric tensor (see Ramanan and Baker , 2011, for a review on this
matter). It is typically crucial that the local metrics be le arned simultaneously in order to
make them meaningfully comparable and also to alleviate ove rﬁtting. Local metric learning
has been shown to signiﬁcantly outperform global methods on some problems, but typically
comes at the expense of higher time and memory requirements. Furthermore, they usually
do not give rise to a consistent global metric, although some recent work partially addresses
this issue ( Zhan et al. , 2009; Hauberg et al. , 2012).
M2-LMNN (W einberger & Saul) Multiple Metrics LMNN 34 (Weinberger and Saul ,
2008, 2009) learns several Mahalanobis distances in diﬀerent parts of t he space. As a pre-
processing step, training data is partitioned in C clusters. These can be obtained either in
a supervised way (using class labels) or without supervisio n (e.g., using K-Means). Then,
C metrics (one for each cluster) are learned in a coupled fashi on in the form of a general-
ization of the LMNN’s objective, where the distance to a targ et neighbor or an impostor
33. The work of Frome et al. (2007) is one of the ﬁrst to propose to learn multiple local metrics . However,
their approach is speciﬁc to computer vision so we chose not t o review it here.
34. Source code available at: http://www.cse.wustl.edu/~kilian/code/code.html
27
Bellet, Habrard and Sebban
x is measured under the local metric associated with the clust er to which x belongs. In
practice, M 2-LMNN can yield signiﬁcant improvements over standard LMNN (especially
with supervised clustering), but this comes at the expense o f a higher computational cost,
and important overﬁtting (since each local metric can be ove rly speciﬁc to its region) unless
a large validation set is used ( Wang et al. , 2012c).
GLML (Noh et al.) The work of Noh et al. (2010), Generative Local Metric Learning,
aims at leveraging the power of generative models (known to o utperform purely discrimi-
native models when the training set is small) in the context o f metric learning. They focus
on nearest neighbor classiﬁcation and express the expected error of a 1-NN classiﬁer as the
sum of two terms: the asymptotic probability of misclassiﬁc ation and a metric-dependent
term representing the bias due to ﬁnite sampling. They show t hat this bias can be min-
imized locally by learning a Mahalanobis distance dMi at each training point xi. This is
done by solving, for each training instance, an independent semideﬁnite program that has
an analytical solution. Each matrix M i is further regularized towards a diagonal matrix in
order to alleviate overﬁtting. Since each local metric is co mputed independently, GLML can
be very scalable. Its performance is competitive on some dat asets (where the assumption
of Gaussian distribution to model the distribution of data i s reasonable) but can perform
very poorly on more complex problems ( Wang et al. , 2012c). Note that GLML does not
straightforwardly extend to the k-NN setting for k >1. Shi et al. (2011) use GLML metrics
as base kernels to learn a global kernel in a discriminative m anner.
Bk-means (W u et al.) Wu et al. (2009, 2012) propose to learn Bregman distances (or
Bregman divergences), a family of metrics that do not necess arily satisfy the triangle in-
equality or symmetry ( Bregman, 1967). Given the strictly convex and twice diﬀerentiable
function ϕ : Rd → R, the Bregman distance is deﬁned as:
dϕ(x, x′) = ϕ (x) − ϕ (x′) − (x − x′)T ∇ ϕ (x′).
It generalizes many widely-used measures: the Mahalanobis distance is recovered by setting
ϕ (x) = 1
2 xT M x, the KL divergence ( Kullback and Leibler , 1951) by choosing ϕ (p) =∑d
i=1pi logpi (here, p is a discrete probability distribution), etc. Wu et al. cons ider the
following symmetrized version:
dϕ(x, x′) =
(
∇ ϕ (x) − ∇ ϕ (x′)
)T (x − x′)
= ( x − x′)T ∇ 2ϕ (˜x)(x − x′),
where ˜x is a point on the line segment between x and x′. Therefore, dϕ amounts to
a Mahalanobis distance parameterized by the Hessian matrix of ϕ which depends on the
location of x and x′. In this respect, learning ϕ can be seen as learning an inﬁnite number of
local Mahalanobis distances. They take a nonparametric app roach by assuming φ to belong
to a Reproducing Kernel Hilbert Space HK associated to a kernel function K(x, x′) =
h(xT x′) where h(z) is a strictly convex function (set to exp( z) in the experiments). This
allows the derivation of a representer theorem. Setting ϕ (x) = ∑n
i=1α ih(xT
i x) leads to the
following formulation based on classic positive/negative pairs:
min
α∈ Rn
+,b
1
2 αT Kα + C
∑
(xi,xj )∈S∪D
ℓ(yij [dϕ(xi, xj) − b]), (15)
28
A Survey on Metric Learning for Feature Vectors and Structur ed Data
where K is the Gram matrix, ℓ(t) = max(0, 1 − t) is the hinge loss and C is the trade-oﬀ
parameter. Problem ( 15) is solved by a simple subgradient descent approach where ea ch
iteration has a linear complexity. Note that ( 15) only has n + 1 variables instead of d2
in most metric learning formulations, leading to very scala ble learning. The downside is
that computing the learned distance requires n kernel evaluations, which can be expensive
for large datasets. The method is evaluated on clustering pr oblems and exhibits good
performance, matching or improving that of other metric lea rning approaches.
PLML (W ang et al.) Wang et al. (2012c) propose PLML, 35 a Parametric Local Metric
Learning method where a Mahalanobis metric d2
Mi is learned for each training instance xi:
d2
M i(xi, xj) = ( xi − xj)T M i(xi − xj).
M i is parameterized to be a weighted linear combination of metr ic bases M b1,..., M b2,
where M bj ⪰ 0 is associated with an anchor point uj.
36 In other words, M i is deﬁned as:
M i =
m∑
j=1
Wibj M bj, W i,bj ≥ 0,
m∑
j=1
Wibj = 1,
where the nonnegativity of the weights ensures that the comb ination is PSD. The weight
learning procedure is a trade-oﬀ between three terms: (i) ea ch point x should be close to
its linear approximation ∑m
j=1Wibj uj, (ii) the weighting scheme should be local (i.e., Wibj
should be large if xi and ui are similar), and (iii) the weights should vary smoothly ove r
the data manifold (i.e., similar training instances should be assigned similar weights).
37
Given the weights, the basis metrics M b1,..., M bm are then learned in a large-margin
fashion using positive and negative training pairs and Frob enius regularization. In terms
of scalability, the weight learning procedure is fairly eﬃc ient. However, the metric bases
learning procedure requires at each step an eigen-decompos ition that scales in O(d3), mak-
ing the approach intractable for high-dimensional problem s. In practice, PLML performs
very well on the evaluated datasets, and is quite robust to ov erﬁtting due to its global
manifold regularization. However, like LMNN, PLML is sensi tive to the relevance of the
Euclidean distance to assess the similarity between (ancho r) points. Note that PLML has
many hyper-parameters but in the experiments the authors us e default values for most of
them. Huang et al. (2013) propose to regularize the anchor metrics to be low-rank and use
alternating optimization to solve the problem.
RFD (Xiong et al.) The originality of the Random Forest Distance (
Xiong et al. , 2012)
is to see the metric learning problem as a pair classiﬁcation problem.38 Each pair of examples
(x, x′) is mapped to the following feature space:
φ(x, x′) =
[
|x − x′|
1
2 (x + x′)
]
∈ R2d.
35. Source code available at: http://cui.unige.ch/~wangjun/papers/PLML.zip
36. In practice, these anchor points are deﬁned as the means o f clusters constructed by the K-Means algo-
rithm.
37. The weights of a test instance can be learned by optimizin g the same trade-oﬀ given the weights of the
training instances, and simply set to the weights of the near est training instance.
38. Source code available at: http://www.cse.buffalo.edu/~cxiong/RFD_Package.zip
29
Bellet, Habrard and Sebban
The ﬁrst part of φ(x, x′) encodes the relative position of the examples and the secon d part
their absolute position, as opposed to the implicit mapping of the Mahalanobis distance
which only encodes relative information. The metric is base d on a random forest F , i.e.,
dRF D(x, x′) = F (φ(x, x′)) = 1
T
T∑
t=1
ft(φ(x, x′)),
where ft(·) ∈ { 0, 1} is the output of decision tree t. RFD is thus highly nonlinear and is
able to implicitly adapt the metric throughout the space: wh en a decision tree in F selects a
node split based on a value of the absolute position part, the n the entire sub-tree is speciﬁc
to that region of R2d. As compared to other local metric learning methods, traini ng is very
eﬃcient: each tree takes O(n logn) time to generate and trees can be built in parallel. A
drawback is that the evaluation of the learned metric requir es to compute the output of
the T trees. The experiments highlight the importance of encodin g absolute information,
and show that RFD outperforms some global and local metric le arning methods on several
datasets and appears to be quite fast.
4.4 Metric Learning for Histogram Data
Histograms are feature vectors that lie on the probability s implex S d. This representation
is very common in areas dealing with complex objects, such as natural language processing,
computer vision or bioinformatics: each instance is repres ented as a bag of features, i.e.,
a vector containing the frequency of each feature in the obje ct. Bags-of(-visual)-words
(
Salton et al. , 1975; Li and Perona , 2005) are a common example of such data. We present
here three metric learning methods designed speciﬁcally fo r histograms.
χ 2-LMNN (Kedem et al.) Kedem et al. (2012) propose χ 2-LMNN, which is based on
a simple yet prominent histogram metric, the χ 2 distance ( Hafner et al. , 1995), deﬁned as
χ 2(x, x′) = 1
2
d∑
i=1
(xi − x′i)2
xi +x′i , (16)
wherexi denotes the ith feature of x.39 Note that χ 2 is a (nonlinear) proper distance. They
propose to generalize this distance with a linear transform ation, introducing the following
pseudo-distance:
χ 2
L(x, x′) = χ 2(Lx, Lx′),
where L ∈ Rr× d, with the constraint that L maps any x onto S d (the authors show that this
can be enforced using a simple trick). The objective functio n is the same as LMNN
40 and
is optimized using a standard subgradient descent procedur e. Although subject to local
optima, experiments show great improvements on histogram d ata compared to standard
histogram metrics and Mahalanobis distance learning metho ds, and promising results for
dimensionality reduction (when r<d ).
39. The sum in ( 16) must be restricted to entries that are nonzero in either x or x′ to avoid division by zero.
40. To be precise, it requires an additional parameter. In st andard LMNN, due to the linearity of the
Mahalanobis distance, solutions obtained with diﬀerent va lues of the margin only diﬀer up to a scaling
factor—the margin is thus set to 1. Here, χ 2 is nonlinear and therefore this value must be tuned.
30
A Survey on Metric Learning for Feature Vectors and Structur ed Data
GML (Cuturi & Avis) While χ 2-LMNN optimizes a simple bin-to-bin histogram dis-
tance, Cuturi and Avis (2011) propose to consider the more powerful cross-bin Earth Move r’s
Distance (EMD) introduced by Rubner et al. (2000), which can be seen as the distance be-
tween a source histogram x and a destination histogram x′. On an intuitive level, x is
viewed as piles of earth at several locations (bins) and x′ as several holes, where the value
of each feature represents the amount of earth and the capaci ty of the hole respectively.
The EMD is then equal to the minimum amount of eﬀort needed to mo ve all the earth from
x to x′. The costs of moving one unit of earth from bin i of x to bin j of x′ is encoded
in the so-called ground distance matrix D ∈ Rd× d.41 The computation of EMD amounts
to ﬁnding the optimal ﬂow matrix F , where fij corresponds to the amount of earth moved
from bin i of x to bin j of x′. Given the ground distance matrix D, EMD D(x, x′) is linear
and can be formulated as a linear program:
EMDD(x, x′) = min
f ∈ C(x,x′)
dT f,
where f and d are respectively the ﬂow and the ground matrices rewritten a s vectors
for notational simplicity, and C(x, x′) is the convex set of feasible ﬂows (which can be
represented as linear constraints). Ground Metric Learnin g (GML) aims at learning D
based on training triplets ( xi, xj,w ij) where xi and xj are two histograms and wij ∈ R is
a weight quantifying the similarity between xi and xj. The optimized criterion essentially
aims at minimizing the sum of wijEMDD(xi, xj) — which is a nonlinear function in D —
by casting the problem as a diﬀerence of two convex functions. A local minima is found
eﬃciently by a subgradient descent approach. Experiments o n image datasets show that
GML outperforms standard histogram distances as well as Mah alanobis distance methods.
EMDL (W ang & Guibas) Building on GML and successful Mahalanobis distance learn-
ing approaches such as LMNN, Wang and Guibas (2012) aim at learning the EMD ground
matrix in the more ﬂexible setting where the algorithm is pro vided with a set of relative
constraints R that must be satisﬁed with a large margin. The problem is form ulated as
min
D∈ D
∥D∥2
F + C
∑
i,j,k
ξijk
s.t. EMD D(xi, xk) − EMDD(xi, xj) ≥ 1 − ξijk ∀(xi, xj, xk) ∈ R,
(17)
where D =
{
D ∈ Rd× d : ∀i,j ∈ { 1,...,d },d ij ≥ 0,d ii = 0
}
and C ≥ 0 is the trade-oﬀ pa-
rameter.
42 The authors also propose a pair-based formulation. Problem (17) is bi-convex
and is solved using an alternating procedure: ﬁrst ﬁx the gro und metric and solve for
the ﬂow matrices (this amounts to a set of standard EMD proble ms), then solve for the
ground matrix given the ﬂows (this is a quadratic program). T he algorithm stops when
the changes in the ground matrix are suﬃciently small. The pr ocedure is subject to local
optima (because ( 17) is not jointly convex) and is not guaranteed to converge: th ere is a
need for a trade-oﬀ parameter α between stable but conservative updates (i.e., staying clo se
41. For EMD to be proper distance, D must satisfy the following ∀i,j,k ∈ { 1,...,d }: (i) dij ≥ 0, (ii) dii = 0,
(iii) dij =dji and (iv) dij ≤ dik +dkj .
42. Note that unlike in GML, D ∈ D may not be a valid distance matrix. In this case, EMD D is not a
proper distance.
31
Bellet, Habrard and Sebban
Underlying
distribution
Metric learning
algorithm
Metric-based
algorithm
Data
sample
Learned
metric
Learned
predictor
Prediction
Consistency guarantees for the learned metric
Generalization guarantees for the predictor that uses the metric
Figure 5: The two-fold problem of generalization in metric l earning. We may be interested
in the generalization ability of the learned metric itself: can we say anything about
its consistency on unseen data drawn from the same distribut ion? Furthermore,
we may also be interested in the generalization ability of th e predictor using that
metric: can we relate its performance on unseen data to the qu ality of the learned
metric?
to the previous ground matrix) and aggressive but less stabl e updates. Experiments on face
veriﬁcation datasets conﬁrm that EMDL improves upon standa rd histogram distances and
Mahalanobis distance learning methods.
4.5 Generalization Guarantees for Metric Learning
The derivation of guarantees on the generalization perform ance of the learned model is a
wide topic in statistical learning theory (
Vapnik and Chervonenkis , 1971; Valiant, 1984).
Assuming that data points are drawn i.i.d. from some (unknow n but ﬁxed) distribution
P , one essentially aims at bounding the deviation of the true risk of the learned model
(its performance on unseen data) from its empirical risk (its performance on the training
sample).43
In the speciﬁc context of metric learning, we claim that the q uestion of generalization
can be seen as two-fold ( Bellet, 2012), as illustrated by Figure 5:
• First, one may consider the consistency of the learned metric , i.e., trying to bound
the deviation between the empirical performance of the metr ic on the training sample
and its generalization performance on unseen data.
• Second, the learned metric is used to improve the performanc e of some prediction
model (e.g., k-NN or a linear classiﬁer). It would thus be meaningful to exp ress the
generalization performance of this predictor in terms of that of the learned metric.
As in the classic supervised learning setting (where traini ng data consist of individual
labeled instances), generalization guarantees may be deri ved for supervised metric learning
(where training data consist of pairs or triplets). Indeed, most of supervised metric learning
methods can be seen as minimizing a (regularized) loss funct ion ℓ based on the training
pairs/triplets. However, the i.i.d. assumption is violate d in the metric learning scenario
since the training pairs/triplets are constructed from the training sample. For this reason,
43. This deviation is typically a function of the number of tr aining examples and some notion of complexity
of the model.
32
A Survey on Metric Learning for Feature Vectors and Structur ed Data
establishing generalization guarantees for the learned me tric is challenging and only recently
has this question been investigated from a theoretical stan dpoint.
Metric consistency bounds for batch methods Given a training sample T = {zi =
(xi,y i)}n
i=1 drawn i.i.d. from an unknown distribution µ , let us consider fully supervised
Mahalanobis metric learning of the following general form:
min
M ∈ Sd
+
1
n2
∑
zi,zj ∈T
ℓ(d2
M, zi, zj) + λR(M ),
where R(M ) is the regularizer, λ the regularization parameter and the loss function ℓ is of
the form ℓ(d2
M, zi, zj) = g(yiyj[c − d2
M (xi, xj)]) with c > 0 a decision threshold variable
and g convex and Lipschitz continuous. This includes popular los s functions such as the
hinge loss. Several recent work have proposed to study the co nvergence of the empirical
risk (as measured by ℓ on pairs from T ) to the true risk over the unknown probability
distribution µ . The framework proposed by Bian & Tao (
2011; 2012) is quite rigid since it
relies on strong assumptions on the distribution of the exam ples and cannot accommodate
any regularization (a constraint to bound M is used instead). Jin et al. (2009) use a notion
of uniform stability ( Bousquet and Elisseeﬀ , 2002) adapted to the case of metric learning
(where training data is made of pairs to derive generalizati on bounds that are limited
to Frobenius norm regularization. Bellet and Habrard (2012) demonstrate how to adapt
the more ﬂexible notion of algorithmic robustness ( Xu and Mannor , 2012) to the metric
learning setting to derive (loose) generalization bounds f or any matrix norm (including
sparsity-inducing ones) as regularizer. They also show tha t a weak notion of robustness is
necessary and suﬃcient for metric learning algorithms to ge neralize well. Lastly, Cao et al.
(2012a) use a notion of Rademacher complexity ( Bartlett and Mendelson , 2002) dependent
on the regularizer to derive bounds for several matrix norms . All these results can easily
adapted to non-Mahalanobis linear metric learning formula tions.
Regret bound conversion for online methods Wang et al. (2012d, 2013b) deal with
the online learning setting. They show that existing proof t echniques to convert regret
bounds into generalization bounds (see for instance Cesa-Bianchi and Gentile , 2008) only
hold for univariate loss functions, but derive an alternati ve framework that can deal with
pairwise losses. At each round, the online algorithm receiv es a new instance and is assumed
to pair it with all previously-seen data points. As this is ex pensive or even infeasible in
practice, Kar et al. (2013) propose to use a buﬀer containing only a bounded number of the
most recent instances. They are also able to obtain tighter b ounds based on a notion of
Rademacher complexity, essentially adapting and extendin g the work of Cao et al. (2012a).
These results suggest that one can obtain generalization bo unds for most/all online metric
learning algorithms with bounded regret (such as those pres ented in Section 3.4).
Link between learned metric and classiﬁcation performance The second question
of generalization (i.e., at the classiﬁer level) remains an open problem for the most part.
To the best of our knowledge, it has only been addressed in the context of metric learn-
ing for linear classiﬁcation. Bellet et al. (2011, 2012a,b) rely upon the theory of learning
with (ǫ,γ,τ )-good similarity function ( Balcan et al. , 2008a), which makes the link between
properties of a similarity function and the generalization of a linear classiﬁer built from this
33
Bellet, Habrard and Sebban
similarity. Bellet et al. propose to use ( ǫ,γ,τ )-goodness as an objective function for metric
learning, and show that in this case it is possible to derive g eneralization guarantees not
only for the learned similarity but also for the linear class iﬁer. Guo and Ying (2014) extend
the results of Bellet et al. to several matrix norms using a Ra demacher complexity analysis,
based on techniques from Cao et al. (2012a).
4.6 Semi-Supervised Metric Learning Methods
In this section, we present two categories of metric learnin g methods that are designed to
deal with semi-supervised learning tasks. The ﬁrst one corr esponds to the standard semi-
supervised setting, where the learner makes use of unlabele d pairs in addition to positive and
negative constraints. The second one concerns approaches w hich learn metrics to address
semi-supervised domain adaptation problems where the lear ner has access to labeled data
drawn according to a source distribution and unlabeled data generated from a diﬀerent (but
related) target distribution.
4.6.1 Standard Semi-Supervised Setting
The following metric learning methods leverage the informa tion brought by the set of un-
labeled pairs, i.e., pairs of training examples that do not belong to the se ts of positive and
negative pairs:
U = {(xi, xj) : i ̸=j, (xi, xj) /∈ S ∪ D} .
An early approach by
Bilenko et al. (2004) combined semi-supervised clustering with
metric learning. In the following, we review general metric learning formulations that
incorporate information from the set of unlabeled pairs U .
LRML (Hoi et al.) Hoi et al. (2008, 2010) propose to follow the principles of mani-
fold regularization for semi-supervised learning ( Belkin and Niyogi , 2004) by resorting to a
weight matrix W that encodes the similarity between pairs of points. 44 Hoi et al. construct
W using the Euclidean distance as follows:
Wij =
{
1 if xi ∈ N (xj) or xj ∈ N (xi)
0 otherwise
where N (xj) denotes the nearest neighbor list of xj. Using W , they use the following
regularization known as the graph Laplacian regularizer:
1
2
n∑
i,j=1
d2
M (xi, xj)Wij = tr(XLX T M ),
where X is the data matrix and L = D − W is the graph Laplacian matrix with D a
diagonal matrix such that Dii = ∑
jWij. Intuitively, this regularization favors an “aﬃnity-
preserving” metric: the distance between points that are si milar according to W should
remain small according to the learned metric. Experiments s how that LRML (Laplacian
Regularized Metric Learning) signiﬁcantly outperforms su pervised methods when the side
44. Source code available at: http://www.ee.columbia.edu/~wliu/
34
A Survey on Metric Learning for Feature Vectors and Structur ed Data
information is scarce. An obvious drawback is that computin g W is intractable for large-
scale datasets. This work has inspired a number of extension s and improvements: Liu et al.
(2010) introduce a reﬁned way of constructing W while Baghshah and Shouraki (2009),
Zhong et al. (2011) and Wang et al. (2013a) use a diﬀerent (but similar in spirit) manifold
regularizer.
M-DML (Zha et al.) The idea of
Zha et al. (2009) is to augment Laplacian regulariza-
tion with metrics M 1,..., M K learned from auxiliary datasets. Formally, for each availa ble
auxiliary metric, a weight matrix W k is constructed following Hoi et al. (2008, 2010) but
using metric M k instead of the Euclidean distance. These are then combined t o obtain the
following regularizer:
K∑
k=1
α k tr(XL kX T M ),
where Lk is the Laplacian associated with weight matrix W k andα k is the weight reﬂecting
the utility of auxiliary metric M k. As such weights are diﬃcult to set in practice, Zha et
al. propose to learn them together with the metric M by alternating optimization (which
only converges to a local minimum). Experiments on a face rec ognition task show that
metrics learned from auxiliary datasets can be successfull y used to improve performance
over LRML.
SERAPH (Niu et al.)
Niu et al. (2012) tackle semi-supervised metric learning from
an information-theoretic perspective by optimizing a prob ability of labeling a given pair
parameterized by a Mahalanobis distance: 45
pM (y|x, x′) = 1
1 + exp
(
y(d2
M (x, x′) − η)
).
M is optimized to maximize the entropy of pM on the labeled pairs S ∪D and minimize it on
unlabeled pairs U , following the entropy regularization principle (
Grandvalet and Bengio ,
2004). Intuitively, the regularization enforces low uncertain ty of unobserved weak labels.
They also encourage a low-rank projection by using the trace norm. The resulting noncon-
vex optimization problem is solved using an EM-like iterati ve procedure where the M-step
involves a projection on the PSD cone. The proposed method ou tperforms supervised met-
ric learning methods when the amount of supervision is very s mall, but was only evaluated
against one semi-supervised method ( Baghshah and Shouraki , 2009) known to be subject
to overﬁtting.
4.6.2 Metric Learning for Domain Adaptation
In the domain adaptation (DA) setting (
Mansour et al. , 2009; Qui˜ nonero-Candela, 2009;
Ben-David et al. , 2010), the labeled training data and the test data come from diﬀere nt
(but somehow related) distributions (referred to as the sou rce and target distributions
respectively). This situation occurs very often in real-wo rld applications—famous examples
include speech recognition, spam detection and object reco gnition—and is also relevant
for metric learning. Although domain adaptation is sometim es achieved by using a small
45. Source code available at: http://sugiyama-www.cs.titech.ac.jp/~gang/software.html
35
Bellet, Habrard and Sebban
sample of labeled target data ( Saenko et al. , 2010; Kulis et al. , 2011), we review here the
more challenging case where only unlabeled target data is av ailable.
CDML (Cao et al.) CDML (Cao et al. , 2011), for Consistent Distance Metric Learning,
deals with the setting of covariate shift, which assumes tha t source and target data distri-
butions pS(x) and pT (x) are diﬀerent but the conditional distribution of the labels given
the features, p(y|x), remains the same. In the context of metric learning, the as sumption
is made at the pair level, i.e., p(yij|xi, xj) is stable across domains. Cao et al. show that if
some metric learning algorithm minimizing some training lo ss ∑
(xi,xj)∈S∪D ℓ(d2
M, xi, xj) is
asymptotically consistent without covariate shift, then t he following algorithm is consistent
under covariate shift:
min
M ∈ Sd
+
∑
(xi,xj)∈S∪D
wijℓ(d2
M, xi, xj), where wij = pT (xi)pT (xj)
pS(xi)pS(xj). (18)
Problem ( 18) can be seen as cost-sensitive metric learning, where the co st of each pair is
given by the importance weight wij. Therefore, adapting a metric learning algorithm to
covariate shift boils down to computing the importance weig hts, which can be done reliably
using unlabeled data ( Tsuboi et al. , 2008). The authors experiment with ITML and show
that their adapted version outperforms the regular one in si tuations of (real or simulated)
covariate shift.
DAML (Geng et al.) DAML (
Geng et al. , 2011), for Domain Adaptation Metric Learn-
ing, tackles the general domain adaptation setting. In this case, a classic strategy in DA
is to use a term that brings the source and target distributio n closer. Following this line
of work, Geng et al. regularize the metric using the empirica l Maximum Mean Discrep-
ancy (MMD, Gretton et al. , 2006), a nonparametric way of measuring the diﬀerence in
distribution between the source sample S and the target sample T :
MMD (S,T ) =






1
|S|
|S|∑
i=1
ϕ (xi) − 1
|T |
|T |∑
i=1
ϕ (x′
i)






2
H
,
whereϕ (x) is a nonlinear feature mapping function that maps x to the Reproducing Kernel
Hilbert Space H. The MMD can be computed eﬃciently using the kernel trick and can thus
be used as a (convex) regularizer in kernelized metric learn ing algorithms (see Section
4.2.1).
DAML is thus a trade-oﬀ between satisfying the constraints o n the labeled source data
and ﬁnding a projection that minimizes the discrepancy betw een the source and target
distribution. Experiments on face recognition and image an notation tasks in the DA setting
highlight the eﬀectiveness of DAML compared to classic metri c learning methods.
5. Metric Learning for Structured Data
In many domains, data naturally come structured, as opposed to the “ﬂat” feature vector
representation we have focused on so far. Indeed, instances can come in the form of strings,
such as words, text documents or DNA sequences; trees like XM L documents, secondary
structure of RNA or parse trees; and graphs, such as networks , 3D objects or molecules. In
36
A Survey on Metric Learning for Feature Vectors and Structur ed Data
Page Name Y ear Source Data Method Script Optimum Negative
Code Type Pairs
39 R&Y 1998 Yes String Generative+EM All Local No
39 O&S 2006 Yes String Discriminative+EM All Local No
40 Saigo 2006 Yes String Gradient Descent All Local No
40 GESL 2011 Yes All Gradient Descent Levenshtein Global Yes
41 Bernard 2006 Yes Tree Both+EM All Local No
41 Boyer 2007 Yes Tree Generative+EM All Local No
41 Dalvi 2009 No Tree Discriminative+EM All Local No
41 Emms 2012 No Tree Discriminative+EM Optimal Local No
41 N&B 2007 No Graph Generative+EM All Local No
Table 3: Main features of metric learning methods for struct ured data. Note that all meth-
ods make use of positive pairs.
the context of structured data, metrics are especially appe aling because they can be used
as a proxy to access data without having to manipulate these c omplex objects. Indeed,
given an appropriate structured metric, one can use any metr ic-based algorithm as if the
data consisted of feature vectors. Many of these metrics act ually rely on representing
structured objects as feature vectors, such as some string k ernels (see Lodhi et al. , 2002,
and variants) or bags-of-(visual)-words ( Salton et al. , 1975; Li and Perona , 2005). In this
case, metric learning can simply be performed on the feature vector representation, but
this strategy can imply a signiﬁcant loss of structural info rmation. On the other hand,
there exist metrics that operate directly on the structured objects and can thus capture
more structural distortions. However, learning such metri cs is challenging because most
of structured metrics are combinatorial by nature, which ex plains why it has received less
attention than metric learning from feature vectors. In thi s section, we focus on the edit
distance, which basically measures (in terms of number of op erations) the cost of turning
an object into another. Edit distance has attracted most of t he interest in the context
of metric learning for structured data because (i) it is deﬁn ed for a variety of objects:
sequences ( Levenshtein, 1966), trees ( Bille, 2005) and graphs ( Gao et al. , 2010), (ii) it is
naturally amenable to learning due to its parameterization by a cost matrix.
We review string edit distance learning in Section 5.1, while methods for trees and graphs
are covered in Section 5.2. The features of each approach are summarized in Table 3.
5.1 String Edit Distance Learning
In this section, we ﬁrst introduce some notations as well as t he string edit distance. We
then review the relevant metric learning methods.
5.1.1 Notations and Definitions
Deﬁnition 1 (Alphabet and string) An alphabet Σ is a ﬁnite nonempty set of symbols.
A string x is a ﬁnite sequence of symbols from Σ. The empty string/symbol is denoted by $
and Σ∗ is the set of all ﬁnite strings (including $) that can be generated from Σ. Finally,
the length of a string x is denoted by |x|.
37
Bellet, Habrard and Sebban
Figure 6: A memoryless stochastic transducer that models th e edit probability of any pair
of strings built from Σ = {a, b}. Edit probabilities assigned to each transition
are not shown here for the sake of readability.
Deﬁnition 2 (String edit distance) Let C be a nonnegative (|Σ|+ 1) × (|Σ|+ 1) matrix
giving the cost of the following elementary edit operations : insertion, deletion and substitu-
tion of a symbol, where symbols are taken from Σ∪{ $}. Given two strings x, x′ ∈ Σ∗, an edit
script is a sequence of operations that turns x into x′. The string edit distance ( Levenshtein,
1966) between x and x′ is deﬁned as the cost of the cheapest edit script and can be com puted
in O(|x| · |x′|) time by dynamic programming.
Similar metrics include the Needleman-Wunsch score ( Needleman and Wunsch , 1970)
and the Smith-Waterman score ( Smith and Waterman , 1981). These alignment-based mea-
sures use the same substitution operations as the edit dista nce, but a linear gap penalty
function instead of insertion/deletion costs.
The standard edit distance, often called Levenshtein edit d istance, is based on a unit
cost for all operations. However, this might not reﬂect the r eality of the considered task: for
example, in typographical error correction, the probabili ty that a user hits the Q key instead
of W on a QWERTY keyboard is much higher than the probability t hat he hits Q instead of
Y. For some applications, such as protein alignment or handw ritten digit recognition, hand-
tuned cost matrices may be available ( Dayhoﬀ et al. , 1978; Henikoﬀ and Henikoﬀ , 1992;
Mic´ o and Oncina, 1998). Otherwise, there is a need for automatically learning the cost
matrix C for the task at hand.
5.1.2 Stochastic String Edit Distance Learning
Optimizing the edit distance is challenging because the opt imal sequence of operations
depends on the edit costs themselves, and therefore updatin g the costs may change the
optimal edit script. Most general-purpose approaches get r ound this problem by consid-
ering a stochastic variant of the edit distance, where the co st matrix deﬁnes a probability
distribution over the edit operations. One can then deﬁne an edit similarity as the pos-
terior probability pe(x′|x) that an input string x is turned into an output string x′. This
corresponds to summing over all possible edit scripts that t urn x into x′ instead of only
considering the optimal script. Such a stochastic edit proc ess can be represented as a
probabilistic model, such as a stochastic transducer (Figu re
6), and one can estimate the
parameters of the model (i.e., the cost matrix) that maximiz e the expected log-likelihood
of positive pairs. This is done via an EM-like iterative proc edure ( Dempster et al. , 1977).
38
A Survey on Metric Learning for Feature Vectors and Structur ed Data
Note that unlike the standard edit distance, the obtained ed it similarity does not usually
satisfy the properties of a distance (in fact, it is often not symmetric and rarely satisﬁes the
triangular inequality).
Ristad and Yianilos The ﬁrst method for learning a string edit metric, in the form of
a generative model, was proposed by
Ristad and Yianilos (1998).46 They use a memory-
less stochastic transducer which models the joint probabil ity of a pair pe(x, x′) from which
pe(x′|x) can be estimated. Parameter estimation is performed with a n EM procedure. The
Expectation step takes the form of a probabilistic version o f the dynamic programing algo-
rithm of the standard edit distance. The M-step aims at maxim izing the likelihood of the
training pairs of strings so as to deﬁne a joint distribution over the edit operations:
∑
(u,v)∈ (Σ∪{ $})2\{ $,$}
Cuv +c(#) = 1, with c(#)> 0 and Cuv ≥ 0,
where # is a termination symbol and c(#) the associated cost (probability).
Note that Bilenko and Mooney (2003) extended this approach to the Needleman-Wunsch
score with aﬃne gap penalty and applied it to duplicate detec tion. To deal with the ten-
dency of Maximum Likelihood estimators to overﬁt when the nu mber of parameters is large
(in this case, when the alphabet size is large), Takasu (2009) proposes a Bayesian parameter
estimation of pair-HMM providing a way to smooth the estimat ion.
Oncina and Sebban The work of Oncina and Sebban (2006) describes three levels of
bias induced by the use of generative models: (i) dependence between edit operations,
(ii) dependence between the costs and the prior distributio n of strings pe(x), and (iii) the
fact that to obtain the posterior probability one must divid e by the empirical estimate of
pe(x). These biases are highlighted by empirical experiments co nducted with the method
of Ristad and Yianilos (1998). To address these limitations, they propose the use of a con -
ditional transducer as a discriminative model that directl y models the posterior probability
p(x′|x) that an input string x is turned into an output string x′ using edit operations. 46
Parameter estimation is also done with EM where the maximiza tion step diﬀers from that
of Ristad and Yianilos (1998) as shown below:
∀u ∈ Σ,
∑
v∈ Σ∪{ $}
Cv|u +
∑
v∈ Σ
Cv|$ = 1, with
∑
v∈ Σ
Cv|$ +c(#) = 1.
In order to allow the use of negative pairs, McCallum et al. (2005) consider another
discriminative model, conditional random ﬁelds, that can d eal with positive and negative
pairs in speciﬁc states, still using EM for parameter estima tion.
5.1.3 String Edit Distance Learning by Gradient Descent
The use of EM has two main drawbacks: (i) it may converge to a lo cal optimum, and
(ii) parameter estimation and distance calculations must b e done at each iteration, which
can be very costly if the size of the alphabet and/or the lengt h of the strings are large.
46. An implementation is available within the SEDiL platfor m ( Boyer et al. , 2008):
http://labh-curien.univ-st-etienne.fr/SEDiL/
39
Bellet, Habrard and Sebban
The following methods get round these drawbacks by formulat ing the learning problem in
the form of an optimization problem that can be eﬃciently sol ved by a gradient descent
procedure.
Saigo et al.
Saigo et al. (2006) manage to avoid the need for an iterative procedure
like EM in the context of detecting remote homology in protei n sequences. 47 They learn
the parameters of the Smith-Waterman score which is plugged in their local alignment
kernel kLA where all the possible local alignments π for changing x into x′ are taken into
account ( Saigo et al. , 2004):
kLA(x,x ′) =
∑
π
et·s(x,x′,π). (19)
In the above formula, t is a parameter and s(x,x ′,π ) is the corresponding score of π and
deﬁned as follows:
s(x,x ′,π ) =
∑
u,v∈ Σ
nu,v(x,x ′,π ) ·C uv − ngd(x,x ′,π ) ·gd − nge(x,x ′,π ) ·ge, (20)
where nu,v(x,x ′,π ) is the number of times that symbol u is aligned with v while gd and
ge, along with their corresponding number of occurrences ngd(x,x ′,π ) and nge(x,x ′,π ), are
two parameters dealing respectively with the opening and ex tension of gaps.
Unlike the Smith-Waterman score, kLA is diﬀerentiable and can be optimized by a
gradient descent procedure. The objective function that th ey optimize is meant to favor
the discrimination between positive and negative examples , but this is done by only using
positive pairs of distant homologs. The approach has two add itional drawbacks: (i) the
objective function is nonconvex and thus subject to local mi nima, and (ii) in general, kLA
does not fulﬁll the properties of a kernel.
GESL (Bellet et al.)
Bellet et al. (2011, 2012a) propose a convex programming ap-
proach to learn edit similarity functions from both positiv e and negative pairs without
requiring a costly iterative procedure. 48 They use the following simpliﬁed edit function:
eC(x, x′) =
∑
(u,v)∈ (Σ∪{ $})2\{ $,$}
C uv ·#uv(x, x′),
where # uv(x, x′) is the number of times the operation u → v appears in the Levenshtein
script. Therefore, eC can be optimized directly since the sequence of operations i s ﬁxed (it
does not depend on the costs). The authors optimize the nonli near similarity KC (x, x′) =
2 exp(−eC (x, x′)) − 1, derived from eC . Note that KC is not required to be PSD nor
symmetric. GESL (Good Edit Similarity Learning) is express ed as follows:
min
C,B1,B2
1
n2
∑
zi,zj
ℓ(C,z i,z j) + β ∥C∥2
F
s.t. B1 ≥ − log( 1
2 ), 0 ≤ B2 ≤ − log( 1
2 ), B 1 − B2 =ηγ,
47. Source code available at: http://sunflower.kuicr.kyoto-u.ac.jp/~hiroto/project/optaa.html
48. Source code available at: http://www-bcf.usc.edu/~bellet/
40


*(Extracción truncada en la página 40 de 59 totales)*