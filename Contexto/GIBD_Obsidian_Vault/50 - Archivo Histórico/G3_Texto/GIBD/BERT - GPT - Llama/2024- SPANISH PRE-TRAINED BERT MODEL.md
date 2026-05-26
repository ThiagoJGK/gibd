---
aliases: [2024- SPANISH PRE-TRAINED BERT MODEL]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2024-09-12
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Papers/2024- SPANISH PRE-TRAINED BERT MODEL.pdf"
tamanio_bytes: 154494
---

# 2024- SPANISH PRE-TRAINED BERT MODEL

Ruta interna: `GIBD/BERT - GPT - Llama/Papers/2024- SPANISH PRE-TRAINED BERT MODEL.pdf`

---

arXiv:2308.02976v1  [cs.CL]  6 Aug 2023
Accepted as a workshop paper at PML4DC, ICLR 2020
SPANISH PRE -TRAINED BERT MODEL
AND EVALUATION DATA
Jos´e Ca ˜nete∗
, Gabriel Chaperon, Rodrigo Fuentes
Department of Computer Science, Universidad de Chile
{jcanete,gchapero,rfuentes}@dcc.uchile.cl
Jou-Hui Ho, Hojin Kang
Department of Electrical Engineering, Universidad de Chil e
{jouhui.ho,ho.kang.k}@ug.uchile.cl
Jorge P´erez
Department of Computer Science, Universidad de Chile &
Millennium Institute for Foundational Research on Data (IM FD), Chile
jperez@dcc.uchile.cl
ABSTRACT
The Spanish language is one of the top 5 spoken languages in th e world. Never-
theless, ﬁnding resources to train or evaluate Spanish lang uage models is not an
easy task. In this paper we help bridge this gap by presenting a BERT-based lan-
guage model pre-trained exclusively on Spanish data. As a se cond contribution,
we also compiled several tasks speciﬁcally for the Spanish l anguage in a single
repository much in the spirit of the GLUE benchmark. By ﬁne-t uning our pre-
trained Spanish model, we obtain better results compared to other BERT-based
models pre-trained on multilingual corpora for most of the t asks, even achieving
a new state-of-the-art on some of them. We have publicly rele ased our model, the
pre-training data, and the compilation of the Spanish bench marks.
1 I NTRODUCTION
The ﬁeld of natural language processing (NLP) has made incre dible progress in the last two years.
Two of the most decisive features that have driven this impro vement are the self-attention mecha-
nism, particularly the Transformer architecture (V aswani et al., 2017), and the introduction of un-
supervised pre-training methods (Peters et al., 2018; Howa rd & Ruder, 2018; Devlin et al., 2018),
which take advantage of huge amounts of unlabeled text corpo ra. Thus the leading strategy to-
day for achieving good performance is to ﬁrst train a Transfo rmer-based model, say M , with a
general language-modeling task over a huge unlabeled corpu s and then, after this ﬁrst training is
over, “ﬁne-tune” M by continuing to train it for a speciﬁc task using labeled dat a. Built upon
these ideas, the BERT architecture –which stands for “Bidir ectional Encoder Representations from
Transformers”– (Devlin et al., 2018), and several improvem ents thereof (Liu et al., 2019; Lan et al.,
2019; Y ang et al., 2019b; Clark et al., 2019), changed the lan dscape of NLP in 2019.
BERT was initially released in two versions, one pre-traine d on an English corpus and an-
other on a Chinese corpus (Devlin et al., 2018). As a way of pro viding a resource for other
languages besides English and Chinese, the authors also rel eased a “multilingual” version of
BERT (we’ll refer to it as mBERT from now on) pre-trained simu ltaneously on a corpus in-
cluding more than 100 different languages. The mBERT model h as shown impressive per-
formance when ﬁne-tuned for language-speciﬁc tasks and has achieved state-of-the-art results
in many cross-lingual benchmarks (Wu & Dredze, 2019; Pires e t al., 2019). The good perfor-
mance of mBERT has drawn the attention of many different NLP c ommunities, and efforts have
been made to produce BERT versions trained on monolingual da ta. This has led to the release
∗Work partially performed while at Adereso.
1
Accepted as a workshop paper at PML4DC, ICLR 2020
of BERT models in Russian (Kuratov & Arkhipov, 2019), French (Martin et al., 2019; Le et al.,
2019), Dutch (de Vries et al., 2019; Delobelle et al., 2020), Italian (Polignano et al., 2019), and Por-
tugese (Souza et al., 2019).
In this paper we present the ﬁrst BERT model pre-trained for t he Spanish language. Despite Span-
ish being widely spoken (much more than the previously menti oned languages), ﬁnding resources
to train or evaluate Spanish language models is not an easy ta sk. For this reason, we also com-
piled several Spanish-speciﬁc tasks in a single repository , much in the spirit of the GLUE bench-
mark (Wang et al., 2019). By ﬁne-tuning our pre-trained Span ish model, we obtain better results
compared to other BERT-based models that were pre-trained on multilingual corpora for most of the
tasks, and we even achieve a new state-of-the-art on some of t hem. We have released our pre-trained
model, the training corpus, and the compilation of benchmar ks as free resources to be used by the
Spanish NLP community. 1
In the rest of this paper, we ﬁrst present our Spanish-BERT mo del. Then, we describe the tasks
that we have compiled into a benchmark that we call GLUES (GLU E for Spanish), and ﬁnally, we
present the results obtained by our model in some of the GLUES tasks. Before going into the details
of our model and results, we will brieﬂy review the related wo rk.
2 R ELATED WORK
Pre-trained language models using deep neural networks bec ame very popular starting with ULM-
Fit (Howard & Ruder, 2018). ULMFit is based on a standard recu rrent neural network architecture
and a language-modeling task (predicting the next token fro m the previous sequence of tokens). By
using vast amounts of text, ULMFit is ﬁrst trained for the language-modeling task, aiming to help the
model acquire general knowledge from a big corpus. The model is then ﬁne-tuned in a supervised
way to solve a speciﬁc task using labeled data. The empirical results showed that the combination
of pre-training and ﬁne-tuning can considerably outperfor m training a model from scratch for the
same supervised task.
A similar pre-training strategy was later used by Devlin et a l. (2018) to propose the BERT
model. Compared with ULMFit, in BERT, the recurrent archite cture is replaced with self-
attention (V aswani et al., 2017), which allows the predicti on of a token to depend on every other
token in the same sequence. The task used for pre-training BE RT, calledmasked language modeling,
is based on corrupting an input sequence by arbitrarily dele ting some of the tokens and then training
the model to reconstruct the original sequence (Devlin et al ., 2018). Several variations of BERT in
terms of the training method and the task used for pre-traini ng have been proposed (Liu et al., 2019;
Joshi et al., 2019; Y ang et al., 2019b). There have also been e fforts to make models more efﬁcient in
terms of the number of parameters or training time (Sanh et al ., 2019; Lan et al., 2019; Clark et al.,
2019).
Wu & Dredze (2019) and Pires et al. (2019) studied Multilingu al BERT models, that is, models pre-
trained simultaneously on corpora from different language s. These works showed how a single
model can learn from several languages, setting strong base lines for tasks involving non-English
languages. XLM (Lample & Conneau, 2019) introduced a superv ised objective which involved par-
allel multilingual data, and XLM-RoBERTa (Conneau et al., 2 019) brought the multilingual models
to the big leagues in terms of model size.
Several single-language BERT models came with results that usually got better perfor-
mance than multilingual models as it is the case with CamemBE RT (Martin et al., 2019)
and FlauBERT (Le et al., 2019) for French, BERTje (de Vries et al., 2019) and Rob-
BERT (Delobelle et al., 2020) for Dutch, FinBERT (Virtanen e t al., 2019) for Finish, to name a few.
Our work is similar to these models, but for the Spanish langu age. To the best of our knowledge,
our paper presents the ﬁrst publicly available Spanish BERT model and evaluation.
1https://github.com/dccuchile/beto
2
Accepted as a workshop paper at PML4DC, ICLR 2020
3 S PANISH -BERT MODEL , DATA AND TRAINING
We trained a model similar in size to a BERT-Base model (Devli n et al., 2018). Our model has 12
self-attention layers, with 12 attention-heads each (V aswani et al., 2017), using a hidden size of 768.
In total, our model has 110M parameters. We trained two versi ons: one with cased data and one
with uncased data, using a dataset that we will describe next .
For training our model, we collected text from different sources. We used all the data from Wikipedia
and all the sources of the OPUS Project (Tiedemann, 2012) tha t had text in Spanish. These sources
include United Nations and Government journals, TED Talks, Subtitles, News Stories, and more.
The total size of the corpora gathered was comparable to the c orpora used in the original BERT. Our
training corpus contains about 3 billion words, and we have r eleased it for later use. 2. Our corpus
can be considered an updated version of the one compiled by Ca rdellino (2016).
For both versions of our model, cased and uncased, we constru cted a vocabulary of 31K subwords
using the byte pair encoding algorithm provided by the Sente ncePiece library (Kudo & Richardson,
2018). We added 1K placeholder tokens for later use, which ga ve us a vocabulary of 32K tokens.
For training our BERT models, we considered certain trainin g details that have been successful in
RoBERTa (Liu et al., 2019). In particular, we integrated the Dynamic Masking technique into our
training, which involves using different masks for the same sentence in our corpus. The Dynamic
Masking we used was 10x, meaning that every sentence had 10 di fferent masks. We also consid-
ered the Whole-Word Masking (WWM) technique from the update d version of BERT (Devlin et al.,
2018). WWM ensures that when masking a speciﬁc token, if the t oken corresponds to a sub-word in
a sentence, then all contiguous tokens conforming the same w ord are also masked. Additionally, we
trained on larger batches compared to the original BERT (but smaller than RoBERTa). We trained
each model (cased and uncased) for 2M steps, with learning ra te of 0.0001, and the ﬁrst 10000 steps
as warm-up. The training was also divided into two phases, as described by Y ou et al. (2019): we
trained the ﬁrst 900k steps with a batch size of 2048 and maxim um sequence length of 128, and the
rest of the training with batch size of 256 and maximum sequen ce length of 512. All the pre-training
was done using Google’s preemptible TPU v3-8.
4 GLUES
In this section, we present the GLUES benchmark, which is a co mpilation of common NLP tasks
in the Spanish language, following the idea of the original E nglish GLUE benchmark (Wang et al.,
2019). Through this benchmarks, we hope to help standardize future Spanish NLP efforts 3. Next,
we describe the tasks that we currently consider in GLUES.
Natural Language Inference: XNLI The Multi-Genre Natural Language Inference (MNLI)
dataset (Williams et al., 2017) consists of pair of sentence s. The ﬁrst sentence is called the premise,
and the second is the hypothesis. The task is to predict wheth er the premise entails the hypothesis
(entailment), contradicts it (contradiction), or neither entails nor contradicts it (neutral). In other
words, the task is a 3-class classiﬁcation. The Cross-Lingu al Natural Language Inference (XNLI)
corpus (Conneau et al., 2018) is an evaluation dataset that e xtends the MNLI by adding development
and test sets for 15 languages. In this setup, we train using t he Spanish machine translation of the
MNLI dataset, and use the development and test sets from the X NLI corpus. This task is evaluated
using simple accuracy.
Paraphrasing: PA WS-X P AWS-X (Y ang et al., 2019a) is the multilingual version of th e P AWS
dataset (Zhang et al., 2019). The task consists of determini ng whether two sentences are semanti-
cally equivalent or not. The dataset provides standard (tra nslated) train, development and test sets,
and we used the Spanish portion. It is evaluated using simple accuracy.
Named Entity Recognition: CoNLL Named Entity Recognition consists of determining phrases
in a sentence that contain the names of persons, organizatio ns, locations, times, and quantities.
2https://github.com/josecannete/spanish-corpora
3See https://github.com/dccuchile/glues for a detailed description on how to obtain train-
dev-test data for each task.
3
Accepted as a workshop paper at PML4DC, ICLR 2020
For this task, we use the dataset by Tjong Kim Sang (2002), whi ch focuses on persons, organi-
zations, and locations, with a fourth category of miscellan eous entities, all tagged using the BIO
format (Ramshaw & Marcus, 1999). This dataset provides stan dard train, development, and test
sets, and the performance is measured with an F1 score.
Part-of-Speech T agging: Universal Dependencies v1.4 A part of speech (POS) is a category of
words with similar grammatical properties, such as noun, ve rb, adjective, preposition, and con-
junction. For the POS tagging task, we use the Spanish subset of the Universal Dependencies
(v1.4) Treebank (Nivre et al., 2016). The version of the data set was chosen following the works
of Wu & Dredze (2019) and Kim et al. (2017). The dataset provid es standard train, development,
and test sets. This task is evaluated based on the F1 score of p redicted POS tags.
Document Classiﬁcation: MLDoc The MLDoc dataset (Schwenk & Li, 2018) is a balanced sub-
set of the Reuters corpus. This task consists of classifying the documents into four categories,
CCA T (Corporate/Industrial), ECA T (Economics), GCA T (Government/Social), and MCA T (Mar-
kets). This dataset provides multiple sizes for the train sp lit (1k, 2k, 5k and 10k), along with standard
development and test sets for eight languages. We chose to tr ain using the largest available train split
in Spanish. This task is evaluated using simple accuracy.
Dependency Parsing: Universal Dependencies v2.2 A dependency tree represents the grammat-
ical structure of a sentence, deﬁning relationships in the f orm of labeled-edges from “dependent” to
“head” words. The label of each edge represents the type of re lationship. The task of dependency
parsing consists of constructing a dependency tree for a giv en sentence. For this task, we use a
subset of the Universal Dependencies v2.2 Treebank (Nivre e t al., 2018), in particular, the concate-
nation of the AnCora and GSD Spanish portions of the dataset. This decision and the version we
chose follow the work of Ahmad et al. (2019). The task is evalu ated using the Unlabeled Attachment
Score (UAS) and Labeled Attachment Score (LAS) (K¨ ubler et al., 2009). UAS is the percentage of
words that have been assigned the correct head, while LAS is t he percentage of words that have been
assigned both the correct head and the correct label for the r elationship.
Question Answering: MLQA, TAR and XQuAD Given a context and a question, the task of
question answering (QA) consists of ﬁnding the sequence of c ontiguous words within the context
that answers the question. This task is usually evaluated us ing two metrics averaged across all
questions: exact match, that corresponds to the percentage of answers that match exactly, and F1
score, where answers are treated as bags of words. For our ben chmark, we consider three trans-
lated versions of the SQuAD v1.1 dataset (Rajpurkar et al., 2 016), namely, MLQA (Lewis et al.,
2019), XQuAD (Artetxe et al., 2019), and TAR (Carrino et al., 2019), which we will explain next.
MLQA provides translations for seven language with train da ta produced using machine transla-
tion from English, and development and test data translated by professionals (Lewis et al., 2019).
XQuAD provides test sets to evaluate cross-lingual models i n 11 languages (Artetxe et al., 2019).
Carrino et al. (2019) proposed the TAR method (Translate Ali gn Retrieve) that can be used to pro-
duce automatically machine translated versions of QA datas ets. They provided a TAR-translation
from English to Spanish of SQuAD v1.1 (Carrino et al., 2019). In our benchmark, we consider the
train sets in MLQA and TAR, and the test sets in MLQA and XQuAD.
5 E VALUATION
5.1 F INE -TUNING
Since one of the goals of our work was to compare the performan ce of Spanish-BERT to
mBERT (Wu & Dredze, 2019; Y ang et al., 2019a), our experiment al setup closely follows the one
by Wu & Dredze (2019). Task speciﬁc output layers are incorpo rated, following the original BERT
work (Devlin et al., 2018).
For each task, no preprocessing is performed except tokeniz ation from words into subwords
using the learned vocabulary and WordPiece (Wu et al., 2016) . We use the Adam opti-
mizer (Kingma & Ba, 2014) for ﬁne-tuning with standard param eters ( β1 = 0 .9, β2 = 0 .999),
4
Accepted as a workshop paper at PML4DC, ICLR 2020
Model XNLI PA WS-X NER POS MLDoc
Best mBERT 78.50 a 89.00b 87.38a 97.10a 95.70a
es-BERT uncased 80.15 89.55 82.67 98.44 96.12∗
es-BERT cased 82.01 89.05 88.43 98.97 ∗ 95.60
Table 1: Comparison of Spanish BERT (es-BERT) with the best r esults obtained by multilingual
BERT models where the ﬁne tune was done only on the Spanish tra in data in every dataset. Super-
script denotes results obtained by: ( a) Wu & Dredze (2019) and ( b) Y ang et al. (2019a). The “ ∗”
indicates a new state-of-the-art resut in the respective Sp anish benchmark.
Model MLQA, MLQA TAR, XQuAD TAR, MLQA
Best mBERT 53.90 / 37.40 c 77.60 / 61.80d 68.10 / 48.30d
es-BERT uncased 67.85 / 46.03 77.52 / 55.46 68.04 / 45.00
es-BERT cased 68.01 / 45.88 77.56 / 57.06 69.15 / 45.63
Table 2: Comparison of Spanish BERT (es-BERT) with the best r esults obtained by multilingual
BERT models in Question Answering. We only consider models w here the ﬁne tune was done on
the Spanish train data in every dataset. Results are present ed as F1 / ExactMatch. Every column
header denotes the train set (left) and test set (right) used in everry case. Different superscript denotes
results obtained by different authors: ( c) Lewis et al. (2019) and ( d) Carrino et al. (2019)
and L2 weight decay of 0.01. We warm up the learning rate for the ﬁrst 10% of steps, with a l inear
decay afterward.
To allow for ﬁne-tuning on a single GPU, we limit the length of each sentence to 128 tokens for
all tasks. To accommodate tasks that require word-level cla ssiﬁcation, we use the sliding window
approach described by Wu & Dredze (2019). After the ﬁrst wind ow, the last 64 tokens are kept for
the next window, and only 64 new tokens are fed into the model.
For hyperparameter selection, we run experiments using dif ferent combinations of batch size, learn-
ing rate and number of epochs, following the values recommen ded by Devlin et al. (2018): batch
size {16, 32}; learning rate {5e-5, 3e-5, 2e-5 }; number of epochs {2, 3, 4 }. An extensive hyperpa-
rameter search is left for future work.
5.2 R ESULTS
Tables 1 and 2 show our results compared to the best mBERT resu lt reported in the literature for the
same setting. Spanish BERT outperforms most of the mBERT res ults, except for some QA settings
(Table 2). One of the largest difference can be seen in the XNL I task, which is the task that has
the largest training dataset in Spanish. We note that for two of the most standard Spanish datasets
(POS and MLDoc) we obtained a new state of the art. We also note that there are some important
differences in the performances in the QA datasets. The low quality of machine translation in MLQA
might be one possible reason. We observed that nearly half of the 81K examples in MLQA have a
mismatch between the answer and its starting position in the context.
It is important to note that our models are Spanish-only and thus cannot take advantage neither of
the original English train set in every translated benchmar k, nor from train data in other languages.
Taking advantage of data in different languages is a capabil ity that multilingual models have by
design. In fact, there has been recent work on large multilin gual models that can achieve better
results on the Spanish datasets when ﬁne-tuned with general , not necessarily Spanish, data. This is
the case with the XLM-RoBERTa model (Conneau et al., 2019), w hich, with 560M parameters and
consuming training data from different languages, obtaine d results of 85.6% for XNLI and 89% for
NER in the Spanish test set. Both results are better than the o ne that we obtained by ﬁne-tuning with
Spanish data only (Table 1). Similarly, Y ang et al. (2019a) obtained 90.7% in the Spanish test-set of
P AWS-X when ﬁne-tuning mBERT including the original English train set.
5
Accepted as a workshop paper at PML4DC, ICLR 2020
6 C ONCLUSION
The advent of Transformer-based pre-trained language models has greatly improved the accessibility
of high-performing models for the average user. In this pape r, we successfully pre-train a Spanish-
only model and open-source it, along with the training corpu s and evaluation benchmarks, for the
community to use.
The ease of use of a pre-trained NLP model implies that its use cases are much broader, given that
practitioners from disciplines other than computer scienc e could ﬁne-tune them for their domain-
speciﬁc downstream tasks. By releasing our Spanish-BERT mo del, we hope to encourage research
and the application of deep learning techniques in Spanish- speaking countries.
Another direction of the work is to improve Spanish NLP model s in terms of size, memory, and
computation requirements. We are currently working on pre- training different ALBERT models
(Lan et al., 2019) for Spanish, with number of parameters ran ging from 5M to 223M. Our initial
results with the smaller models are encouraging, and we plan to release these models as well.
ACKNOWLEDGMENTS
We thank Adereso 4 for kindly providing support for training our uncased model , and Google for
helping us with the Cloud TPU program for research. This work was partially funded by the Mil-
lennium Institute for Foundational Research on Data.
REFERENCES
Wasi Ahmad, Zhisong Zhang, Xuezhe Ma, Eduard Hovy, Kai-Wei Chang, and Nanyun Peng. On dif-
ﬁculties of cross-lingual transfer with order differences : A case study on dependency parsing. In
Proceedings of the 2019 Conference of the North American Cha pter of the Association for Com-
putational Linguistics: Human Language T echnologies, V olume 1 (Long and Short Papers) , pp.
2440–2452, Minneapolis, Minnesota, June 2019. Associatio n for Computational Linguistics. doi:
10.18653/v1/N19-1253. URL https://www.aclweb.org/anthology/N19-1253.
Mikel Artetxe, Sebastian Ruder, and Dani Y ogatama. On the cr oss-lingual transferability of mono-
lingual representations. arXiv preprint arXiv:1910.11856 , 2019.
Cristian Cardellino. Spanish Billion Words Corpus and Embe ddings, March 2016. URL
https://crscardellino.github.io/SBWCE/.
Casimiro Pio Carrino, Marta R Costa-juss` a, and Jos´ e AR Fonollosa. Automatic spanish translation
of the squad dataset for multilingual question answering. arXiv preprint arXiv:1912.05200, 2019.
Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D M anning. Electra: Pre-training
text encoders as discriminators rather than generators. In International Conference on Learning
Representations, 2019.
Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Willi ams, Samuel R Bowman, Holger
Schwenk, and V eselin Stoyanov. Xnli: Evaluating cross-lin gual sentence representations. arXiv
preprint arXiv:1809.05053, 2018.
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek,
Francisco Guzm´ an, Edouard Grave, Myle Ott, Luke Zettlemoy er, and V eselin Stoyanov. Un-
supervised cross-lingual representation learning at scal e. arXiv preprint arXiv:1911.02116, 2019.
Wietse de Vries, Andreas van Cranenburgh, Arianna Bisazza, Tommaso Caselli, Gertjan van Noord,
and Malvina Nissim. Bertje: A dutch bert model. arXiv preprint arXiv:1912.09582 , 2019.
Pieter Delobelle, Thomas Winters, and Bettina Berendt. Rob bert: a dutch roberta-based language
model. arXiv preprint arXiv:2001.06286 , 2020.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Tout anova. Bert: Pre-training of deep
bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805 , 2018.
4https://www.adere.so/
6
Accepted as a workshop paper at PML4DC, ICLR 2020
Jeremy Howard and Sebastian Ruder. Universal language mode l ﬁne-tuning for text classiﬁcation.
arXiv preprint arXiv:1801.06146 , 2018.
Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S Weld, Luke Zet tlemoyer, and Omer Levy.
Spanbert: Improving pre-training by representing and pred icting spans. arXiv preprint
arXiv:1907.10529, 2019.
Joo-Kyung Kim, Y oung-Bum Kim, Ruhi Sarikaya, and Eric Fosle r-Lussier. Cross-lingual transfer
learning for POS tagging without cross-lingual resources. In Proceedings of the 2017 Conference
on Empirical Methods in Natural Language Processing , pp. 2832–2838, Copenhagen, Denmark,
September 2017. Association for Computational Linguistic s. doi: 10.18653/v1/D17-1302. URL
https://www.aclweb.org/anthology/D17-1302.
Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint
arXiv:1412.6980, 2014.
Sandra K¨ ubler, Ryan McDonald, and Joakim Nivre. Dependenc y parsing. Synthesis lectures on
human language technologies, 1(1):1–127, 2009.
Taku Kudo and John Richardson. Sentencepiece: A simple and l anguage independent subword
tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226 , 2018.
Y uri Kuratov and Mikhail Arkhipov. Adaptation of deep bidirectional multilingual transformers for
russian language. arXiv preprint arXiv:1905.07213 , 2019.
Guillaume Lample and Alexis Conneau. Cross-lingual langua ge model pretraining. arXiv preprint
arXiv:1901.07291, 2019.
Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpe l, Piyush Sharma, and Radu Sori-
cut. Albert: A lite bert for self-supervised learning of lan guage representations. arXiv preprint
arXiv:1909.11942, 2019.
Hang Le, Lo¨ ıc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexan-
dre Allauzen, Benoˆ ıt Crabb´ e, Laurent Besacier, and Didie r Schwab. Flaubert: Unsupervised
language model pre-training for french. arXiv preprint arXiv:1912.05372 , 2019.
Patrick Lewis, Barlas O˘ guz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. Mlqa: Evaluating
cross-lingual extractive question answering. arXiv preprint arXiv:1910.07475 , 2019.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and V eselin Stoyanov. Roberta: A r obustly optimized bert pretraining
approach. arXiv preprint arXiv:1907.11692 , 2019.
Louis Martin, Benjamin Muller, Pedro Javier Ortiz Su´ arez, Y oann Dupont, Laurent Romary,
´Eric Villemonte de la Clergerie, Djam´ e Seddah, and Benoˆ ıt Sagot. Camembert: a tasty french
language model. arXiv preprint arXiv:1911.03894 , 2019.
Joakim Nivre, Marie-Catherine De Marneffe, Filip Ginter, Yoav Goldberg, Jan Hajic, Christopher D
Manning, Ryan McDonald, Slav Petrov, Sampo Pyysalo, Natali a Silveira, et al. Universal de-
pendencies v1: A multilingual treebank collection. In Proceedings of the T enth International
Conference on Language Resources and Evaluation (LREC’16) , pp. 1659–1666, 2016.
Joakim Nivre, Mitchell Abrams, ˇZeljko Agi´ c, Lars Ahrenberg, Lene Antonsen, Maria Jesus Ar an-
zabe, Gashaw Arutie, Masayuki Asahara, Luma Ateyah, Mohamm ed Attia, et al. Universal de-
pendencies 2.2. 2018.
Matthew E. Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and
Luke Zettlemoyer. Deep contextualized word representatio ns. preprint arXiv 1802.05365 , 2018.
Telmo Pires, Eva Schlinger, and Dan Garrette. How multiling ual is multilingual bert? CoRR,
abs/1906.01502, 2019. URL http://arxiv.org/abs/1906.01502.
7
Accepted as a workshop paper at PML4DC, ICLR 2020
Marco Polignano, Pierpaolo Basile, Marco de Gemmis, Giovan ni Semeraro, and V a-
lerio Basile. AlBERTo: Italian BERT Language Understandin g Model for NLP
Challenging Tasks Based on Tweets. In Proceedings of the Sixth Italian Confer-
ence on Computational Linguistics (CLiC-it 2019) , volume 2481. CEUR, 2019. URL
https://www.scopus.com/inward/record.uri?eid=2-s2.0 -85074851349&partnerID=40&md5=7abed946e06f76b3825ae5e294ffac14.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Perc y Liang. Squad: 100,000+ questions
for machine comprehension of text. arXiv preprint arXiv:1606.05250 , 2016.
Lance A Ramshaw and Mitchell P Marcus. Text chunking using tr ansformation-based learning. In
Natural language processing using very large corpora , pp. 157–176. Springer, 1999.
Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wo lf. Distilbert, a distilled version of
bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 , 2019.
Holger Schwenk and Xian Li. A corpus for multilingual docume nt classiﬁcation in eight languages.
In Nicoletta Calzolari (Conference chair), Khalid Choukri , Christopher Cieri, Thierry Declerck,
Sara Goggi, Koiti Hasida, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, H´ el` ene Mazo, Asun-
cion Moreno, Jan Odijk, Stelios Piperidis, and Takenobu Tok unaga (eds.), Proceedings of the
Eleventh International Conference on Language Resources a nd Evaluation (LREC 2018) , Paris,
France, may 2018. European Language Resources Association (ELRA). ISBN 979-10-95546-00-
9.
F´ abio Souza, Rodrigo Nogueira, and Roberto Lotufo. Portug uese named entity recognition using
bert-crf. arXiv preprint arXiv:1909.10649 , 2019.
J¨ org Tiedemann. Parallel data, tools and interfaces in opu s. In Lrec, volume 2012, pp. 2214–2218,
2012.
Erik F. Tjong Kim Sang. Introduction to the CoNLL-2002 share d task: Language-independent
named entity recognition. In COLING-02: The 6th Conference on Natural Language Learning
2002 (CoNLL-2002), 2002. URL https://www.aclweb.org/anthology/W02-2024.
Ashish V aswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit , Llion Jones, Aidan N Gomez,
Łukasz Kaiser, and Illia Polosukhin. Attention is all you ne ed. In Advances in neural information
processing systems, pp. 5998–6008, 2017.
Antti Virtanen, Jenna Kanerva, Rami Ilo, Jouni Luoma, Juhan i Luotolahti, Tapio Salakoski, Filip
Ginter, and Sampo Pyysalo. Multilingual is not enough: Bert for ﬁnnish. arXiv preprint
arXiv:1912.07076, 2019.
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Ome r Levy, and Samuel R. Bowman.
GLUE: A multi-task benchmark and analysis platform for natu ral language understanding. 2019.
In the Proceedings of ICLR.
Adina Williams, Nikita Nangia, and Samuel R Bowman. A broad- coverage challenge corpus for
sentence understanding through inference. arXiv preprint arXiv:1704.05426 , 2017.
Shijie Wu and Mark Dredze. Beto, bentz, becas: The surprisin g cross-lingual effectiveness of bert.
arXiv preprint arXiv:1904.09077 , 2019.
Y onghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey,
Maxim Krikun, Y uan Cao, Qin Gao, Klaus Macherey, et al. Googl e’s neural machine trans-
lation system: Bridging the gap between human and machine tr anslation. arXiv preprint
arXiv:1609.08144, 2016.
Yinfei Y ang, Y uan Zhang, Chris Tar, and Jason Baldridge. Paw s-x: A cross-lingual adversarial
dataset for paraphrase identiﬁcation. arXiv preprint arXiv:1908.11828 , 2019a.
Zhilin Y ang, Zihang Dai, Yiming Y ang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le.
Xlnet: Generalized autoregressive pretraining for langua ge understanding. In Advances in neural
information processing systems, pp. 5754–5764, 2019b.
8
Accepted as a workshop paper at PML4DC, ICLR 2020
Y ang Y ou, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kuma r, Srinadh Bhojanapalli, Xiaodan
Song, James Demmel, Kurt Keutzer, and Cho-Jui Hsieh. Large b atch optimization for deep
learning: Training bert in 76 minutes. In International Conference on Learning Representations ,
2019.
Y uan Zhang, Jason Baldridge, and Luheng He. Paws: Paraphrase adversaries from word scrambling.
arXiv preprint arXiv:1904.01130 , 2019.
9