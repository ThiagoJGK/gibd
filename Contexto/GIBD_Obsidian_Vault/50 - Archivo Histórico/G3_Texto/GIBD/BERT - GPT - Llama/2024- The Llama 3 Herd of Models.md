---
aliases: [2024- The Llama 3 Herd of Models]
tags:
  - grupo/g3_texto
  - estado/archivo
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2024-09-12
origen_zip: GIBD-20260521T205218Z-3-004.zip
ruta_interna: "GIBD/BERT - GPT - Llama/Papers/Referencias/2024- The Llama 3 Herd of Models.pdf"
tamanio_bytes: 9268677
---

# 2024- The Llama 3 Herd of Models

Ruta interna: `GIBD/BERT - GPT - Llama/Papers/Referencias/2024- The Llama 3 Herd of Models.pdf`

---

The Llama 3 Herd of Models
Llama Team, AI @ Meta1
1A detailed contributor list can be found in the appendix of this paper.
Modern artiﬁcial intelligence (AI) systems are powered by foundation models. This paper presents a
new set of foundation models, called Llama 3. It is a herd of language models that natively support
multilinguality, coding, reasoning, and tool usage. Our largest model is a dense Transformer with
405B parameters and a context window of up to 128K tokens. This paper presents an extensive
empirical evaluation of Llama 3. We ﬁnd that Llama 3 delivers comparable quality to leading language
models such as GPT-4 on a plethora of tasks. We publicly release Llama 3, including pre-trained and
post-trained versions of the 405B parameter language model and our Llama Guard 3 model for input
and output safety. The paper also presents the results of experiments in which we integrate image,
video, and speech capabilities into Llama 3 via a compositional approach. We observe this approach
performs competitively with the state-of-the-art on image, video, and speech recognition tasks. The
resulting models are not yet being broadly released as they are still under development.
Date: July 23, 2024
Website: https://llama.meta.com/
1 Introduction
Foundation models are general models of language, vision, speech, and/or other modalities that are designed
to support a large variety of AI tasks. They form the basis of many modern AI systems.
The development of modern foundation models consists of two main stages:(1) a pre-training stage in which
the model is trained at massive scale using straightforward tasks such as next-word prediction or captioning
and (2) a post-training stage in which the model is tuned to follow instructions, align with human preferences,
and improve speciﬁc capabilities (for example, coding and reasoning).
In this paper, we present a new set of foundation models for language, calledLlama 3. The Llama 3 Herd
of models natively supports multilinguality, coding, reasoning, and tool usage. Our largest model is dense
Transformer with 405B parameters, processing information in a context window of up to 128K tokens. Each
member of the herd is listed in Table 1. All the results presented in this paper are for the Llama 3.1 models,
which we will refer to as Llama 3 throughout for brevity.
We believe there are three key levers in the development of high-quality foundation models: data, scale, and
managing complexity. We seek to optimize for these three levers in our development process:
• Data. Compared to prior versions of Llama (Touvron et al., 2023a,b), we improved both the quantity and
qualityofthedataweuseforpre-trainingandpost-training. Theseimprovementsincludethedevelopment
of more careful pre-processing and curation pipelines for pre-training data and the development of more
rigorous quality assurance and ﬁltering approaches for post-training data. We pre-train Llama 3 on a
corpus of about 15T multilingual tokens, compared to 1.8T tokens for Llama 2.
• Scale. We train a model at far larger scale than previous Llama models: our ﬂagship language model was
pre-trained using 3.8× 1025 FLOPs, almost 50× more than the largest version of Llama 2. Speciﬁcally,
we pre-trained a ﬂagship model with 405B trainable parameters on 15.6T text tokens. As expected per
1
Finetuned Multilingual Long context Tool use Release
Llama 3 8B   1   April 2024
Llama 3 8B Instruct     April 2024
Llama 3 70B   1   April 2024
Llama 3 70B Instruct     April 2024
Llama 3.1 8B     July 2024
Llama 3.1 8B Instruct     July 2024
Llama 3.1 70B     July 2024
Llama 3.1 70B Instruct     July 2024
Llama 3.1 405B     July 2024
Llama 3.1 405B Instruct     July 2024
Table 1 Overview of the Llama 3 Herd of models. All results in this paper are for the Llama 3.1 models.
scaling laws for foundation models, our ﬂagship model outperforms smaller models trained using the
same procedure. While our scaling laws suggest our ﬂagship model is an approximately compute-optimal
size for our training budget, we also train our smaller models for much longer than is compute-optimal.
The resulting models perform better than compute-optimal models at the same inference budget. We
use the ﬂagship model to further improve the quality of those smaller models during post-training.
• Managing complexity. We make design choices that seek to maximize our ability to scale the model
development process. For example, we opt for a standard dense Transformer model architecture (Vaswani
et al., 2017) with minor adaptations, rather than for a mixture-of-experts model (Shazeer et al., 2017)
to maximize training stability. Similarly, we adopt a relatively simple post-training procedure based
on supervised ﬁnetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO;
Rafailov et al. (2023)) as opposed to more complex reinforcement learning algorithms (Ouyang et al.,
2022; Schulman et al., 2017) that tend to be less stable and harder to scale.
The result of our work is Llama 3: a herd of three multilingual1 language models with 8B, 70B, and 405B
parameters. We evaluate the performance of Llama 3 on a plethora of benchmark datasets that span a wide
range of language understanding tasks. In addition, we perform extensive human evaluations that compare
Llama 3 with competing models. An overview of the performance of the ﬂagship Llama 3 model on key
benchmarks is presented in Table 2. Our experimental evaluation suggests that our ﬂagship model performs
on par with leading language models such as GPT-4 (OpenAI, 2023a) across a variety of tasks, and is close to
matching the state-of-the-art. Our smaller models are best-in-class, outperforming alternative models with
similar numbers of parameters (Bai et al., 2023; Jiang et al., 2023). Llama 3 also delivers a much better
balance between helpfulness and harmlessness than its predecessor (Touvron et al., 2023b). We present a
detailed analysis of the safety of Llama 3 in Section 5.4.
We are publicly releasing all three Llama 3 models under an updated version of the Llama 3 Community License;
see https://llama.meta.com. This includes pre-trained and post-trained versions of our 405B parameter
language model and a new version of our Llama Guard model (Inan et al., 2023) for input and output safety.
We hope that the open release of a ﬂagship model will spur a wave of innovation in the research community,
and accelerate a responsible path towards the development of artiﬁcial general intelligence (AGI).
As part of the Llama 3 development process we also develop multimodal extensions to the models, enabling
image recognition, video recognition, and speech understanding capabilities. These models are still under
active development and not yet ready for release. In addition to our language modeling results, the paper
presents results of our initial experiments with those multimodal models.
1The Llama 3 8B and 70B were pre-trained on multilingual data but were intended for use in English at the time.
2
Category Benchmark
Llama 3 8B
Gemma 2 9B
Mistral 7B
Llama 3 70B
Mixtral 8x22B
GPT 3.5 Turbo
Llama 3 405B
Nemotron 4 340B
GPT-4(0125)
GPT-4o
Claude 3.5 Sonnet
General
MMLU(5-shot) 69.4 72.3 61.1 83.6 76.9 70.7 87.3 82.6 85.1 89.1 89.9
MMLU(0-shot, CoT) 73.0 72.3△ 60.5 86.0 79.9 69.8 88.6 78.7◁ 85.4 88.7 88.3
MMLU-Pro(5-shot, CoT) 48.3 – 36.9 66.4 56.3 49.2 73.3 62.7 64.8 74.0 77.0
IFEval 80.4 73.6 57.6 87.5 72.7 69.9 88.6 85.1 84.3 85.6 88.0
Code HumanEval(0-shot) 72.6 54.3 40.2 80.5 75.6 68.0 89.0 73.2 86.6 90.2 92.0
MBPP EvalPlus(0-shot) 72.8 71.7 49.5 86.0 78.6 82.0 88.6 72.8 83.6 87.8 90.5
Math GSM8K(8-shot, CoT) 84.5 76.7 53.2 95.1 88.2 81.6 96.8 92.3♦ 94.2 96.1 96.4♦
MATH(0-shot, CoT) 51.9 44.3 13.0 68.0 54.1 43.1 73.8 41.1 64.5 76.6 71.1
Reasoning ARC Challenge(0-shot) 83.4 87.6 74.2 94.8 88.7 83.7 96.9 94.6 96.4 96.7 96.7
GPQA(0-shot, CoT) 32.8 – 28.8 46.7 33.3 30.8 51.1 – 41.4 53.6 59.4
Tool use BFCL 76.1 – 60.4 84.8 – 85.9 88.5 86.5 88.3 80.5 90.2
Nexus 38.5 30.0 24.7 56.7 48.5 37.2 58.7 – 50.3 56.1 45.7
Long context
ZeroSCROLLS/QuALITY81.0 – – 90.5 – – 95.2 – 95.2 90.5 90.5
InﬁniteBench/En.MC 65.1 – – 78.2 – – 83.4 – 72.1 82.5 –
NIH/Multi-needle 98.8 – – 97.5 – – 98.1 – 100.0 100.0 90.8
Multilingual MGSM(0-shot, CoT) 68.9 53.2 29.9 86.9 71.1 51.4 91.6 – 85.9 90.5 91.6
Table 2 Performance of finetuned Llama 3 models on key benchmark evaluations. The table compares the performance of
the 8B, 70B, and 405B versions of Llama 3 with that of competing models. Weboldface the best-performing model in
each of three model-size equivalence classes.△Results obtained using 5-shot prompting (no CoT).◁Results obtained
without CoT. ♦Results obtained using zero-shot prompting.
2 General Overview
The model architecture of Llama 3 is illustrated in Figure 1. The development of our Llama 3 language
models comprises two main stages:
• Language model pre-training. We start by converting a large, multilingual text corpus to discrete tokens
and pre-training a large language model (LLM) on the resulting data to perform next-token prediction.
In the language model pre-training stage, the model learns the structure of language and obtains large
amounts of knowledge about the world from the text it is “reading”. To do this eﬀectively, pre-training
is performed at massive scale: we pre-train a model with 405B parameters on 15.6T tokens using a
context window of 8K tokens. This standard pre-training stage is followed by a continued pre-training
stage that increases the supported context window to 128K tokens. See Section 3 for details.
• Language model post-training. The pre-trained language model has a rich understanding of language
but it does not yet follow instructions or behave in the way we would expect an assistant to. We
align the model with human feedback in several rounds, each of which involves supervised ﬁnetuning
(SFT) on instruction tuning data and Direct Preference Optimization (DPO; Rafailov et al., 2024).
At this post-training2 stage, we also integrate new capabilities, such as tool-use, and observe strong
improvements in other areas, such as coding and reasoning. See Section 4 for details. Finally, safety
mitigations are also incorporated into the model at the post-training stage, the details of which are
described in Section 5.4.
The resulting models have a rich set of capabilities. They can answer questions in at least eight languages,
write high-quality code, solve complex reasoning problems, and use tools out-of-the-box or in a zero-shot way.
We also perform experiments in which we add image, video, and speech capabilities to Llama 3 using a
compositional approach. The approach we study comprises the three additional stages illustrated in Figure 28:
• Multi-modal encoder pre-training. We train separate encoders for images and speech. We train our
image encoder on large amounts of image-text pairs. This teaches the model the relation between visual
content and the description of that content in natural language. Our speech encoder is trained using a
2In this paper, we use the term “post-training” to refer to any model training that happens outside of pre-training.
3
Figure 1 Illustration of the overall architecture and training of Llama 3. Llama 3 is a Transformer language model trained to
predict the next token of a textual sequence. See text for details.
self-supervised approach that masks out parts of the speech inputs and tries to reconstruct the masked
out parts via a discrete-token representation. As a result, the model learns the structure of speech
signals. See Section 7 for details on the image encoder and Section 8 for details on the speech encoder.
• Vision adapter training. We train an adapter that integrates the pre-trained image encoder into the
pre-trained language model. The adapter consists of a series of cross-attention layers that feed image-
encoder representations into the language model. The adapter is trained on text-image pairs. This
aligns the image representations with the language representations. During adapter training, we also
update the parameters of the image encoder but we intentionally do not update the language-model
parameters. We also train a video adapter on top of the image adapter on paired video-text data. This
enables the model to aggregate information across frames. See Section 7 for details.
• Speech adapter training. Finally, we integrate the speech encoder into the model via an adapter that
converts speech encodings into token representations that can be fed directly into the ﬁnetuned language
model. The parameters of the adapter and encoder are jointly updated in a supervised ﬁnetuning stage
to enable high-quality speech understanding. We do not change the language model during speech
adapter training. We also integrate a text-to-speech system. See Section 8 for details.
Our multimodal experiments lead to models that can recognize the content of images and videos, and support
interaction via a speech interface. These models are still under development and not yet ready for release.
3 Pre-Training
Language model pre-training involves:(1) the curation and ﬁltering of a large-scale training corpus,(2) the
development of a model architecture and corresponding scaling laws for determining model size,(3) the
development of techniques for eﬃcient pre-training at large scale, and(4) the development of a pre-training
recipe. We present each of these components separately below.
3.1 Pre-Training Data
We create our dataset for language model pre-training from a variety of data sources containing knowledge
until the end of 2023. We apply several de-duplication methods and data cleaning mechanisms on each data
source to obtain high-quality tokens. We remove domains that contain large amounts of personally identiﬁable
information (PII), and domains with known adult content.
3.1.1 Web Data Curation
Much of the data we utilize is obtained from the web and we describe our cleaning process below.
PII and safety filtering. Among other mitigations, we implement ﬁlters designed to remove data from websites
are likely to contain unsafe content or high volumes of PII, domains that have been ranked as harmful
according to a variety of Meta safety standards, and domains that are known to contain adult content.
4
Text extraction and cleaning. We process the raw HTML content for non-truncated web documents to extract
high-quality diverse text. To do so, we build a custom parser that extracts the HTML content and optimizes
for precision in boilerplate removal and content recall. We evaluate our parser’s quality in human evaluations,
comparing it with popular third-party HTML parsers that optimize for article-like content, and found it
to perform favorably. We carefully process HTML pages with mathematics and code content to preserve
the structure of that content. We maintain the imagealt attribute text since mathematical content is often
represented as pre-rendered images where the math is also provided in thealt attribute. We experimentally
evaluate diﬀerent cleaning conﬁgurations. We ﬁnd markdown is harmful to the performance of a model that
is primarily trained on web data compared to plain text, so we remove all markdown markers.
De-duplication. We apply several rounds of de-duplication at the URL, document, and line level:
• URL-level de-duplication. We perform URL-level de-duplication across the entire dataset. We keep the
most recent version for pages corresponding to each URL.
• Document-level de-duplication. We perform global MinHash (Broder, 1997) de-duplication across the
entire dataset to remove near duplicate documents.
• Line-level de-duplication. We perform aggressive line-level de-duplication similar toccNet (Wenzek
et al., 2019). We remove lines that appeared more than 6 times in each bucket of 30M documents.
Although our manual qualitative analysis showed that the line-level de-duplication removes not only
leftover boilerplate from various websites such as navigation menus, cookie warnings, but also frequent
high-quality text, our empirical evaluations showed strong improvements.
Heuristic filtering. We develop heuristics to remove additional low-quality documents, outliers, and documents
with excessive repetitions. Some examples of heuristics include:
• We use duplicated n-gram coverage ratio (Rae et al., 2021) to remove lines that consist of repeated
content such as logging or error messages. Those lines could be very long and unique, hence cannot be
ﬁltered by line-dedup.
• We use “dirty word” counting (Raﬀel et al., 2020) to ﬁlter out adult websites that are not covered by
domain block lists.
• We use a token-distribution Kullback-Leibler divergence to ﬁlter out documents containing excessive
numbers of outlier tokens compared to the training corpus distribution.
Model-based quality filtering. Further, we experiment with applying various model-based quality classiﬁers
to sub-select high-quality tokens. These include using fast classiﬁers such asfasttext (Joulin et al., 2017)
trained to recognize if a given text would be referenced by Wikipedia (Touvron et al., 2023a), as well as more
compute-intensive Roberta-based classiﬁers (Liu et al., 2019a) trained on Llama 2 predictions. To train a
quality classiﬁer based on Llama 2, we create a training set of cleaned web documents, describe the quality
requirements, and instruct Llama 2’s chat model to determine if the documents meets these requirements. We
use DistilRoberta (Sanh et al., 2019) to generate quality scores for each document for eﬃciency reasons. We
experimentally evaluate the eﬃcacy of various quality ﬁltering conﬁgurations.
Code and reasoning data. Similar to DeepSeek-AI et al. (2024), we build domain-speciﬁc pipelines that extract
code and math-relevant web pages. Speciﬁcally, both the code and reasoning classiﬁers are DistilRoberta
models trained on web data annotated by Llama 2. Unlike the general quality classiﬁer mentioned above, we
conduct prompt tuning to target web pages containing math deduction, reasoning in STEM areas and code
interleaved with natural language. Since the token distribution of code and math is substantially diﬀerent
than that of natural language, these pipelines implement domain-speciﬁc HTML extraction, customized text
features and heuristics for ﬁltering.
Multilingual data. Similar to our processing pipelines for English described above, we implement ﬁlters to
remove data from websites that are likely to contain PII or unsafe content. Our multilingual text processing
pipeline has several unique features:
• We use afasttext-based language identiﬁcation model to categorize documents into 176 languages.
• We perform document-level and line-level de-duplication within data for each language.
5
• We apply language-speciﬁc heuristics and model-based ﬁlters to remove low-quality documents.
In addition, we perform quality ranking of multilingual documents using a multilingual Llama 2-based classiﬁer
to ensure that high-quality content is prioritized. We determine the amount of multilingual tokens used in
pre-training experimentally, balancing model performance on English and multilingual benchmarks.
3.1.2 Determining the Data Mix
To obtain a high-quality language model, it is essential to carefully determine the proportion of diﬀerent data
sources in the pre-training data mix. Our main tools in determining this data mix are knowledge classiﬁcation
and scaling law experiments.
Knowledge classification. We develop a classiﬁer to categorize the types of information contained in our web
data to more eﬀectively determine a data mix. We use this classiﬁer to downsample data categories that are
over-represented on the web, for example, arts and entertainment.
Scaling laws for data mix. To determine the best data mix, we perform scaling law experiments in which we
train several small models on a data mix and use that to predict the performance of a large model on that mix
(see Section 3.2.1). We repeat this process multiple times for diﬀerent data mixes to select a new data mix
candidate. Subsequently, we train a larger model on this candidate data mix and evaluate the performance of
that model on several key benchmarks.
Data mix summary. Our ﬁnal data mix contains roughly 50% of tokens corresponding to general knowledge,
25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.
3.1.3 Annealing Data
Empirically, we ﬁnd that annealing (see Section 3.4.3) on small amounts of high-quality code and mathematical
data can boost the performance of pre-trained models on key benchmarks. Akin to Li et al. (2024b), we
perform annealing with a data mix that upsamples high-quality data in select domains. We do not include
any training sets from commonly used benchmarks in our annealing data. This enables us to assess the true
few-shot learning capabilities and out-of-domain generalization of Llama 3.
Following OpenAI (2023a), we evaluate the eﬃcacy of annealing on the GSM8k (Cobbe et al., 2021) and
MATH (Hendrycks et al., 2021b) training sets in annealing. We ﬁnd that annealing improved the performance
of a pre-trained Llama 3 8B model on the GSM8k and MATH validation sets by 24.0% and 6.4%, respectively.
However, the improvements on the 405B model are negligible, suggesting that our ﬂagship model has strong
in-context learning and reasoning capabilities and does not require speciﬁc in-domain training samples to
obtain strong performance.
Using annealing to assess data quality. Similar to Blakeney et al. (2024), we ﬁnd that annealing enables us to
judge the value of small domain-speciﬁc datasets. We measure the value of such datasets by annealing the
learning rate of a 50% trained Llama 3 8B model linearly to 0 on 40B tokens. In those experiments, we assign
30% weight to the new dataset and the remaining 70% weight to the default data mix. Using annealing to
evaluate new data sources is more eﬃcient than performing scaling law experiments for every small dataset.
3.2 Model Architecture
Llama 3 uses a standard, dense Transformer architecture (Vaswani et al., 2017). It does not deviate signiﬁcantly
from Llama and Llama 2 (Touvron et al., 2023a,b) in terms of model architecture; our performance gains are
primarily driven by improvements in data quality and diversity as well as by increased training scale.
We make a few small modiﬁcations compared to Llama 2:
• We use grouped query attention (GQA; Ainslie et al. (2023)) with 8 key-value heads to improve inference
speed and to reduce the size of key-value caches during decoding.
• We use an attention mask that prevents self-attention between diﬀerent documents within the same
sequence. We ﬁnd that this change had limited impact during in standard pre-training, but ﬁnd it to be
important in continued pre-training on very long sequences.
6
8B 70B 405B
Layers 32 80 126
Model Dimension 4,096 8192 16,384
FFN Dimension 14,336 28,672 53,248
Attention Heads 32 64 128
Key/Value Heads 8 8 8
Peak Learning Rate 3× 10−4 1.5× 10−4 8× 10−5
Activation Function SwiGLU
Vocabulary Size 128,000
Positional Embeddings RoPE (θ = 500, 000)
Table 3 Overview of the key hyperparameters of Llama 3. We display settings for 8B, 70B, and 405B language models.
• We use a vocabulary with 128K tokens. Our token vocabulary combines 100K tokens from thetiktoken3
tokenizer with 28K additional tokens to better support non-English languages. Compared to the Llama
2 tokenizer, our new tokenizer improves compression rates on a sample of English data from 3.17 to
3.94 characters per token. This enables the model to “read” more text for the same amount of training
compute. We also found that adding 28K tokens from select non-English languages improved both
compression ratios and downstream performance, with no impact on English tokenization.
• We increase the RoPE base frequency hyperparameter to 500,000. This enables us to better support
longer contexts; Xiong et al. (2023) showed this value to be eﬀective for context lengths up to 32,768.
Llama 3 405B uses an architecture with 126 layers, a token representation dimension of 16,384, and 128
attention heads; see Table 3 for details. This leads to a model size that is approximately compute-optimal
according to scaling laws on our data for our training budget of3.8× 1025 FLOPs.
3.2.1 Scaling Laws
We develop scaling laws (Hoﬀmann et al., 2022; Kaplan et al., 2020) to determine the optimal model size for
our ﬂagship model given our pre-training compute budget. In addition to determining the optimal model size,
a major challenge is to forecast the ﬂagship model’s performance on downstream benchmark tasks, due to a
couple of issues: (1) Existing scaling laws typically predict only next-token prediction loss rather than speciﬁc
benchmark performance. (2) Scaling laws can be noisy and unreliable because they are developed based on
pre-training runs conducted with small compute budgets (Wei et al., 2022b).
To address these challenges, we implement a two-stage methodology to develop scaling laws that accurately
predict downstream benchmark performance:
1. We ﬁrst establish a correlation between the compute-optimal model’s negative log-likelihood on down-
stream tasks and the training FLOPs.
2. Next, we correlate the negative log-likelihood on downstream tasks with task accuracy, utilizing both the
scaling law models and older models trained with higher compute FLOPs. In this step, we speciﬁcally
leverage the Llama 2 family of models.
This approach enables us to predict downstream task performance given a speciﬁc number of training FLOPs
for compute-optimal models. We use a similar method to select our pre-training data mix (see Section 3.4).
Scaling law experiments. Concretely, we construct our scaling laws by pre-training models using compute
budgets between 6× 1018 FLOPs and 1022 FLOPs. At each compute budget, we pre-train models ranging
in size between 40M and 16B parameters, using a subset of model sizes at each compute budget. In these
training runs, we use a cosine learning rate schedule with a linear warmup for 2,000 training steps. The peak
learning rate is set between2× 10−4 and 4× 10−4 depending on the size of the model. We set the cosine
decay to 0.1 of the peak value. The weight decay at each step is set to 0.1 times the learning rate at that step.
We use a ﬁxed batch size for each compute scale, ranging between 250K and 4M.
3https://github.com/openai/tiktoken/tree/main
7
1010 1011 1012
Training Tokens
0.70
0.75
0.80
0.85
0.90
0.95Validation Loss
Compute
6e18
1e19
3e19
6e19
1e20
3e20
6e20
1e21
3e21
1e22
Figure 2 Scaling law IsoFLOPs curves between 6× 1018
and 1022 FLOPs. The loss is the negative log-
likelihood on a held-out validation set. We approx-
imate measurements at each compute scale using a
second degree polynomial.
1019 1020 1021 1022
Compute (FLOPs)
1010
1011
Training Tokens
Fitted Line,  = 0.537, A = 0.299
Figure 3 Number of training tokens in identified compute-
optimal models as a function of pre-training compute
budget. We include the ﬁtted scaling-law prediction
as well. The compute-optimal models correspond to
the parabola minimums in Figure 2.
These experiments give rise to the IsoFLOPs curves in Figure 2. The loss in these curves is measured on
a separate validation set. We ﬁt the measured loss values using a second-degree polynomial and identify
the minimums of each parabola. We refer to minimum of a parabola as thecompute-optimal model at the
corresponding pre-training compute budget.
We use the compute-optimal models we identiﬁed this way to predict the optimal number of training tokens
for a speciﬁc compute budget. To do so, we assume a power-law relation between compute budget,C, and
the optimal number of training tokens,N⋆(C):
N⋆(C) = ACα.
We ﬁtA and α using the data from Figure 2. We ﬁnd that(α,A ) = (0.53, 0.29); the corresponding ﬁt is
shown in Figure 3. Extrapolation of the resulting scaling law to3.8× 1025 FLOPs suggests training a 402B
parameter model on 16.55T tokens.
An important observation is that IsoFLOPs curves becomeﬂatter around the minimum as the compute
budget increases. This implies that performance of the ﬂagship model is relatively robust to small changes in
the trade-oﬀ between model size and training tokens. Based on this observation, we ultimately decided to
train a ﬂagship model with 405B parameters.
Predicting performance on downstream tasks. We use the resulting compute-optimal models to forecast
the performance of the ﬂagship Llama 3 model on benchmark data sets. First, we linearly correlate the
(normalized) negative log-likelihood of correct answer in the benchmark and the training FLOPs. In this
analysis, we use only the scaling law models trained up to1022 FLOPs on the data mix described above. Next,
we establish a sigmoidal relation between the log-likelihood and accuracy using both the scaling law models
and Llama 2 models, which were trained using the Llama 2 data mix and tokenizer. We show the results of
this experiment on the ARC Challenge benchmark in Figure 4). We ﬁnd this two-step scaling law prediction,
which extrapolates over four orders of magnitude, to be quite accurate: it only slightly underestimates the
ﬁnal performance of the ﬂagship Llama 3 model.
3.3 Infrastructure, Scaling, and Efficiency
We describe our hardware and infrastructure that powered Llama 3 405B pre-training at scale and discuss
several optimizations that leads to improvements in training eﬃciency.
3.3.1 Training Infrastructure
The Llama 1 and 2 models were trained on Meta’s AI Research SuperCluster (Lee and Sengupta, 2022). As
we scaled further, the training for Llama 3 was migrated to Meta’s production clusters (Lee et al., 2024).This
8
1020 1021 1022 1023 1024 1025
Compute (FLOPs)
1.200
1.225
1.250
1.275
1.300
1.325
1.350
1.375
1.400Normalized NLL per Char.
1.201.251.301.351.40
Normalized NLL per Char.
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0Accuracy
Scaling Law Models
Llama 2 Models
Scaling Law Prediction
Llama 3 405B
Figure 4 Scaling law forecast for ARC Challenge. Left: Normalized negative log-likelihood of the correct answer on the
ARC Challenge benchmark as a function of pre-training FLOPs.Right: ARC Challenge benchmark accuracy as a
function of the normalized negative log-likelihood of the correct answer. This analysis enables us to predict model
performance on the ARC Challenge benchmark before pre-training commences. See text for details.
setup optimizes for production-grade reliability, which is essential as we scale up training.
Compute. Llama 3 405B is trained on up to 16K H100 GPUs, each running at 700W TDP with 80GB HBM3,
using Meta’s Grand Teton AI server platform (Matt Bowman, 2022). Each server is equipped with eight GPUs
and two CPUs. Within a server, the eight GPUs are connected via NVLink. Training jobs are scheduled
using MAST (Choudhury et al., 2024), Meta’s global-scale training scheduler.
Storage. Tectonic (Pan et al., 2021), Meta’s general-purpose distributed ﬁle system, is used to build a storage
fabric (Battey and Gupta, 2024) for Llama 3 pre-training. It oﬀers 240 PB of storage out of 7,500 servers
equipped with SSDs, and supports a sustainable throughput of 2 TB/s and a peak throughput of 7 TB/s. A
major challenge is supporting the highly bursty checkpoint writes that saturate the storage fabric for short
durations. Checkpointing saves each GPU’s model state, ranging from 1 MB to 4 GB per GPU, for recovery
and debugging. We aim to minimize GPU pause time during checkpointing and increase checkpoint frequency
to reduce the amount of lost work after a recovery.
Network. Llama 3 405B used RDMA over Converged Ethernet (RoCE) fabric based on the Arista 7800
and Minipack2 Open Compute Project4 OCP rack switches. Smaller models in the Llama 3 family were
trained using Nvidia Quantum2 Inﬁniband fabric. Both RoCE and Inﬁniband clusters leverage 400 Gbps
interconnects between GPUs. Despite the underlying network technology diﬀerences between these clusters,
we tune both of them to provide equivalent performance for these large training workloads. We elaborate
further on our RoCE network since we fully own its design.
• Network topology. Our RoCE-based AI cluster comprises 24K GPUs5 connected by a three-layer Clos
network (Lee et al., 2024). At the bottom layer, each rack hosts 16 GPUs split between two servers and
connected by a single Minipack2 top-of-the-rack (ToR) switch. In the middle layer, 192 such racks are
connected by Cluster Switches to form a pod of 3,072 GPUs with full bisection bandwidth, ensuring no
oversubscription. At the top layer, eight such pods within the same datacenter building are connected via
Aggregation Switches to form a cluster of 24K GPUs. However, network connectivity at the aggregation
layer does not maintain full bisection bandwidth and instead has an oversubscription ratio of 1:7. Our
model parallelism methods (see Section 3.3.2) and training job scheduler (Choudhury et al., 2024) are
all optimized to be aware of network topology, aiming to minimize network communication across pods.
• Load balancing. LLM training produces fat network ﬂows that are hard to load balance across all
available network paths using traditional methods such as Equal-Cost Multi-Path (ECMP) routing. To
address this challenge, we employ two techniques. First, our collective library creates 16 network ﬂows
between two GPUs, instead of just one, thereby reducing the traﬃc per ﬂow and providing more ﬂows
4Open Compute Project: https://www.opencompute.org/
5Note that we use only up to 16K of these 24K GPUs for Llama 3 pre-training.
9
GPUs TP CP PP DP Seq. Len. Batch size/DP Tokens/Batch TFLOPs/GPU BF16 MFU
8,192 8 1 16 64 8,192 32 16M 430 43%
16,384 8 1 16 128 8,192 16 16M 400 41%
16,384 8 16 16 4 131,072 16 16M 380 38%
Table 4 Scaling configurations and MFU for each stage of Llama 3 405B pre-training. See text and Figure 5 for descriptions
of each type of parallelism.
for load balancing. Second, our Enhanced-ECMP (E-ECMP) protocol eﬀectively balances these 16 ﬂows
across diﬀerent network paths by hashing on additional ﬁelds in the RoCE header of packets.
• Congestion control. We use deep-buﬀer switches in the spine (Gangidi et al., 2024) to accommodate
transient congestion and buﬀering caused by collective communication patterns. This setup helps
limit the impact of persistent congestion and network back pressure caused by slow servers, which is
common in training. Finally, better load balancing through E-ECMP signiﬁcantly reduces the chance
of congestion. With these optimizations, we successfully run a 24K GPU cluster without traditional
congestion control methods such as Data Center Quantized Congestion Notiﬁcation (DCQCN).
3.3.2 Parallelism for Model Scaling
To scale training for our largest models, we use 4D parallelism—a combination of four diﬀerent types of
parallelism methods—to shard the model. This approach eﬃciently distributes computation across many
GPUs and ensures each GPU’s model parameters, optimizer states, gradients, and activations ﬁt in its
HBM. Our implementation of 4D parallelism is illustrated in Figure 5. It combines tensor parallelism (TP;
Krizhevsky et al. (2012); Shoeybi et al. (2019); Korthikanti et al. (2023)), pipeline parallelism (PP; Huang
et al. (2019); Narayanan et al. (2021); Lamy-Poirier (2023)), context parallelism (CP; Liu et al. (2023a)), and
data parallelism (DP; Rajbhandari et al. (2020); Ren et al. (2021); Zhao et al. (2023b)).
Tensor parallelism splits individual weight tensors into multiple chunks on diﬀerent devices. Pipeline parallelism
partitions the model vertically into stages by layers, so that diﬀerent devices can process in parallel diﬀerent
stages of the full model pipeline. Context parallelism divides the input context into segments, reducing memory
bottleneck for very long sequence length inputs. We use fully sharded data parallelism (FSDP; Rajbhandari
et al., 2020; Ren et al., 2021; Zhao et al., 2023b), which shards the model, optimizer, and gradients while
implementing data parallelism which processes data in parallel on multiple GPUs and synchronizes after each
training step. Our use of FSDP for Llama 3 shards optimizer states and gradients, but for model shards we do
not reshard after forward computation to avoid an extraall-gather communication during backward passes.
GPU utilization. Through careful tuning of the parallelism conﬁguration, hardware, and software, we achieve
an overall BF16 Model FLOPs Utilization (MFU; Chowdhery et al. (2023)) of 38-43% for the conﬁgurations
shown in Table 4. The slight drop in MFU to 41% on 16K GPUs with DP=128 compared to 43% on 8K
GPUs with DP=64 is due to the lower batch size per DP group needed to keep the global tokens per batch
constant during training.
Pipeline parallelism improvements. We encountered several challenges with existing implementations:
• Batch size constraint. Current implementations have constraints on supported batch size per GPU,
requiring it to be divisible by the number of pipeline stages. For the example in Figure 6, the depth-ﬁrst
schedule (DFS) of pipeline parallelism (Narayanan et al., 2021) requiresN = PP = 4 , while the
breadth-ﬁrst schedule (BFS; Lamy-Poirier (2023)) requiresN = M, where M is the total number
of micro-batches andN is the number of contiguous micro-batches for the same stage’s forward or
backward. However, pre-training often needs ﬂexibility to adjust batch size.
• Memory imbalance. Existing pipeline parallelism implementations lead to imbalanced resource consump-
tion. The ﬁrst stage consumes more memory due to the embedding and the warm-up micro-batches.
• Computation imbalance. After the last layer of the model, we need to calculate output and loss, making
this stage the execution latency bottleneck.
10
Figure 5 Illustration of 4D parallelism. GPUs are divided into parallelism groups in the order of [TP, CP, PP, DP], where
DP stands for FSDP. In this example, 16 GPUs are conﬁgured with a group size of |TP|=2, |CP|=2, |PP|=2, and
|DP|=2. A GPU’s position in 4D parallelism is represented as a vector, [D1, D2, D3, D4], where Di is the index on
the i-th parallelism dimension. In this example, GPU0[TP0, CP0, PP0, DP0] and GPU1[TP1, CP0, PP0, DP0] are in
the same TP group, GPU0 and GPU2 are in the same CP group, GPU0 and GPU4 are in the same PP group, and
GPU0 and GPU8 are in the same DP group.
To address these issues, we modify our pipeline schedule as shown in Figure 6, which allows settingN
ﬂexibly—in this case N = 5, which can run a arbitrary number of micro-batches in each batch. This allows
us to run: (1) fewer micro-batches than the number of stages when we have batch size limit at large scale;
or (2) more micro-batches to hide point-to-point communication, ﬁnding a sweet spot between DFS and
breadth ﬁrst schedule (BFS) for the best communication and memory eﬃciency. To balance the pipeline,
we reduce one Transformer layer each from the ﬁrst and the last stages, respectively. This means that
the ﬁrst model chunk on the ﬁrst stage has only the embedding, and the last model chunk on the last
stage has only output projection and loss calculation. To reduce pipeline bubbles, we use an interleaved
schedule (Narayanan et al., 2021) withV pipeline stages on one pipeline rank. Overall pipeline bubble ratio
is PP−1
V∗M . Further, we adopt asynchronous point-to-point communication in PP, which considerably speeds up
training, especially in cases when the document mask introduces extra computation imbalance. We enable
TORCH_NCCL_AVOID_RECORD_STREAMSto reduce memory usage from asynchronous point-to-point
communication. Finally, to reduce memory cost, based on detailed memory allocation proﬁling, we proactively
deallocate tensors that will not be used for future computation, including the input and output tensors of each
pipeline stage, that will not be used for future computation. With these optimizations, we could pre-train
Llama 3 on sequences of 8K tokens without activation checkpointing.
Context parallelism for long sequences. We utilize context parallelism (CP) to improve memory eﬃciency when
scaling the context length of Llama 3 and enable training on extremely long sequences up to 128K in length.
In CP, we partition across the sequence dimension, and speciﬁcally we partition the input sequence into
2×CP chunks so each CP rank receives two chunks for better load balancing. Thei-th CP rank received
both thei-th and the(2×CP− 1−i)-th chunks.
Diﬀerent from existing CP implementations that overlap communication and computation in a ring-like
structure (Liu et al., 2023a), our CP implementation adopts anall-gather based method where we ﬁrst
all-gather the key (K) and value (V) tensors, and then compute attention output for the local query (Q)
tensor chunk. Although theall-gather communication latency is exposed in the critical path, we still adopt
this approach for two main reasons: (1) it is easier and more ﬂexible to support diﬀerent types of attention
masks inall-gather based CP attention, such as the document mask; and (2) the exposedall-gather latency
11
Figure 6 Illustration of pipeline parallelism in Llama 3. Pipeline parallelism partitions eight pipeline stages (0 to 7) across
four pipeline ranks (PP ranks 0 to 3), where the GPUs with rank 0 run stages 0 and 4, the GPUs with P rank 1 run
stages 1 and 5,etc. The colored blocks (0 to 9) represent a sequence of micro-batches, whereM is the total number of
micro-batches and N is the number of continuous micro-batches for the same stage’s forward or backward. Our key
insight is to makeN tunable.
is small as the communicated K and V tensors are much smaller than Q tensor due to the use of GQA (Ainslie
et al., 2023). Hence, the time complexity of attention computation is an order of magnitude larger than
all-gather (O(S2) versusO(S), whereS represents the sequence length in the full causal mask), making the
all-gather overhead negligible.
Network-aware parallelism configuration. The order of parallelism dimensions, [TP, CP, PP, DP], is optimized
for network communication. The innermost parallelism requires the highest network bandwidth and lowest
latency, and hence is usually constrained to within the same server. The outermost parallelism may spread
across a multi-hop network and should tolerate higher network latency. Therefore, based on the requirements
for network bandwidth and latency, we place parallelism dimensions in the order of [TP, CP, PP, DP]. DP
(i.e., FSDP) is the outermost parallelism because it can tolerate longer network latency by asynchronously
prefetching sharded model weights and reducing gradients. Identifying the optimal parallelism conﬁguration
with minimal communication overhead while avoiding GPU memory overﬂow is challenging. We develop a
memory consumption estimator and a performance-projection tool which helped us explore various parallelism
conﬁgurations and project overall training performance and identify memory gaps eﬀectively.
Numerical stability. By comparing training loss between diﬀerent parallelism setups, we ﬁxed several numerical
issues that impact training stability. To ensure training convergence, we use FP32 gradient accumulation
during backward computation over multiple micro-batches and alsoreduce-scatter gradients in FP32 across
data parallel workers in FSDP. For intermediate tensors,e.g., vision encoder outputs, that are used multiple
times in the forward computation, the backward gradients are also accumulated in FP32.
3.3.3 Collective Communication
Our collective communication library for Llama 3 is based on a fork of Nvidia’s NCCL library, called NCCLX.
NCCLX signiﬁcantly improves the performance of NCCL, especially for higher latency networks. Recall that
the order of parallelism dimensions is [TP, CP, PP, DP], where DP corresponds to FSDP. The outermost
parallelism dimensions, PP and DP, may communicate through a multi-hop network, with latency up to tens
of microseconds. The original NCCL collectives—all-gather and reduce-scatter in FSDP, andpoint-to-point
in PP—require data chunking and staged data copy. This approach incurs several ineﬃciencies, including
(1) requiring a large number of small control messages to be exchanged over the network to facilitate data
transfer, (2) extra memory-copy operations, and (3) using extra GPU cycles for communication. For Llama 3
training, we address a subset of these ineﬃciencies by tuning chunking and data transfer to ﬁt our network
latencies, which can be as high as tens of microseconds for a large cluster. We also allow small control messages
to traverse our network at a higher priority, especially avoiding being head-of-line blocked in deep-buﬀer
core switches. Our ongoing work for future Llama versions involves making deeper changes in NCCLX to
holistically address all the aforementioned problems.
12
Component Category Interruption Count % of Interruptions
Faulty GPU GPU 148 30.1%
GPU HBM3 Memory GPU 72 17.2%
Software Bug Dependency 54 12.9%
Network Switch/Cable Network 35 8.4%
Host Maintenance Unplanned
Maintenance 32 7.6%
GPU SRAM Memory GPU 19 4.5%
GPU System Processor GPU 17 4.1%
NIC Host 7 1.7%
NCCL Watchdog Timeouts Unknown 7 1.7%
Silent Data Corruption GPU 6 1.4%
GPU Thermal Interface + Sensor GPU 6 1.4%
SSD Host 3 0.7%
Power Supply Host 3 0.7%
Server Chassis Host 2 0.5%
IO Expansion Board Host 2 0.5%
Dependency Dependency 2 0.5%
CPU Host 2 0.5%
System Memory Host 2 0.5%
Table 5 Root-cause categorization of unexpected interruptions during a 54-day period of Llama 3 405B pre-training. About
78% of unexpected interruptions were attributed to conﬁrmed or suspected hardware issues.
3.3.4 Reliability and Operational Challenges
The complexity and potential failure scenarios of 16K GPU training surpass those of much larger CPU clusters
that we have operated. Moreover, the synchronous nature of training makes it less fault-tolerant—a single
GPU failure may require a restart of the entire job. Despite these challenges, for Llama 3, we achieved higher
than 90% eﬀective training time while supporting automated cluster maintenance, such as ﬁrmware and Linux
kernel upgrades (Vigraham and Leonhardi, 2024), which resulted in at least one training interruption daily.
The eﬀective training time measures the time spent on useful training over the elapsed time.
During a 54-day snapshot period of pre-training, we experienced a total of 466 job interruptions. Of these, 47
were planned interruptions due to automated maintenance operations such as ﬁrmware upgrades or operator-
initiated operations like conﬁguration or dataset updates. The remaining 419 were unexpected interruptions,
which are classiﬁed in Table 5. Approximately 78% of the unexpected interruptions are attributed to conﬁrmed
hardware issues, such as GPU or host component failures, or suspected hardware-related issues like silent data
corruption and unplanned individual host maintenance events. GPU issues are the largest category, accounting
for 58.7% of all unexpected issues. Despite the large number of failures, signiﬁcant manual intervention was
required only three times during this period, with the rest of issues handled by automation.
To increase the eﬀective training time, we reduced job startup and checkpointing time, and developed tools
for fast diagnosis and problem resolution. We extensively use PyTorch’s built-in NCCL ﬂight recorder (Ansel
et al., 2024), a feature that captures collective metadata and stack traces into a ring buﬀer, and hence allowing
us to diagnose hangs and performance issues quickly at scale, particularly with regard to NCCLX. Using
this, we eﬃciently record every communication event and the duration of each collective operation, and also
automatically dump tracing data on NCCLX watchdog or heartbeat timeout. We enable more computationally
intensive tracing operations and metadata collection selectively as needed live in production through online
conﬁguration changes (Tang et al., 2015) without needing a code release or job restart.
Debugging issues in large-scale training is complicated by the mixed use of NVLink and RoCE in our network.
Data transfer over NVLink typically occurs through load/store operations issued by CUDA kernels, and
failures in either the remote GPU or NVLink connectivity often manifest as stalled load/store operations
within CUDA kernels without returning a clear error code. NCCLX enhances the speed and accuracy of failure
13
detection and localization through a tight co-design with PyTorch, allowing PyTorch to access NCCLX’s
internal state and track relevant information. While stalls due to NVLink failures cannot be completely
prevented, our system monitors the state of the communication library and automatically times out when
such a stall is detected. Additionally, NCCLX traces the kernel and network activities of each NCCLX
communication and provides a snapshot of the failing NCCLX collective’s internal state, including ﬁnished
and pending data transfers between all ranks. We analyze this data to debug NCCLX scaling issues.
Sometimes, hardware issues may cause still-functioning but slow stragglers that are hard to detect. Even a single
straggler can slow down thousands of other GPUs, often appearing as functioning but slow communications.
We developed tools to prioritize potentially problematic communications from selected process groups. By
investigating just a few top suspects, we were usually able to eﬀectively identify the stragglers.
One interesting observation is the impact of environmental factors on training performance at scale. For
Llama 3 405B , we noted a diurnal 1-2% throughput variation based on time-of-day. This ﬂuctuation is the
result of higher mid-day temperatures impacting GPU dynamic voltage and frequency scaling.
During training, tens of thousands of GPUs may increase or decrease power consumption at the same time,
for example, due to all GPUs waiting for checkpointing or collective communications to ﬁnish, or the startup
or shutdown of the entire training job. When this happens, it can result in instant ﬂuctuations of power
consumption across the data center on the order of tens of megawatts, stretching the limits of the power grid.
This is an ongoing challenge for us as we scale training for future, even larger Llama models.
3.4 Training Recipe
The recipe used to pre-train Llama 3 405B consists of three main stages:(1) initial pre-training,(2) long-context
pre-training, and (3) annealing. The three stages are described separately below. We use similar recipes to
pre-train the 8B and 70B models.
3.4.1 Initial Pre-Training
We pre-train Llama 3 405B using AdamW with a peak learning rate of8× 10−5, a linear warm up of 8,000
steps, and a cosine learning rate schedule decaying to8× 10−7 over 1,200,000 steps. We use a lower batch size
early in training to improve training stability, and increase it subsequently to improve eﬃciency. Speciﬁcally,
we use an initial batch size of 4M tokens and sequences of length 4,096, and double these values to a batch
size of 8M sequences of 8,192 tokens after pre-training 252M tokens. We double the batch size again to 16M
after pre-training on 2.87T tokens. We found this training recipe to be very stable: we observed few loss
spikes and did not require interventions to correct for model training divergence.
Adjusting the data mix. We made a several adjustments to the pre-training data mix during training to improve
model performance on particular downstream tasks. In particular, we increased the percentage of non-English
data during pre-training to improve the multilingual performance of Llama 3. We also upsample mathematical
data to improve the model’s mathematical reasoning performance, we added more recent web data in the
later stages of pre-training to advance the model’s knowledge cut-oﬀ, and we downsampled subsets of the
pre-training data that were later identiﬁed as being lower quality.
3.4.2 Long Context Pre-Training
In the ﬁnal stages of pre-training, we train on long sequences to support context windows of up to 128K tokens.
We do not train on long sequences earlier because the compute in self-attention layers grows quadratically in
the sequence length. We increase the supported context length in increments, pre-training until the model has
successfully adapted to the increased context length. We assess successful adaptation by measuring whether(1)
model performance on short-context evaluations has recovered completely and(2) the model perfectly solves
“needle in a haystack” tasks up to that length. In Llama 3 405B pre-training, we increased context length
gradually in six stages, starting from the original 8K context window and ending in the ﬁnal 128K context
window. This long-context pre-training stage was performed using approximately 800B training tokens.
14
Figure 7 Illustration of the overall post-training approach for Llama 3. Our post-training strategy involves rejection sampling,
supervised ﬁnetuning, and direct preference optimization. See text for details.
3.4.3 Annealing
During pre-training on the ﬁnal 40M tokens, we linearly annealed the learning rate to 0, maintaining a context
length of 128K tokens. During this annealing phase, we also adjusted the data mix to upsample data sources
of very high quality; see Section 3.1.3. Finally, we compute the average of model checkpoints (Polyak (1991)
averaging) during annealing to produce the ﬁnal pre-trained model.
4 Post-Training
We produce the aligned Llama 3 models by applying several rounds of post-training,6 or aligning the model
with human feedback (Ouyang et al., 2022; Rafailov et al., 2024) on top of a pre-trained checkpoint. Each
round of post-training involves supervised ﬁnetuning (SFT) followed by Direct Preference Optimization (DPO;
Rafailov et al., 2024) on examples collected either via human annotations or generated synthetically. Our
post-training modeling and data approaches are described in Sections 4.1 and 4.2 respectively. We further
detail custom data curation strategies to improve the reasoning, coding, factuality, multilingual, tool use, long
context, and precise instruction following in Section 4.3.
4.1 Modeling
The backbone of our post-training strategy is a reward model and a language model. We ﬁrst train a reward
model on top of the pre-trained checkpoint using human-annotated preference data (see Section 4.1.2). We
then ﬁnetune pre-trained checkpoints with supervised ﬁnetuning (SFT; see Section 4.1.3), and further align
the checkpoints with Direct Preference Optimization (DPO; see Section 4.1.4). This process is illustrated
in Figure 7. Unless otherwise noted, our modeling procedure applies to Llama 3 405B, and we refer to
Llama 3 405B as Llama 3 for simplicity.
4.1.1 Chat Dialog Format
To tune LLMs for human-AI interaction, we need to deﬁne a chat dialog protocol for the model to understand
human instructions and perform conversational tasks. Compared to its predecessor, Llama 3 has new
capabilities such as tool use (Section 4.3.5) which may require generating multiple messages and sending
6We use the term “post-training” to refer to any model training that happens outside of pre-training.
15
them to diﬀerent locations (e.g., user,ipython) within a single dialog turn. To support this, we design a new
multi-message chat protocol which uses various special header and termination tokens. The header tokens
are used to indicate the source and destination of each message in a conversation. Similarly, the termination
tokens indicate when it is the time to alternate between human and AI to speak.
4.1.2 Reward Modeling
We train a reward model (RM) covering diﬀerent capabilities on top of the pre-trained checkpoint. The
training objective is the same as Llama 2 except that we remove the margin term in the loss, as we observe
diminishing improvements after data scaling. Following Llama 2, we use all of our preference data for reward
modeling after ﬁltering out samples with similar responses. In addition to standard preference pair of (chosen,
rejected) response, annotations also create a third “edited response” for some prompts, where the chosen
response from the pair is further edited for improvement (see Section 4.2.1). Hence, each preference ranking
sample has two or three responses with clear ranking (edited > chosen > rejected). We concatenate the
prompt and multiple responses into a single row during training with responses randomly shuﬄed. This is an
approximation to the standard scenario of putting the responses in separate rows and computing the scores,
but in our ablations, this approach improves training eﬃciency without a loss in accuracy.
4.1.3 Supervised Finetuning
The reward model is then used to perform rejection sampling on our human annotation prompts, the details
of which are described in Section 4.2. Together with this rejection-sampled data and other data sources
(including synthetic data), we ﬁnetune the pre-trained language model using a standard cross entropy loss
on the target tokens (while masking loss on prompt tokens). More details about the data mix can be found
in Section 4.2. We refer to this stage assupervised ﬁnetuning(SFT; Wei et al., 2022a; Sanh et al., 2022;
Wang et al., 2022b), even though many of the training targets are model-generated. Our largest models are
ﬁnetuned with a learning rate of10−5 over the course of 8.5K to 9K steps. We found these hyperparameter
settings to work well across diﬀerent rounds and data mixes.
4.1.4 Direct Preference Optimization
We further train our SFT models with Direct Preference Optimization (DPO; Rafailov et al., 2024) for human
preference alignment. For training, we primarily use the most recent batches of preference data collected using
the best performing models from the previous alignment rounds. As a result, our training data conforms better
to the distribution of the policy model that is being optimized in each round. We also explored on-policy
algorithms such as PPO (Schulman et al., 2017), but found that DPO required less compute for large-scale
models and performed better, especially on instruction following benchmarks like IFEval (Zhou et al., 2023).
For Llama 3, we use a learning rate of10−5 and set theβ hyper-parameter to be 0.1. In addition, we apply
the following algorithmic modiﬁcations to DPO:
• Masking out formatting tokens in DPO loss : We mask out special formatting tokens including header
and termination tokens (described in Section 4.1.1) from both chosen and rejected responses in the
loss to stabilize DPO training. We observe that having these tokens contribute to the loss may lead
to undesired model behaviors such as tail repetition or abruptly generating termination tokens. We
hypothesize that this is due to the contrastive nature of the DPO loss – the presence of common tokens
in both chosen and rejected responses leads to a conﬂicting learning objective as the model needs to
increase and reduce the likelihood of these tokens simultaneously.
• Regularization with NLL loss: We add an additional negative log-likelihood (NLL) loss term with a scaling
coeﬃcient of 0.2 on the chosen sequences, similar to Pang et al. (2024). This helps further stabilize DPO
training by maintaining desired formatting for generation and preventing the decrease of log probability
of chosen responses (Pang et al., 2024; Pal et al., 2024).
4.1.5 Model Averaging
Finally, we average models obtained from experiments using various versions of data or hyperparameters at
each RM, SFT, or DPO stage (Izmailov et al., 2019; Wortsman et al., 2022; Li et al., 2022).
16
% of Avg. # turns Avg. # tokens Avg. # tokens Avg. # tokens
Dataset comparisons per dialog per example in prompt in response
General English 81.99% 4.1 1,000.4 36.4 271.2
Coding 6.93% 3.2 1,621.0 113.8 462.9
Multilingual 5.19% 1.8 1,299.4 77.1 420.9
Reasoning and tools 5.89% 1.6 707.7 46.6 129.9
Total 100% 3.8 1,041.6 44.5 284.0
Table 6 Statistics of human preference data. We list statistics of the internally collected human preference data used for
Llama 3 alignment. We ask annotators to perform multi-turn dialogues with the models and make comparisons among
responses at each turn. In post-processing, we split each dialogue to multiple examples at a turn level. Each example
consists of a prompt (including previous dialog if available) and a response (e.g., chosen or rejected response).
4.1.6 Iterative Rounds
Following Llama 2, we apply the above methods in six rounds. In each cycle, we collect new preference
annotations and SFT data, sampling synthetic data from the latest models.
4.2 Post-training Data
The post-training data composition plays a critical role in the usefulness and behavior of language models. In
this section, we discuss our human annotation procedures and preference data collection (Section 4.2.1), the
composition of our SFT data (Section 4.2.2), and methods for data quality control and cleaning (Section 4.2.3).
4.2.1 Preference Data
Our preference data annotation process is similar to Llama 2. We deploy multiple models for annotation after
each round and sample two responses from two diﬀerent models for each user prompt. These models can
be trained with diﬀerent data mixes and alignment recipes, allowing for diﬀerent capability strength (e.g.,
code expertise) and increased data diversity. We ask annotators to rate the strength of their preference by
categorizing it into one of four levels, based on how much more they prefer the chosen response over the
rejected one: signiﬁcantly better, better, slightly better, or marginally better. We also incorporate an editing
step after preference ranking to encourage annotators to further improve the preferred response. Annotators
edit the chosen response directly or prompt the model with feedback to reﬁne its own response. Consequently,
a portion of our preference data has three responses ranked (edited > chosen > rejected).
In Table 6, we report the statistics of preference annotations that we use for Llama 3 training. General English
covers multiple subcategories such as knowledge-based question and answering or precise instruction-following,
which fall outside the scope of speciﬁc capabilities. Compared to Llama 2, we observe an increase in the
average length of prompt and response, suggesting that we train Llama 3 on more complex tasks. In addition,
we implement a quality analysis and human evaluation process to rigorously assess the data collected, allowing
us to reﬁne our prompts and provide systematic, actionable feedback to annotators. For example, as Llama 3
improves after each round, we increase prompt complexity accordingly to target areas where the model lags.
In each round of post-training, we use all the preference data that is available at the time for reward modeling,
while only using the latest batches from various capabilities for DPO training. For both reward modeling and
DPO, we use samples that are labeled as the chosen response being signiﬁcantly better or better than the
rejected counterpart for training and discard samples with similar responses.
4.2.2 SFT Data
Our ﬁnetuning data is largely comprised of the following sources:
• Prompts from our human annotation collection with rejection-sampled responses.
• Synthetic data targeting speciﬁc capabilities (see Section 4.3 for more details).
17
Avg. # tokens Avg. # tokens
Dataset % of examples Avg. # turns Avg. # tokens in context in final response
General English 52.66% 6.3 974.0 656.7 317.1
Code 14.89% 2.7 753.3 378.8 374.5
Multilingual 3.01% 2.7 520.5 230.8 289.7
Exam-like 8.14% 2.3 297.8 124.4 173.4
Reasoning and tools 21.19% 3.1 661.6 359.8 301.9
Long context 0.11% 6.7 38,135.6 37,395.2 740.5
Total 100% 4.7 846.1 535.7 310.4
Table 7 Statistics of SFT data. We list internally collected SFT data used for Llama 3 alignment. Each SFT example
consists of a context (i.e., all conversation turns except the last one) and a ﬁnal response.
• Small amounts of human-curated data (see Section 4.3 for more details).
As our post-training rounds progress, we develop stronger Llama 3 variants that we use to collect larger
datasets that cover a wide range of complex capabilities. In this section, we discuss the details for the
rejection-sampling procedure and overall composition of our ﬁnal SFT datamix.
Rejection sampling. During rejection sampling (RS), for each prompt collected during human annotation
(Section 4.2.1) we sampleK (typically between 10 and 30) outputs from the latest chat model policy (usually
the best performing checkpoint from the previous post-training iteration, or the best performing checkpoint
for a particular capability) and use our reward model to select the best candidate, consistent with Bai et al.
(2022). In later rounds of post-training, we introduce system prompts to steer RS responses to conform with
desirable tone, style, or formatting, which might be diﬀerent for diﬀerent capabilities.
To increase the eﬃciency of rejection sampling, we adopt PagedAttention (Kwon et al., 2023). PagedAttention
enhances memory eﬃciency through dynamic key-value cache allocation. It supports arbitrary output lengths
by dynamically scheduling requests based on the current cache capacity. Unfortunately, this carries the risk of
swap-out when running out of memory. To eliminate such swap overhead, we deﬁne a maximum output length
and perform a request only if suﬃcient memory is available to ﬁt an output with that length. PagedAttention
also enables us to share the key-value cache pages for a prompt across all corresponding outputs. Together,
this leads to a throughput improvement of over2× during rejection sampling.
Overall data composition. Table 7 shows data statistics for each broad category of our “helpfulness” mix. While
SFT and preference data contain overlapping domains, they are curated diﬀerently, yielding distinct count
statistics. In Section 4.2.3 we describe techniques for categorizing topic, complexity, and quality of our data
samples. In each round of post-training, we adjust our overall data mix carefully across these axes to tune
performance across a wide range of benchmarks. Our ﬁnal data mix epochs multiple times on some high
quality sources and downsamples others.
4.2.3 Data Processing and Quality Control
Given that most of our training data ismodel-generated, it requires careful cleaning and quality control.
Data cleaning. In the early rounds, we observed a number of undesirable patterns common in our data, such
as excessive use of emojis or exclamation points. Therefore, we implement a series of rule-based data removal
and modiﬁcation strategies to ﬁlter or clean problematic data. For example, to mitigate overly-apologetic
tonal issues, we identify overused phrases (such as “I’m sorry” or “I apologize”) and carefully balance the
proportion of such samples in our dataset.
Data pruning. We also apply a collection of model-based techniques to remove low-quality training samples
and improve overall model performance:
• Topic classification: We ﬁrst ﬁnetune Llama 3 8B into a topic classiﬁer, and perform inference over
all data to classify it into both coarsely-grained buckets (“mathematical reasoning”) and ﬁne-grained
18
buckets (“geometry and trigonometry”).
• Quality scoring: We use both reward model and Llama-based signals to obtain a quality score for each
sample. For an RM-based score, we consider data that is in the top quartile of RM scores as high quality.
For a Llama-based score, we prompt Llama 3 checkpoint to rate each sample on a three-point scale for
general English data (accuracy, instruction following, and tone/presentation) and a two-point scale for
coding data (bug identiﬁcation and user intention), and consider samples that obtain the maximum
score as high quality. The RM and Llama-based scores have high disagreement rates, and we ﬁnd that
combining these signals yield the best recall on our internal test set. Ultimately, we select examples
that are marked as high quality by the RMor the Llama-based ﬁlter.
• Difficulty scoring: Because we are also interested in prioritizing examples that are more complex for
the model, we score data using two measures of diﬃculty: Instag (Lu et al., 2023) and Llama-based
scoring. For Instag, we prompt Llama 3 70B to perform intention tagging of SFT prompts, where more
intentions implies more complexity. We also prompt Llama 3 to measure the diﬃculty (Liu et al., 2024c)
of dialogs on a three-point scale.
• Semantic deduplication: Finally, we perform semantic deduplication (Abbas et al., 2023; Liu et al.,
2024c). We ﬁrst cluster complete dialogs using RoBERTa (Liu et al., 2019b) and within each cluster
sort them by quality score× diﬃculty score. We then do greedy selection by iterating through all sorted
examples, and only keeping the ones that have maximum cosine similarity less than a threshold to the
examples seen so far in the cluster.
4.3 Capabilities
We highlight special eﬀorts to improve performance for speciﬁc capabilities such as code (Section 4.3.1),
multilinguality (Section 4.3.2), math and reasoning (Section 4.3.3), long context (Section 4.3.4), tool use
(Section 4.3.5), factuality (Section 4.3.6), and steerability (Section 4.3.7).
4.3.1 Code
LLMs for code have received signiﬁcant attention since the release of Copilot and Codex (Chen et al., 2021).
Developers are now widely using these models to generate code snippets, debug, automate tasks, and improve
code quality. For Llama 3, we target improving and evaluating code generation, documentation, debugging,
and review capabilities for the following high priority programming languages: Python, Java, Javascript,
C/C++, Typescript, Rust, PHP, HTML/CSS, SQL, bash/shell. Here, we present our work on improving
these coding capabilities via training a code expert, generating synthetic data for SFT, improving formatting
with system prompt steering, and creating quality ﬁlters to remove bad samples from our training data.
Expert training. We train acode expert which we use to collect high quality human annotations for code
throughout subsequent rounds of post-training. This is accomplished by branching the main pre-training run
and continuing pre-training on a 1T token mix of mostly (>85%) code data. Continued pre-training on domain-
speciﬁc data has been shown to be eﬀective for improving performance in a speciﬁc domain (Gururangan
et al., 2020). We follow a recipe similar to that of CodeLlama (Rozière et al., 2023). For the last several
thousand steps of training we perform long-context ﬁnetuning (LCFT) to extend the expert’s context length
to 16K tokens on a high quality mix of repo-level code data. Finally, we follow the similar post-training
modeling recipes described in Section 4.1 to align this model, except with SFT and DPO data mixes primarily
targeting code. This model is also used for rejection sampling (Section 4.2.2) for coding prompts.
Synthetic data generation. During development, we identiﬁed key issues in code generation, including diﬃculty
in following instructions, code syntax errors, incorrect code generation, and diﬃculty in ﬁxing bugs. While
intensive human annotation could theoretically resolve these issues, synthetic data generation oﬀers a
complementary approach at a lower cost and higher scale, unconstrained by the expertise level of annotators.
As such, we use Llama 3 and the code expert to generate a large quantity of synthetic SFT dialogs.
We describe three high-level approaches for generating synthetic code data. In total, we generate over2.7M
synthetic examples which were used during SFT.
19
1. Synthetic data generation: execution feedback. The 8B and 70B models show signiﬁcant performance
improvements when trained on data generated by a larger, more competent model. However, our initial
experiments revealed that training Llama 3 405B on its own generated data is not helpful (and can
even degrade performance). To address this limitation, we introduced execution feedback as a source of
truth, enabling the model to learn from its mistakes and stay on track. In particular, we generate large
dataset of approximately one million synthetic coding dialogues using the following process:
• Problem description generation: First, we generate a large collection of programming problem
descriptions that span a diverse range of topics, including those in the long tail distribution. To
achieve this diversity, we sample random code snippets from various sources and prompt the model
to generate programming problems inspired by these examples. This allowed us to tap into a wide
range of topics and create a comprehensive set of problem descriptions (Wei et al., 2024).
• Solution generation: Then, we prompt Llama 3 to solve each problem in a given programming
language. We observe that adding general rules of good programming to the prompt improves the
generated solution quality. Also, we ﬁnd it is helpful to require the model to explain its thought
process in comments.
• Correctness analysis: After generating a solution, it is crucial to recognize that its correctness is
not guaranteed, and including incorrect solutions in the ﬁnetuning dataset could harm the model’s
quality. While we do not ensure complete correctness, we develop methods to approximate it. To
achieve this, we extract the source code from the generated solution and applied a combination of
static and dynamic analysis techniques to test its correctness, including:
– Static analysis: We run all generated code through a parser and a linter to ensure syntactic
correctness, catching errors such as syntax errors, use of uninitialized variables or non-imported
functions, code style issues, typing errors, and others.
– Unit test generation and execution : For each problem and solution, we prompt the model
to generate unit tests, executed in a containerized environment together with the solution,
catching run-time execution errors and some semantic errors.
• Error feedback and iterative self-correction: When a solution fails at any step, we prompt the
model to revise it. The prompt included the original problem description, the faulty solution,
and feedback from the parser/linter/tester (stdout, stderr/ and return code). After a unit test
execution failure, the model could either ﬁx the code to pass the existing tests or modify its unit
tests to accommodate the generated code. Only dialogs that pass all checks are included in the ﬁnal
dataset, used for supervised ﬁnetuning (SFT). Notably, we observed that about 20% of solutions
were initially incorrect but self-corrected, indicating that the model learned from the execution
feedback and improved its performance.
• Fine-tuning and iterative improvement: The ﬁnetuning process is conducted over multiple rounds,
with each round building on the previous one. After each round, the model is improved, generating
higher-quality synthetic data for the next round. This iterative process allows for progressive
reﬁnement and enhancement of the model’s performance.
2. Synthetic data generation: programming language translation. We observe a performance gap between
major programming languages (e.g., Python/C++) and less common ones (e.g., Typescript/PHP). This
is not surprising as we have less training data for less common programming languages. To mitigate
this, we supplement our existing data bytranslating data from common programming languages to
less common languages (similar to Chen et al. (2023) in the context of reasoning). This is achieved
by prompting Llama 3 and ensuring quality via syntax parsing, compilation, and execution. Figure 8
demonstrates an example of synthetic PHP code translated from Python. This improves performance
signiﬁcantly for less common languages as measured by the MultiPL-E (Cassano et al., 2023) benchmark.
3. Synthetic data generation: backtranslation. To improve certain coding capabilities (e.g., documentation,
explanations) where execution feedback is less informative for determining quality, we employ an
alternative multi-step approach. Using this procedure, we generated approximately 1.2M synthetic
20
Figure 8 Code translation example. We display an example of using Llama 3 to translate Python code (left) to PHP
code (right) to augment our SFT dataset with a wider range of programming languages.
Figure 9 Improving generated code quality with system prompts. Left: without system promptRight: with system prompt.
dialogs related to code explanation, generation, documentation, and debugging. Beginning with code
snippets from a variety of languages in our pre-training data:
• Generate: We prompt Llama 3 to generate data that represents our target capability (e.g., we add
comments and docstrings for the code snippet, or we ask the model to explain a piece of code).
• Backtranslate: We then prompt the model to “backtranslate” the synthetically generated data to
the original code (e.g., we prompt the model to generate code only from its documentation, or we
ask the model to generate code only from its explanation).
• Filter: Using the original code as a reference, we prompt the Llama 3 to determine the quality of
the output (e.g., we ask the model how faithful the backtranslated code is to the original). We
then use the generated examples that have the highest self-veriﬁcation scores in SFT.
System prompt steering during rejection sampling. During the rejection sampling process, we used code speciﬁc
system prompts to improve code readability, documentation, thoroughness, and speciﬁcity. Recall, from
Section 7 this data is used to ﬁnetune the language model. Figure 9 shows an example of how the system
prompt helps improve the generated code quality — it adds necessary comments, uses more informative
variable names, saves memory, etc.
Filtering training data with execution and model-as-judge signals. As described in Section 4.2.3, we occasionally
encounter quality issues in our rejection-sampled data, such as code blocks containing bugs. Detecting these
issues in our rejection-sampled data is not as straightforward as it is for oursynthetic code data, as the
rejection-sampled responses typically contain a mix of natural language and code for which the code may not
21
always be expected to be executable. (For example, user prompts may explicitly ask for pseudo-code or edits to
only a very small snippet of an executable program.) To address this, we utilize the “model-as-judge” approach,
where earlier versions of Llama 3 assess and assign a binary (0/1) score based on two criteria: code correctness
and code style. We retain only those samples that achieve a perfect score of 2. Initially, this stringent ﬁltering
led to a regression in downstream benchmark performance, primarily because it disproportionately removed
examples with challenging prompts. To counteract this, we strategically revise the responses of some coding
data categorized as most challenging until they met the Llama-based “model-as-judge” criteria. By reﬁning
these challenging problems, the coding data achieves a balance between quality and diﬃculty, resulting in
optimal downstream performance.
4.3.2 Multilinguality
We describe how we improve Llama 3’s multilingual capabilities, including training an expert specialized on
substantially more multilingual data, sourcing and generating high quality multilingual instruction tuning
data for German, French, Italian, Portuguese, Hindi, Spanish, and Thai, and tackling speciﬁc challenges of
multilingual language steering to enhance the overall performance of our model.
Expert training. Our Llama 3 pre-training data mix contains signiﬁcantly more English tokens than non-English
tokens. To collect higher quality human annotations in non-English languages, we train amultilingual expert by
branching oﬀ the pre-training run and continuing to pre-train on a data mix that consists of90% multilingual
tokens. We then perform post-training on this expert following Section 4.1. This expert model is then used to
collect higher quality annotations in non-English languages until pre-training was fully complete.
Multilingual data collection. Our multilingual SFT data is derived primarily from sources described below. The
overall distribution is 2.4% human annotations, 44.2% data from other NLP tasks, 18.8% rejection sampled
data, and 34.6% translated reasoning data.
• Human annotations: We collect high-quality, manually annotated data from linguists and native speakers.
These annotations mostly consist of open-ended prompts that represent real world use cases.
• Data from other NLP tasks: To further augment, we use multilingual training data from other tasks
and rewrite into dialog format. For example, we use data from exams-qa (Hardalov et al., 2020)
and Conic10k (Wu et al., 2023). To improve language alignment, we also use parallel texts from
GlobalVoices (Prokopidis et al., 2016) and Wikimedia (Tiedemann, 2012). We use LID based ﬁltering
and Blaser2.0 (Seamless Communication et al., 2023) to remove low quality data. For parallel text data,
instead of using the bitext pairs directly, we apply a multilingual template inspired by Wei et al. (2022a)
to better simulate real-life conversations in translation and language learning scenarios.
• Rejection sampled data: We apply rejection sampling on our human annotated prompts to generate
high-quality samples for ﬁnetuning, with few modiﬁcations compared to the process for English data:
– Generation: We explored randomly choosing the temperature hyperparameter from the range
0.2− 1 for diverse generations in early rounds of post-training. With high temperature, responses
for multilingual prompts can get creative and inspiring, but are also susceptible to unnecessary
or unnatural code-switching. In the ﬁnal round of post-training, we use a constant value of 0.6
to balance the trade-oﬀ. Additionally, we used specialized system prompts to improve response
format, structure and general readability.
– Selection: Prior to reward model based selection, we implement multilingual-speciﬁc checks to
ensure high language-match rate between the prompt and response (e.g., a romanized Hindi prompt
should not expect a response in Hindi Devanagari script).
• Translated data: We try to avoid using machine-translated data to ﬁnetune the model in order to
prevent translationese (Bizzoni et al., 2020; Muennighoﬀ et al., 2023) or possible name bias (Wang
et al., 2022a), gender bias (Savoldi et al., 2021), or cultural bias (Ji et al., 2023). Moreover, we aim to
prevent the model from being exposed only to tasks that are rooted in English cultural context, which
may not be representative of the linguistic and cultural diversity we aim to capture. We made one
exception to this and translated our synthetic quantitative reasoning data (see Section 4.3.3 for details)
to improve performance in quantitative reasoning in non-English languages. Due to the simple nature of
22
the language in these math problems, the translated samples were found to have little to no quality
issues. We observed strong gains on MGSM (Shi et al., 2022) from adding this translated data.
4.3.3 Math and Reasoning
We deﬁne reasoning as the ability to perform multi-step computations and arrive at the correct ﬁnal answer.
Several challenges guide our approach to training models that excel in mathematical reasoning:
• Lack of prompts: As the complexity of questions increases, the number of valid prompts or questions
for Supervised Fine-Tuning (SFT) decreases. This scarcity makes it diﬃcult to create diverse and
representative training datasets for teaching models various mathematical skills (Yu et al., 2023; Yue
et al., 2023; Luo et al., 2023; Mitra et al., 2024; Shao et al., 2024; Yue et al., 2024b).
• Lack of ground truth chain of thought: Eﬀective reasoning requires a step-by-step solution to facilitate
the reasoning process (Wei et al., 2022c). However, there is often a shortage of ground truth chains of
thought, which are essential for guiding the model how to break down the problem step-by-step and
reach the ﬁnal answer (Zelikman et al., 2022).
• Incorrect intermediate steps: When using model-generated chains of thought, the intermediate steps
may not always be correct (Cobbe et al., 2021; Uesato et al., 2022; Lightman et al., 2023; Wang et al.,
2023a). This inaccuracy can lead to incorrect ﬁnal answers and needs to be addressed.
• Teaching models to use external tools: Enhancing models to utilize external tools, such as code interpreters,
allows them to reason by interleaving code and text (Gao et al., 2023; Chen et al., 2022; Gou et al.,
2023). This capability can signiﬁcantly improve their problem-solving abilities.
• Discrepancy between training and inference: There is often a discrepancy between how the model is
ﬁnetuned during training and how it is used during inference. During inference, the ﬁnetuned model may
interact with humans or other models, requiring it to improve its reasoning using feedback. Ensuring
consistency between training and real-world usage is crucial for maintaining reasoning performance.
To address these challenges, we apply the following methodologies:
• Addressing the lack of prompts: We source relevant pre-training data from mathematical contexts and
converted it into a question-answer format which can then be used for supervised ﬁnetuning. Additionally,
we identify mathematical skills where the model under-performs and actively sourced prompts from
humans to teach models such skills. To facilitate this process, we create a taxonomy of mathematical
skills (Didolkar et al., 2024) and ask humans to provide relevant prompts/questions accordingly.
• Augmenting training data with step-wise reasoning traces : We use Llama 3 to generate step-by-step
solutions for a set of prompts. For each prompt, the model produces a variable number of generations.
These generations are then ﬁltered based on the correct answer (Li et al., 2024a). We also do self-
veriﬁcation where Llama 3 is used to verify whether a particular step-by-step solution is valid for a given
question. This process improves the quality of the ﬁnetuning data by eliminating instances where the
model does not produce valid reasoning traces.
• Filtering incorrect reasoning traces: We train outcome and stepwise reward models (Lightman et al., 2023;
Wang et al., 2023a) to ﬁlter training data where the intermediate reasoning steps were incorrect. These
reward models are used to eliminate data with invalid step-by-step reasoning, ensuring high-quality
data for ﬁnetuning. For more challenging prompts, we use Monte Carlo Tree Search (MCTS) with
learned step-wise reward models to generate valid reasoning traces, further enhancing the collection of
high-quality reasoning data (Xie et al., 2024).
• Interleaving code and text reasoning : We prompt Llama 3 to solve reasoning problems through a
combination of textual reasoning and associated Python code (Gou et al., 2023). Code execution is used
as a feedback signal to eliminate cases where the reasoning chain was not valid, ensuring the correctness
of the reasoning process.
• Learning from feedback and mistakes: To simulate human feedback, we utilize incorrect generations (i.e.,
generations leading to incorrect reasoning traces) and perform error correction by prompting Llama 3 to
23
yield correct generations (An et al., 2023b; Welleck et al., 2022; Madaan et al., 2024a). The iterative
process of using feedback from incorrect attempts and correcting them helps improve the model’s ability
to reason accurately and learn from its mistakes.
4.3.4 Long Context
During the ﬁnal pre-training stage, we extend the context length of Llama 3 from 8K tokens to 128K tokens
(see Section 3.4 for more details). Similar to pre-training, we ﬁnd that during ﬁnetuning we must carefully
tune the recipe to balance short and long-context capabilities.
SFT and synthetic data generation. Naively applying our existing SFT recipe with only short-context data
resulted in signiﬁcant regressions in long-context capabilities from pre-training, highlighting the need to
incorporate long-context data in our SFT data mix. In practice, however, it is largely impractical to get humans
to annotate such examples due to the tedious and time-consuming nature of reading lengthy contexts, so we
predominantly rely on synthetic data to ﬁll this gap. We use earlier versions of Llama 3 to generate synthetic
data based on the key long-context use-cases: (possibly multi-turn) question-answering, summarization for
long documents, and reasoning over code repositories, and describe them in greater detail below.
• Question answering: We carefully curate a set of long documents from our pre-training mix. We split
these documents into chunks of 8K tokens, and prompted an earlier version of the Llama 3 model to
generate QA pairs conditional on randomly selected chunks. During training, the whole document is
used as context.
• Summarization: We applied hierarchical summarization of long-context documents by ﬁrst summarizing
the chunks of 8K input length using our strongest Llama 3 8K context model and then summarizing
the summaries. During training we provide the full document and prompt the model to summarize the
document while preserving all the important details. We also generate QA pairs based on the summaries
of the documents and prompt the model with questions that require global understanding of the whole
long document.
• Long context code reasoning: We parse Python ﬁles to identifyimport statements and determine their
dependencies. From here, we select the most commonly depended-upon ﬁles, speciﬁcally those referenced
by at least ﬁve other ﬁles. We remove one of these key ﬁles from a repository and prompt the model to
identify which ﬁles depended on the missing ﬁle and to generate the necessary missing code.
We further categorize these synthetically generated samples based on the sequence length (16K, 32K, 64K
and 128K) to enable more ﬁne-grained targeting of input lengths.
Through careful ablations, we observe that mixing0.1% of synthetically generated long-context data with the
original short-context data optimizes the performance across both short-context and long-context benchmarks.
DPO. We observe that using only short context training data in DPO did not negatively impact long-context
performance as long as the SFT model is high quality in long context tasks. We suspect this is due to the
fact that our DPO recipe has fewer optimizer steps than SFT. Given this ﬁnding, we keep the standard
short-context recipe for DPO on top of our long-context SFT checkpoints.
4.3.5 Tool Use
Teaching LLMs to use tools such as search engines or code interpreters hugely expands the range of tasks
they can solve, transforming them from pure chat models into more general assistants (Nakano et al., 2021;
Thoppilan et al., 2022; Parisi et al., 2022; Gao et al., 2023; Mialon et al., 2023a; Schick et al., 2024). We train
Llama 3 to interact with the following tools:
• Search engine. Llama 3 is trained to use Brave Search7 to answer questions about recent events that go
beyond its knowledge cutoﬀ or that require retrieving a particular piece of information from the web.
• Python interpreter. Llama 3 can generate and execute code to perform complex computations, read ﬁles
uploaded by the user and solve tasks based on them such as question answering, summarization, data
analysis or visualization.
7https://brave.com/search/api/
24
• Mathematical computational engine. Llama 3 can use the Wolfram Alpha API8 to more accurately solve
math, science problems, or retrieve accurate information from Wolfram’s database.
The resulting model is able to use these tools in a chat setup to solve the user’s queries, including in multi-turn
dialogs. If a query requires multiple tool calls, the model can write a step-by-step plan, call the tools in
sequence, and do reasoning after each tool call.
We also improve Llama 3’s zero-shot tool use capabilities — given in-context, potentially unseen tool deﬁnitions
and a user query, we train the model to generate the correct tool call.
Implementation. We implement our core tools as Python objects with diﬀerent methods. Zero-shot tools can
be implemented as Python functions with descriptions, documentation (i.e., examples for how to use them),
and the model only needs the function’s signature and docstring as context to generate the appropriate call.
We also convert function deﬁnitions and calls to JSON format, e.g., for web API calls. All tool calls are
executed by the Python interpreter, that must be enabled in the Llama 3 system prompt. Core tools can be
individually enabled or disabled in the system prompt.
Data collection. Diﬀerent from Schick et al. (2024), we rely on human annotations and preferences to teach
Llama 3 to use tools. There are two main diﬀerences with the post-training pipeline generally used in Llama 3:
• For tools, dialogs often contain more than a single assistant message (e.g., calling the tool and reasoning
about the tool output). Thus, we annotate at the message level to collect granular feedback: annotators
provide a preference between two assistant messages with the same context or, if both contain major
problems, edit one of the messages. The chosen or edited message is then added to the context and the
dialog continues. This provides human feedback for both the assistant’s ability of calling the tools and
reasoning about the tool outputs. Annotators cannot rank or edit the tool outputs.
• We do not perform rejection sampling, as we did not observe gains in our tool benchmarks.
To accelerate the annotation process, we start by bootstrapping basic tool use capabilities by ﬁnetuning on
synthetically generated data from previous Llama 3 checkpoints. Thus, annotators have fewer edits to perform.
In a similar spirit, as Llama 3 gradually improves through its development, we progressively complexify our
human annotation protocols: we start by single-turn tool use annotations, before moving to tool use in dialogs,
and ﬁnally annotating for multi-step tool use and data analysis.
Tool datasets. To create data for tool usage applications, we leverage the following procedure:
• Single-step tool use: We start by few-shot generation of synthetic user prompts which, by construction,
require a call to one of our core tools (for example, questions that exceed our knowledge cutoﬀ date).
Then, still relying on few-shot generation, we generate appropriate tool calls for these prompts, execute
them, and add the output to the model’s context. Finally, we prompt the model again to generate a
ﬁnal answer to the user’s query based on the tool output. We end up with trajectories of the following
form: system prompt, user prompt, tool call, tool output, ﬁnal answer. We also ﬁlter around30% this
dataset to remove tool calls that cannot be executed or other formatting issues.
• Multi-step tool use: We follow a similar protocol and ﬁrst generate synthetic data to teach the model
basic multi-step tool use capabilities. To do this, we ﬁrst prompt Llama 3 to generate user prompts
that require at least two tool calls, that can be the same or diﬀerent tools from our core set. Then,
conditioned on these prompts, we few-shot prompt Llama 3 to generate a solution consisting of interleaved
reasoning steps and tool calls, similar to ReAct (Yao et al., 2022). See Figure 10 for an example of
Llama 3 performing a task involving multi-step tool usage.
• File uploads: We annotate for the following ﬁletypes:.txt, .docx, .pdf, .pptx, .xlsx, .csv, .tsv,
.py, .json, .jsonl, .html, .xml . Our prompts are based on a provided ﬁle, and ask to summarize the
contents of the ﬁle, ﬁnd and ﬁx bugs, optimize a piece of code, perform data analysis or visualization.
See Figure 11 for an example of Llama 3 performing a task involving a ﬁle upload.
After ﬁnetuning on this synthetic data, we gather human annotations in diverse and challenging scenarios
including multi-turn interactions, more than three step tool use, and instances where a tool call does not yield
8https://products.wolframalpha.com/llm-api/documentation
25
Figure 10 Multi-step tool usage. Example of Llama 3 performing multi-step planning, reasoning, and tool calling to
solve a task.
a satisfying answer. We augment our synthetic data with diﬀerent system prompts to teach the model to use
tools only when activated. To train the model to avoid calling tools for simple queries, we also add queries
from easy math or question answering datasets (Berant et al., 2013; Koncel-Kedziorski et al., 2016; Joshi
et al., 2017; Amini et al., 2019) and their responses without tools, but with tools activated in system prompt.
Zero-shot tool use data. We improve Llama 3 zero-shot tool use abilities (also referred to as function calling)
by ﬁnetuning on a large and diverse set of partly synthetic (functions deﬁnitions, user query, corresponding
call) tuples. We evaluate our model on a set of unseen tools.
• Single, nested, and parallel function calling: Calls can be simple, nested,i.e. we pass a function call as an
argument of another function, or parallel,i.e. the model returns a list of independent function calls.
Generating a diverse set of functions, queries and ground truths can be challenging (Mekala et al., 2024),
and we resort to mining the Stack (Kocetkov et al., 2022) to ground our synthetic user queries in real
functions. More precisely, we extract function calls and their deﬁnitions, clean and ﬁlter them,e.g. for
missing docstrings or non-executable functions, and use Llama 3 to generate a natural language query
corresponding to the function call.
• Multi-turn function calling: We also generate synthetic data for multi-turn dialogs with function calls,
following a protocol similar to the one proposed in Li et al. (2023b). We use multiple agents that
generate domains, APIs, user queries, API calls, and responses, while also ensuring that the generated
data covers a set of diverse domains and realistic APIs. All agents are variants of Llama 3 prompted in
diﬀerent ways depending on their roles and collaborate in a step-by-step manner.
4.3.6 Factuality
Hallucinations remain a major challenge for large language models. Models tend to be overconﬁdent, even in
domains where they have little knowledge. Despite these shortcomings, they are often used as knowledge bases,
which can lead to risky outcomes such as the spread of misinformation. While we recognize that factuality
can go beyond hallucinations, we took a hallucination-ﬁrst approach here.
26
Figure 11 Processing file uploads. Example of Llama 3 performing analysis and visualization of an uploaded ﬁle.
We follow the principle that post-training should align the model to “know what it knows” rather than add
knowledge (Gekhman et al., 2024; Mielke et al., 2020). Our primary approach involves generating data that
aligns model generations with subsets of factual data present in the pre-training data. To achieve this, we
develop a knowledge probing technique that takes advantage of Llama 3’s in-context abilities. This data
generation process involves the following procedure:
1. Extract a data snippet from the pre-training data.
2. Generate a factual question about these snippets (context) by prompting Llama 3.
3. Sample responses from Llama 3 to the question.
4. Score the correctness of the generations using the original context as a reference and Llama 3 as a judge.
5. Score the informativeness of the generations using Llama 3 as a judge.
6. Generate a refusal for responses which are consistently informative and incorrect across the generations,
using Llama 3.
We use data generated from the knowledge probe to encourage the model to only answer questions which it
has knowledge about, and refuse answering those questions that it is unsure about. Further, pre-training data
is not always factually consistent or correct. We therefore also collect a limited set of labeled factuality data
that deals with sensitive topics where factually contradictory or incorrect statements are prevalent.
27
4.3.7 Steerability
Steerability is the ability to direct the model’s actions and outcomes to meet developer and user speciﬁcations.
As Llama 3 is a generic foundational model, it should be maximally steerable to diﬀerent downstream use
cases easily. For Llama 3, we focus on enhancing its steerability through system prompt with natural language
instructions, especially around response length, format, tone and character/persona.
Data collection. We collect steerability preference samples within the general English category by asking
annotators to design diﬀerent system prompts for Llama 3. Annotators then engage in conversations with the
models to evaluate their consistency in following instructions deﬁned in system prompts over the course of the
conversation. We show an example customized system prompt used for enhancing steerability below:
You are a helpful and cheerful AI Chatbot that acts as a meal plan assistant for busy families.
The family consists of 2 adults, 3 teenagers, and 2 preschoolers. Plan two or three days at a time
and use leftovers or extra ingredients for the second day’s plan. The user will let you know if they
want two or three days. If they don’t, assume three days. Each plan should include breakfast,
lunch, snack, and dinner. Ask the user if they approve of the plan or need adjustments. After they
approve provide a grocery list with family size in mind. Always keep family preferences in mind
and if there’s something that they don’t like provide a substitution. If the user is not feeling
inspired then ask them what’s the one place they wish they could visit on vacation this week
and then suggest meals based on that location’s culture. Weekend meals can be more complex.
Weekday meals should be quick and easy. For breakfast and lunch, easy food like cereal, English
muﬃns with pre-cooked bacon, and other quick easy foods are preferred. The family is busy. Be
sure to ask if they have essentials and favorites on hand like coﬀee or energy drinks so they don’t
forget to buy it. Remember to be budget-conscious unless it’s a special occasion.
Modeling. After we collect the preference data, we leverage this data in reward modeling, rejection sampling,
SFT, and DPO to enhance Llama 3’s steerability.
5 Results
We performed an extensive series of evaluations of Llama 3, investigating the performance of:(1) the pre-trained
language model, (2) the post-trained language model, and(3) the safety characteristics of Llama 3. We present
the results of these evaluations in separate subsections below.
5.1 Pre-trained Language Model
In this section, we report evaluation results for our pre-trained Llama 3 (Section 3), comparing with various
other models of comparable sizes. We reproduce results of competitor models whenever possible. For non-
Llama models, we report the best score across results that are publicly reported or (where possible) that we
reproduced ourselves. The speciﬁcs of these evaluations, including conﬁgurations such as the number of shots,
metrics, and other pertinent hyperparameters and settings, can be accessed on our Github repository here.
Additionally, we are releasing the data generated as part of evaluations with publicly available benchmarks
which can be found on Huggingface here. We evaluate the quality of our models on standard benchmarks
(Section 5.1.1), for robustness to changes in multiple-choice question setups (Section 5.1.2), and on adversarial
evaluations (Section 5.1.3). We also conduct a contamination analysis to estimate the extent to which our
evaluations are impacted by contamination of training data (Section 5.1.4).
5.1.1 Standard Benchmarks
To compare our models with the current state-of-the-art, we evaluate Llama 3 on a large number of standard
benchmark evaluations shown in Table 8. These evaluations cover eight top-level categories:(1) commonsense
reasoning; (2) knowledge; (3) reading comprehension; (4) math, reasoning, and problem solving;(5) long
context; (6) code; (7) adversarial evaluations; and(8) aggregate evaluations.
28
Reading Comprehension SQuAD V2 (Rajpurkar et al., 2018), QuaC (Choi et al., 2018),
RACE (Lai et al., 2017),
Code HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021),
Commonsense
reasoning/understanding
CommonSenseQA (Talmor et al., 2019), PiQA (Bisk et al., 2020),
SiQA (Sap et al., 2019), OpenBookQA (Mihaylov et al., 2018),
WinoGrande (Sakaguchi et al., 2021)
Math, reasoning, and problem solving
GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b),
ARC Challenge (Clark et al., 2018), DROP (Dua et al., 2019),
WorldSense (Benchekroun et al., 2023)
Adversarial
Adv SQuAD (Jia and Liang, 2017),
Dynabench SQuAD (Kiela et al., 2021), GSM-Plus (Li et al., 2024c)
PAWS (Zhang et al., 2019)
Long context QuALITY (Pang et al., 2022), many-shot GSM8K (An et al., 2023a)
Aggregate
MMLU (Hendrycks et al., 2021a),
MMLU-Pro (Wang et al., 2024b),
AGIEval (Zhong et al., 2023),
BIG-Bench Hard (Suzgun et al., 2023)
Table 8 Pre-training benchmarks by category. Overview of all benchmarks we use to evaluate pre-trained Llama 3 models,
grouped by capability category.
Experimental setup. For each benchmark, we compute scores for Llama 3 as well as various other pre-trained
models of comparable sizes. Where possible, we recompute numbers with our own pipeline for other models.
To ensure a fair comparison, we then select the best score between the score that we computed and the
reported number for that model with comparable or more conservative settings. You can ﬁnd additional
details on our evaluation setup here. For some models, it is not possible to (re)compute benchmark values,
for instance, because the pre-trained model is not released or because the API does not provide access to
log-probabilities. In particular, this is true for all models comparable to Llama 3 405B. Thus, we do not
report category averages for Llama 3 405B, which requires that all numbers are available for all benchmarks.
Significance estimates. Benchmark scores are estimates of a model’s true performance. These estimates
have variance because benchmark sets are ﬁnite samples drawn from some underlying distribution. We
follow Madaan et al. (2024b) and report on this variance via 95% conﬁdence intervals (CIs), assuming that
benchmark scores are Gaussian distributed. While this assumption is incorrect (e.g., benchmark scores are
bounded), preliminary bootstrap experiments suggest CIs (for discrete metrics) are a good approximation:
CI (S) = 1.96×
√
S× (1−S)
N .
Herein, S is the observed benchmark score (e.g., accuracy or EM) andN the sample size of the benchmark.
We omit CIs for benchmark scores that are not simple averages. We note that because subsampling is not the
only source of variation, our CI values lower bound the actual variation in the capability estimate.
Results for 8B and 70B models. Figure 12 reports the average performance of Llama 3 8B and 70B on the
commonsense reasoning, knowledge, reading comprehension, math and reasoning, and code benchmarks. The
results show that Llama 3 8B outperforms competing models in virtually every category, both in terms of
per-category win rate and in terms of average per-category performance. We also ﬁnd that Llama 3 70B
outperforms its predecessor Llama 2 70B by a large margin on most benchmarks, with the exception of
commonsense benchmarks that are likely saturated. Llama 3 70B also outperforms Mixtral 8x22B.
Detailed results for all models. Table 9, 10, 11, 12, 13, and 14 present the benchmark performance of pre-trained
Llama 3 8B, 70B, and 405B models on reading comprehension tasks, coding tasks, commonsense understanding
tasks, mathematical reasoning tasks, and general tasks. The tables compare Llama 3’s performance with that
29
General
Commonsense
Knowledge
Math and Reasoning
Reading Comprehension
Code
30
40
50
60
70
80
90Model quality
Model
Llama 2 7B
Llama 3 8B
Mistral 7B
Gemma 7B
General
Commonsense
Knowledge
Math and Reasoning
Reading Comprehension
Code
30
40
50
60
70
80
90Model quality
Model
Llama 2 70B
Llama 3 70B
Mixtral 8x22B
Figure 12 Performance of pre-trained Llama 3 8B and 70B models on pre-training benchmarks. Results are aggregated by
capability category by averaging accuracies across all benchmarks corresponding to that category.
Reading Comprehension
SQuAD QuAC RACE
Llama 3 8B 77.0±0.8 44.9±1.1 54.3±1.4
Mistral 7B 73.2±0.8 44.7±1.1 53.0±1.4
Gemma 7B 81.8±0.7 42.4±1.1 48.8±1.4
Llama 3 70B 81.8±0.7 51.1±1.1 59.0±1.4
Mixtral 8×22B 84.1±0.7 44.9±1.1 59.2±1.4
Llama 3 405B 81.8±0.7 53.6±1.1 58.1±1.4
GPT-4 – – –
Nemotron 4 340B – – –
Gemini Ultra – – –
Table 9 Pre-trained model performance on reading compre-
hension tasks. Results include 95% conﬁdence intervals.
Code
HumanEval MBPP
Llama 3 8B 37.2±7.4 47.6±4.4
Mistral 7B 30.5±7.0 47.5±4.4
Gemma 7B 32.3±7.2 44.4±4.4
Llama 3 70B 58.5±7.5 66.2±4.1
Mixtral 8×22B 45.1±7.6 71.2±4.0
Llama 3 405B 61.0±7.5 73.4±3.9
GPT-4 67.0±7.2 –
Nemotron 4 340B 57.3±7.6 –
Gemini Ultra 74.4±6.7 –
Table 10 Pre-trained model performance on coding tasks.
Results include 95% conﬁdence intervals.
of models of similar size. The results show that Llama 3 405B performs competitively with other models in
its class. In particular, Llama 3 405B substantially outperforms prior open-source models. For long-context,
we present more comprehensive results (including probing tasks like needle-in-a-haystack) in Section 5.2.
5.1.2 Model Robustness
In addition to performance on benchmarks, robustness is an important factor in the quality of pre-trained
language models. We investigate the robustness of our pre-trained language models to design choices in
multiple-choice question (MCQ) setups. Prior work has reported that model performance can be sensitive to
seemingly arbitrary design choices in such setups, for example, model scores and even rankings may change
with the order and labels of the in-context examples (Lu et al., 2022; Zhao et al., 2021; Robinson and Wingate,
2023; Liang et al., 2022; Gupta et al., 2024), the exact format of the prompt (Weber et al., 2023b; Mishra
et al., 2022), or the answer choice format and order (Alzahrani et al., 2024; Wang et al., 2024a; Zheng et al.,
2023). Motivated by this work, we use the MMLU benchmark to evaluate the robustness of our pre-trained
models to: (1) few-shot label bias,(2) label variants, (3) answer order, and(4) prompt format:
• Few-shot label bias. Following Zheng et al. (2023) and Weber et al. (2023a), we investigate the impact
of the distribution of labels in four-shot examples. Speciﬁcally, we consider settings in which: (1) all
30
Commonsense Understanding
CommonSenseQA PiQA SiQA OpenBookQA Winogrande
Llama 3 8B 75.0±2.5 81.0±1.8 49.5±2.2 45.0±4.4 75.7±2.0
Mistral 7B 71.2±2.6 83.0±1.7 48.2±2.2 47.8±4.4 78.1±1.9
Gemma 7B 74.4±2.5 81.5±1.8 51.8±2.2 52.8±4.4 74.7±2.0
Llama 3 70B 84.1±2.1 83.8±1.7 52.2±2.2 47.6±4.4 83.5±1.7
Mixtral 8×22B 82.4±2.2 85.5±1.6 51.6±2.2 50.8±4.4 84.7±1.7
Llama 3 405B 85.8±2.0 85.6±1.6 53.7±2.2 49.2±4.4 82.2±1.8
GPT-4 – – – – 87.5±1.5
Nemotron 4 340B – – – – 89.5±1.4
Table 11 Pre-trained model performance on commonsense understanding tasks. Results include 95% conﬁdence intervals.
Math and Reasoning
GSM8K MATH ARC-C DROP WorldSense
Llama 3 8B 57.2±2.7 20.3±1.1 79.7±2.3 59.5±1.0 45.5±0.3
Mistral 7B 52.5±2.7 13.1±0.9 78.2±2.4 53.0±1.0 44.9±0.3
Gemma 7B 46.4±2.7 24.3±1.2 78.6±2.4 56.3±1.0 46.0±0.3
Llama 3 70B 83.7±2.0 41.4±1.4 92.9±1.5 79.6±0.8 61.1±0.3
Mixtral 8×22B 88.4±1.7 41.8±1.4 91.9±1.6 77.5±0.8 51.5±0.3
Llama 3 405B 89.0±1.7 53.8±1.4 96.1±1.1 84.8±0.7 63.7±0.3
GPT-4 92.0±1.5 – 96.3±1.1 80.9±0.8 –
Nemotron 4 340B – – 94.3±1.3 – –
Gemini Ultra 88.9♦±1.7 53.2±1.4 – 82.4△±0.8 –
Table 12 Pre-trained model performance on math and reasoning tasks. Results include 95% conﬁdence intervals.♦11-shot.
△Variable shot.
General
MMLU MMLU-Pro AGIEval BB Hard
Llama 3 8B 66.7 37.1 47.8±1.9 64.2±1.2
Mistral 7B 63.6 32.5 42.7±1.9 56.8±1.2
Gemma 7B 64.3 35.1 46.0±1.9 57.7±1.2
Llama 3 70B 79.3 53.8 64.6±1.9 81.6±0.9
Mixtral 8×22B 77.8 51.5 61.5±1.9 79.5±1.0
Llama 3 405B 85.2 61.6 71.6±1.8 85.9±0.8
GPT-4 86.4 – – –
Nemotron 4 340B 81.1 – – 85.4±0.9
Gemini Ultra 83.7 – – 83.6±0.9
Table 13 Pre-trained model performance on general language tasks. Results include 95% conﬁdence intervals.
31
[A. B. C. D.] [A) B) C) D)] [1 2 3 4] [$ & # @] [  §  ü]
30
40
50
60
70
80
90Micro accuracy
Llama 3 8B
Llama 3 70B
Llama 3 405B
Llama 3 8B Llama 3 70B Llama 3 405B
30
40
50
60
70
80
90
100Micro accuracy
ABCD
AADD
BBCC
AAAA
Figure 13 Robustness of our pre-trained language models to different design choices in the MMLU benchmark.Left: Performance
for diﬀerent label variants.Right: Performance for diﬀerent labels present in few-shot examples.
Llama 3 8B Llama 3 70B Llama 3 405B
60
65
70
75
80
85
90
95
100Micro accuracy
Permutation distance
0
2
3
4
Llama 3 8B Llama 3 70B Llama 3 405B
65
70
75
80
85Micro accuracy
Figure 14 Robustness of our pre-trained language models to different design choices in the MMLU benchmark.Left: Performance
for diﬀerent answer orders.Right: Performance for diﬀerent prompt formats.
few-shot examples have the same label (A A A A); (2) all examples have a diﬀerent label (A B C D);
and (3) there are only two labels present (A A B Band C C D D).
• Label variants. We also study model response to diﬀerent choice token sets. We consider the two sets
proposed by Alzahrani et al. (2024): namely, a set of common language independent tokens ($ & #
@) and a of rare tokens (œ § з ü) that do not have any implicit relative order. We also consider two
versions of the canonical labels (A. B. C. D.and A) B) C) D)) and a numerical list (1. 2. 3. 4.).
• Answer order. Following Wang et al. (2024a), we compute how stable the results are across diﬀerent
answer orders. To compute this, we remap all the answers in the dataset according to a ﬁxed permutation.
For example, for the permutationA B C D, all answer options with labelA and B keep their label, and
all answer options with labelC get labelD, and vice versa.
• Prompt format. We evaluate variance in performance across ﬁve task prompts that diﬀer in the level of
information provided: one prompt simply asks the model to answer the question, whereas other prompts
assert the expertise of the model or that the best answer should be chosen.
Figure 13 presents the results of our experiments studying robustness of model performance to label variants
(left) and few-shot label bias (right). The results show that our pre-trained language models are very robust
to changes in MCQ labels and to the structure of the few-shot prompt labels. This robustness is particularly
32
0.0 0.2 0.4 0.6 0.8 1.0
Non-adversarial score
0.0
0.2
0.4
0.6
0.8
1.0Adversarial score
Size
8B
70B
405B
Category
Question answering
Paraphrase detection
Mathematical reasoning
0.0 0.2 0.4 0.6 0.8 1.0
Non-adversarial score
0.0
0.2
0.4
0.6
0.8
1.0Adversarial score
Size
8B
70B
405B
Category
Question answering
Paraphrase detection
Mathematical reasoning
Figure 15 Adversarial versus non-adversarial performance for question answering, mathematical reasoning, and paraphrase
detection benchmarks. Left: Results for pre-trained models.Right: Results for post-trained models.
pronounced for the 405B parameter model. Figure 14 presents the results of our study of robustness to answer
order and prompt format. The results in the ﬁgure further underscore the robustness of the performance of
our pre-trained language models, in particular, of Llama 3 405B.
5.1.3 Adversarial Benchmarks
In addition to the benchmarks presented above, we evaluate on several adversarial benchmarks in three areas:
question answering, mathematical reasoning, and paraphrase detection. This testing probes the model’s
capabilities on tasks speciﬁcally created to be challenging and can potentially also point to overﬁtting on
benchmarks. For question answering, we use Adversarial SQuAD (Jia and Liang, 2017) and Dynabench
SQuAD (Kiela et al., 2021). For mathematical reasoning, we use GSM-Plus (Li et al., 2024c). For paraphrase
detection, we use PAWS (Zhang et al., 2019).
Figure 15 presents the scores of Llama 3 8B, 70B, and 405B on the adversarial benchmarks as a function of their
performance on non-adversarial benchmarks. The non-adversarial benchmarks we use are SQuAD (Rajpurkar
et al., 2016) for question answering, GSM8K for mathematical reasoning, and QQP (Wang et al., 2017) for
paraphrase detection. Each datapoint represents a pair of an adversarial and non-adversarial datasets (e.g.
QQP paired with PAWS), and we show all possible pairs within a category. The diagonal black line represents
parity between adversarial and non-adversarial datasets — being on the line would indicate the model has
similar performance regardless of the adversarial nature.
On paraphrase detection, neither pre-trained nor post-trained models appear to suﬀer from the type of
adversariality with which PAWS was constructed, marking a substantial step with respect to the previous
generation of models. This result conﬁrms the ﬁndings of Weber et al. (2023a), who also found that LLMs are
less susceptible to the type of spurious correlations found in several adversarial datasets. For mathematical
reasoning and question answering, however, the adversarial performances are substantially lower than the
non-adversarial performances. This pattern is similar for pre-trained and post-trained models.
5.1.4 Contamination Analysis
We conduct a contamination analysis to estimate to what extent benchmark scores may be inﬂuenced
by contamination of the evaluation data in the pre-training corpus. In previous work, several diﬀerent
contamination methods have been used, with various diﬀerent hyperparameters – we refer to Singh et al.
(2024) for an overview. Any of these methods can suﬀer from false positives and negatives, and how to best
run contamination analyses is currently still an open ﬁeld of research. Here, we largely follow the suggestions
of Singh et al. (2024).
33
Llama 3
8B 70B 405B
QuALITY (5-shot) 56.0±2.1 82.8±1.6 87.6±1.4
GSM8K (16-shot) 60.0±9.6 83.0±7.4 90.0±5.9
Table 14 Performance of pre-trained models on long-context
tasks. Results include 95% conﬁdence intervals.
Contam. Performance gain est.
8B 70B 405B
AGIEval 98 8.5 19.9 16.3
BIG-Bench Hard 95 26.0 36.0 41.0
BoolQ 96 4.0 4.7 3.9
CommonSenseQA 30 0.1 0.8 0.6
DROP – – – –
GSM8K 41 0.0 0.1 1.3
HellaSwag 85 14.8 14.8 14.3
HumanEval – – – –
MATH 1 0.0 -0.1 -0.2
MBPP – – – –
MMLU – – – –
MMLU-Pro – – – –
NaturalQuestions 52 1.6 0.9 0.8
OpenBookQA 21 3.0 3.3 2.6
PiQA 55 8.5 7.9 8.1
QuaC 99 2.4 11.0 6.4
RACE – – – –
SiQA 63 2.0 2.3 2.6
SQuAD 0 0.0 0.0 0.0
Winogrande 6 -0.1 -0.1 -0.2
WorldSense 73 -3.1 -0.4 3.9
Table 15 Percentage of evaluation sets considered to be con-
taminated because similar data exists in the training corpus,
and the estimated performance gain that may result from
that contamination. See the text for details.
Method. Speciﬁcally, Singh et al. (2024) propose to
select contamination detection methods empirically,
based on which method results in the largest dif-
ference between the ‘clean’ part of the dataset and
the entire dataset, which they callestimated per-
formance gain. For all our evaluation datasets, we
score examples based on 8-gram overlap, a method
that was found by Singh et al. (2024) to be accurate
for many datasets. We consider an example of a
dataset D to be contaminated if a ratioTD of its
tokens are part of an 8-gram occurring at least once
in the pre-training corpus. We selectTD separately
for each dataset, based on which value shows the
maximal signiﬁcant estimated performance gain
across the three model sizes.
Results. In Table 15, we report the percentage of
evaluation data that is considered contaminated
for the maximal estimated performance gain, as
described above, for all key benchmarks. From
the table, we exclude numbers for benchmarks for
which the results are not signiﬁcant, for instance
because the clean or contaminated set has too few
examples, or because the observed performance
gain estimate shows extremely erratic behavior. In
Table 15, we observe that for some datasets con-
tamination has a large impact, while for others it
does not. For example, for PiQA and HellaSwag,
both the estimation of contamination and the esti-
mation of performance gain are high. For Natural
Questions, on the other hand, the estimated 52%
contamination seems to have virtually no eﬀect
on the performance. For SQuAD and MATH, low
thresholds yield high levels of contamination, but
no performance gains. This suggests that contam-
ination is either not helpful for these datasets, or
that a larger n is required to obtain a better es-
timate. Finally, for MBPP, HumanEval, MMLU
and MMLU-Pro, other contamination detection methods may be needed: even with higher thresholds, 8-gram
overlap gives such high contamination scores that it is impossible to get a good performance gain estimate.
5.2 Post-trained Language Model
We present results for our Llama 3 post-trained models on benchmarks across diﬀerent capabilities. Similar to
pre-training we are releasing the data generated as part of evaluations with publicly available benchmarks
which can be found on Huggingface here. Additional details on our eval setup can be found here.
Benchmarks and metrics. Table 16 contains an overview of all the benchmarks, organized by the capability.
We apply decontamination of the post-training data by running exact match with the prompts from each
benchmark. In addition to the standard academic benchmarks, we also performed extensive human evaluation
of diﬀerent capabilities. Details are provided in Section 5.3.
Experimental setup. We employ a similar experimental setup to the pre-training phase and conduct a
comparative analysis of Llama 3 alongside other models of comparable size and capability. To the extent
possible, we evaluate the performance of other models ourselves and compare the results with the reported
numbers, selecting the best score. You can ﬁnd additional details on our evaluation setup here.
34
General MMLU (Hendrycks et al., 2021a), MMLU-Pro (Wang et al., 2024b),
IFEval (Zhou et al., 2023)
Math and reasoning GSM8K (Cobbe et al., 2021), MATH (Hendrycks et al., 2021b),
GPQA (Rein et al., 2023), ARC-Challenge (Clark et al., 2018)
Code
HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021),
HumanEval+ (Liu et al., 2024a), MBPP EvalPlus (base) (Liu et al., 2024a),
MultiPL-E (Cassano et al., 2023)
Multilinguality MGSM (Shi et al., 2022), Multilingual MMLU (internal benchmark)
Tool-use Nexus (Srinivasan et al., 2023), API-Bank (Li et al., 2023b),
API-Bench (Patil et al., 2023), BFCL (Yan et al., 2024)
Long context ZeroSCROLLS (Shaham et al., 2023), Needle-in-a-Haystack (Kamradt, 2023),
InﬁniteBench (Zhang et al., 2024)
Table 16 Post-training benchmarks by category. Overview of all benchmarks we use to evaluate post-trained Llama 3
models, ordered by capability.
5.2.1 General Knowledge and Instruction-Following Benchmarks
We evaluate Llama 3 on benchmarks for general knowledge and instruction-following in Table 2.
General knowledge. We leverage MMLU (Hendrycks et al., 2021a) and MMLU-Pro (Wang et al., 2024b) to
evaluate Llama 3’s capability on knowledge-based question answering. For MMLU, we report the macro
average of subtask accuracy under the 5-shot standard setting without CoT. MMLU-Pro is an extension
of MMLU, incorporating more challenging, reasoning-focused questions, eliminating noisy questions, and
expanding the choice set from four to ten options. Given its focus on complex reasoning, we report 5-shot
CoT for MMLU-Pro. All tasks are formatted as generation tasks, similar to simple-evals (OpenAI, 2024).
As shown in Table 2, our 8B and 70B Llama 3 variants outperform other models of similar sizes on both
general knowledge tasks. Our 405B model outperforms GPT-4 and Nemotron 4 340B, with Claude 3.5 Sonnet
leading among larger models.
Instruction following. We assess the ability of Llama 3 and other models to follow natural language instructions
on IFEval (Zhou et al., 2023). IFEval comprises approximately 500 “veriﬁable instructions” such as “write
in more than 400 words”, which can be veriﬁed by heuristics. We report the average of prompt-level and
instruction-level accuracy, under strict and loose constraints in Table 2. Note that all Llama 3 variants
outperform comparable models across IFEval.
5.2.2 Proficiency Exams
Next, we evaluate our models on a wide variety of proﬁciency exams originally designed to test humans. We
source these exams from publicly available oﬃcial sources; for some exams, we report average scores across
diﬀerent exam sets per proﬁciency exam. Speciﬁcally, we average:
• GRE: Oﬃcial GRE Practice Test 1 and 2 (from the Educational Testing Services);
• LSAT: Oﬃcial Preptest 71, 73, 80 and 93;
• SAT: 8 exams from The Oﬃcial SAT Study guide edition 2018;
• AP: One oﬃcial practice exam per subject;
• GMAT Oﬃcial GMAT Online Exam.
Questions in these exams contain both MCQ style and generation questions. We exclude the questions that
are accompanied with images. For the GRE exams that contain questions with multiple correct options, we
qualify the outputs as correct only if all the correct options are selected by the model. The evaluations are
35
Exam
Llama 3 8B
Llama 3 70B
Llama 3 405B
GPT-3.5 Turbo
Nemotron 4 340B
GPT-4o
Claude 3.5 Sonnet
LSAT 53.9±4.9 74.2±4.3 81.1±3.8 54.3±4.9 73.7±4.3 77.4±4.1 80.0±3.9
SAT Reading 57.4±4.2 71.4±3.9 74.8±3.7 61.3±4.2 – 82.1±3.3 85.1±3.1
SAT Math 73.3±4.6 91.9±2.8 94.9±2.3 77.3±4.4 – 95.5±2.2 95.8±2.1
GMAT Quant. 56.0±19.5 84.0±14.4 96.0±7.7 36.0±18.8 76.0±16.7 92.0±10.6 92.0±10.6
GMAT Verbal 65.7±11.4 85.1±8.5 86.6±8.2 65.7±11.4 91.0±6.8 95.5±5.0 92.5±6.3
GRE Physics 48.0±11.3 74.7±9.8 80.0±9.1 50.7±11.3 – 89.3±7.0 90.7±6.6
AP Art History 75.6±12.6 84.4±10.6 86.7±9.9 68.9±13.5 71.1±13.2 80.0±11.7 77.8±12.1
AP Biology 91.7±11.1 100.0±0.0 100.0±0.0 91.7±11.1 95.8±8.0 100.0±0.0 100.0±0.0
AP Calculus 57.1±16.4 54.3±16.5 88.6±10.5 62.9±16.0 68.6±15.4 91.4±9.3 88.6±10.5
AP Chemistry 59.4±17.0 96.9±6.0 90.6±10.1 62.5±16.8 68.8±16.1 93.8±8.4 96.9±6.0
AP English Lang. 69.8±12.4 90.6±7.9 94.3±6.2 77.4±11.3 88.7±8.5 98.1±3.7 90.6±7.9
AP English Lit. 59.3±13.1 79.6±10.7 83.3±9.9 53.7±13.3 88.9±8.4 88.9±8.4 85.2±9.5
AP Env. Sci. 73.9±12.7 89.1±9.0 93.5±7.1 73.9±12.7 73.9±12.7 89.1±9.0 84.8±10.4
AP Macro Eco. 72.4±11.5 98.3±3.3 98.3±3.3 67.2±12.1 91.4±7.2 96.5±4.7 94.8±5.7
AP Micro Eco. 70.8±12.9 91.7±7.8 93.8±6.8 64.6±13.5 89.6±8.6 97.9±4.0 97.9±4.0
AP Physics 57.1±25.9 78.6±21.5 92.9±13.5 35.7±25.1 71.4±23.7 71.4±23.7 78.6±21.5
AP Psychology 94.8±4.4 100.0±0.0 100.0±0.0 94.8±4.4 100.0±0.0 100.0±0.0 100.0±0.0
AP Statistics 66.7±17.8 59.3±18.5 85.2±13.4 48.1±18.8 77.8±15.7 92.6±9.9 96.3±7.1
AP US Gov. 90.2±9.1 97.6±4.7 97.6±4.7 78.0±12.7 78.0±12.7 100.0±0.0 100.0±0.0
AP US History 78.0±12.7 97.6±4.7 97.6±4.7 85.4±10.8 70.7±13.9 95.1±6.6 95.1±6.6
AP World History 94.1±7.9 100.0±0.0 100.0±0.0 88.2±10.8 85.3±11.9 100.0±0.0 97.1±5.7
AP Average 74.1±3.4 87.9±2.5 93.5±1.9 70.2±3.5 81.3±3.0 93.0±2.0 92.2±2.1
GRE Quant. 152.0 158.0 162.0 155.0 161.0 166.0 164.0
GRE Verbal 149.0 166.0 166.0 154.0 162.0 167.0 167.0
Table 17 Performance of Llama 3 models and GPT-4o on a variety of proficiency exams including LSAT, SAT, GMAT, and
AP, and GRE tests. For GRE exams, we report normalized score; for all others, we report accuracy. For the bottom
two rows corresponding to GRE Quant. and GRE Verbal, we report the scaled scores out of 170.
run using few shot prompting wherever we have more than 1 exam set per exam. We scale the scores to be in
the range 130-170 for GRE and report accuracy for all other exams.
Our results can be found in Table 17. We observe that the performance of our Llama 3 405B model is very
similar to Claude 3.5 Sonnet and GPT-4 4o. Our 70B model has an even more impressive performance. It is
signiﬁcantly better than GPT-3.5 Turbo and beats Nemotron 4 340B on many tests.
5.2.3 Coding Benchmarks
We evaluate Llama 3 on code generation on several popular Python and multi-programming language
benchmarks. To gauge the eﬀectiveness of our models in generating functionally correct code, we use the
pass@N metric, which evaluates the pass rate for a set of unit tests amongN generations. We report pass@1.
Python code generation. HumanEval(Chenetal.,2021)andMBPP(Austinetal.,2021)arepopularbenchmarks
for Python code generation which focus on relatively simple, self-contained functions. HumanEval+ (Liu et al.,
2024a) is an enhanced version of HumanEval, in which more tests are generated to avoid false positives. The
MBPP EvalPlus base version (v0.2.0) is a selection of 378 well-formed problems out of the 974 initial problems
in all of the original MBPP (train and test) dataset (Liu et al., 2024a). Results for these benchmarks are
reported in Table 18. Across the Python variants of these benchmarks, Llama 3 8B and 70B outperform
36
Model HumanEval HumanEval+ MBPP MBPP
EvalPlus (base)
Llama 3 8B 72.6±6.8 67.1±7.2 60.8±4.3 72.8±4.5
Gemma 2 9B 54.3±7.6 48.8±7.7 59.2±4.3 71.7±4.5
Mistral 7B 40.2±7.5 32.3±7.2 42.6±4.3 49.5±5.0
Llama 3 70B 80.5±6.1 74.4±6.7 75.4±3.8 86.0±3.5
Mixtral 8×22B 75.6±6.6 68.3±7.1 66.2±4.1 78.6±4.1
GPT-3.5 Turbo 68.0±7.1 62.8±7.4 71.2±4.0 82.0±3.9
Llama 3 405B 89.0±4.8 82.3±5.8 78.8±3.6 88.6±3.2
GPT-4 86.6±5.2 77.4±6.4 80.2±3.5 83.6±3.7
GPT-4o 90.2±4.5 86.0±5.3 81.4±3.4 87.8±3.3
Claude 3.5 Sonnet 92.0±4.2 82.3±5.8 76.6±3.7 90.5±3.0
Nemotron 4 340B 73.2±6.8 64.0±7.3 75.4±3.8 72.8±4.5
Table 18 Pass@1 scores on code generation benchmarks. We report results on HumanEval (Chen et al., 2021),
MBPP (Austin et al., 2021), as well as EvalPlus (Liu et al., 2024a) versions of these benchmarks.
Model Dataset C++ Java PHP TS C# Shell
Llama 3 8B HumanEval 52.8 ±7.7 58.2±7.7 54.7±7.7 56.6±7.7 38.0±7.6 39.2±7.6
MBPP 53.7 ±4.9 54.4±5.0 55.7±4.9 62.8±4.8 43.3±4.9 33.0±4.7
Llama 3 70B HumanEval 71.4 ±7.0 72.2±7.0 67.7±7.2 73.0±6.9 50.0±7.8 51.9±7.8
MBPP 65.2 ±4.7 65.3±4.8 64.0±4.7 70.5±4.5 51.0±5.0 41.9±4.9
Llama 3 405B HumanEval 82.0 ±5.9 80.4±6.2 76.4±6.6 81.1±6.1 54.4±7.8 57.6±7.7
MBPP 67.5 ±4.6 65.8±4.7 76.6±4.2 72.6±4.4 53.1±5.0 43.7±5.0
Table 19 Performance of non-Python programming tasks. We report Llama 3 results on MultiPL-E (Cassano et al., 2023).
models of similar sizes. For the largest models, Llama 3 405B, Claude 3.5 Sonnet and GPT-4o perform
similarly, with GPT-4o showing the strongest results.
Multi-programming language code generation. To assess code generation capabilities beyond Python, we report
results for the MultiPL-E (Cassano et al., 2023) benchmark, which is based on translations of problems from
HumanEval and MBPP. Results for a subset of popular programming languages are reported in Table 19.
Note that there is a signiﬁcant drop in performance compared to the Python counterparts in Table 18.
5.2.4 Multilingual Benchmarks
Llama 3 supports 8 languages — English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai,
although the underlying foundation model has been trained on a broader collection of languages.9 In Table 20,
we show results from evaluating Llama 3 on the multilingual MMLU (Hendrycks et al., 2021a) and Multilingual
Grade School Math (MGSM) (Shi et al., 2022) benchmarks.
Multilingual MMLU. We translate MMLU questions, few-shot examples, and answers using Google Translate.
We leave the task instructions in English and perform the evaluation in a 5-shot setting. In Table 20, we
report average results across German, French, Italian, Portuguese, Hindi, Spanish, and Thai.
9Llama 3 has not been optimized or safety tuned for use cases in those other languages. Developers may ﬁne-tune Llama 3
models for languages beyond the 8 supported languages provided they comply with the Llama 3 Community License and the
Acceptable Use Policy and in such cases are responsible for ensuring that any uses of Llama 3 in additional languages is done in a
safe and responsible manner.
37
Model MGSM Multilingual MMLU
Llama 3 8B 68.9 58.6
Mistral 7B 29.9 46.8
Gemma 2 9B 53.2 –
Llama 3 70B 86.9 78.2
GPT-3.5 Turbo 51.4 58.8
Mixtral 8×22B 71.1 64.3
Llama 3 405B 91.6 83.2
GPT-4 85.9 80.2
GPT-4o 90.5 85.5
Claude 3.5 Sonnet 91.6 –
Table 20 Multilingual benchmarks . For MGSM (Shi et al.,
2022), we report 0-shot CoT results for our Llama 3
models. Multilingual MMLU is an internal benchmark
with translated MMLU (Hendrycks et al., 2021a) ques-
tions and answers into 7 languages – we report 5-shot
results averaged across these languages.
MGSM (Shi et al., 2022). We use the same native
prompts as in simple-evals (OpenAI, 2024) for testing
our models in a 0-shot CoT setting. In Table 20,
we report averge results across languages covered in
MGSM benchmark.
We ﬁnd that Llama 3 405B outperforms most other
models on MGSM, achieving an average of 91.6%. On
MMLU, in line with English MMLU results shown
above, Llama 3 405B falls behind GPT-4o by 2%.
On the other hand, both Llama 3 70B and 8B mod-
els demonstrate strong performance, leading among
competitors with a wide margin on both tasks.
5.2.5 Math and Reasoning Benchmarks
Our math and reasoning benchmark results are pre-
sented in Table 2. Llama 3 8B model outperforms
other models of similar sizes on GSM8K, MATH, and
GPQA. Our 70B model performs signiﬁcantly better
than other models in its class on all the benchmarks.
Finally, Llama 3 405B model is the best in its category
on GSM8K and ARC-C, while on MATH, it is the second best model. On GPQA, it is competitive with
GPT-4 4o, with Claude 3.5 Sonnet being the best model by a signiﬁcant margin.
5.2.6 Long Context Benchmarks
We consider a diverse set of tasks that span various domains and text types. In the benchmarks we list below,
we focus on sub-tasks that use unbiased evaluation protocols, i.e., accuracy-based metrics rather than n-gram
overlapping metrics. We also prioritize tasks that we found to be of low variance.
• Needle-in-a-Haystack (Kamradt, 2023) measures a model’s ability to retrieve a hidden information
inserted in random parts of the long document. Our Llama 3 models demonstrate perfect needle retrieval
performance, successfully retrieving 100% of needles at all document depths and context lengths. We
also measure performance on Multi-needle (Table 21), a variation of Needle-in-a-Haystack, where we
insert four needles in the context and test if a model can retrieve two of them. Our Llama 3 models
achieve near perfect retrieval results.
• ZeroSCROLLS (Shaham et al., 2023) is a zero-shot benchmark for natural language understanding over
long texts. We report numbers on the validation set, as the ground truth answers are not publicly
available. Our Llama 3 405B and 70B models either match or surpass other models on various tasks in
this benchmark.
• InfiniteBench (Zhang et al., 2024) requires models to understand long dependencies in the context
window. We evaluate Llama 3 on En.QA (QA over novels) and En.MC (multiple-choice QA over novels),
where our 405B model outperforms all others. The gains are particularly signiﬁcant on En.QA.
5.2.7 Tool Use Performance
We evaluate our models on a range of benchmarks for zero-shot tool use (i.e. function calling): Nexus (Srini-
vasan et al., 2023), API-Bank (Li et al., 2023b), Gorilla API-Bench (Patil et al., 2023), and the Berkeley
Function Calling Leaderboard (BFCL) (Yan et al., 2024). Results are shown in Table 22.
On Nexus, our Llama 3 variants perform the best compared to their counterparts. On the API-Bank, our
Llama 3 8B and 70B models outperform other models in their category by a signiﬁcant margin. The 405B
model is behind Claude 3.5 Sonnet by only 0.6%. Finally, our 405B and 70B models perform competitively on
BFCL and are close second in their respective size class. Llama 3 8B performs the best in its category.
38
ZeroSCROLLS InfiniteBench NIH
QuALITY Qasper SQuALITY En.QA En.MC Multi-needle
Llama 3 8B 81.0±16.8 39.3±18.1 15.3±7.9 27.1±4.6 65.1±6.2 98.8±1.2
Llama 3 70B 90.5±12.6 49.0±18.5 16.4±8.1 36.7±5.0 78.2±5.4 97.5±1.7
Llama 3 405B 95.2±9.1 49.8±18.5 15.4±7.9 30.5±4.8 83.4±4.8 98.1±1.5
GPT-4 95.2±9.1 50.5±18.5 13.2±7.4 15.7±3.8 72.0±5.8 100.0±0.0
GPT-4o 90.5±12.5 49.2±18.5 18.8±8.6 19.1±4.1 82.5±4.9 100.0±0.0
Claude 3.5 Sonnet 90.5±12.6 18.5±14.4 13.4±7.5 11.3±3.3 – 90.8±3.2
Table 21 Long-context benchmarks. For ZeroSCROLLS (Shaham et al., 2023), we report numbers on the validation set.
For QuALITY we report exact match, for Qasper - f1 and for SQuALITY - rougeL. We report f1 for InﬁniteBench
(Zhang et al., 2024) En.QA metric and accuracy for En.MC. For Multi-needle (Kamradt, 2023) we insert 4 needles in
the context and test if a model can retrieve 2 needles at diﬀerent context lengths, we compute average recall across 10
sequence lengths up till 128k.
Human evaluations. We also conduct human evaluations to test the tool use capabilities of the model, with a
focus on code execution tasks. We collect 2000 user prompts related to code execution (without plotting or
ﬁle uploads), plot generation, and ﬁle uploads. These prompts are collected from the LMSys dataset (Chiang
et al., 2024), GAIA benchmark (Mialon et al., 2023b), human annotators, and synthetic generation.
Nexus API-Bank API-Bench BFCL
Llama 3 8B 38.5±4.1 82.6±3.8 8.2±1.3 76.1±2.0
Gemma 2 9B – 56.5±4.9 11.6±1.5 –
Mistral 7B 24.7±3.6 55.8±4.9 4.7±1.0 60.4±2.3
Llama 3 70B 56.7±4.2 90.0±3.0 29.7±2.1 84.8±1.7
Mixtral 8×22B 48.5±4.2 73.1±4.4 26.0±2.0 –
GPT-3.5 Turbo 37.2±4.1 60.9±4.8 36.3±2.2 85.9±1.7
Llama 3 405B 58.7±4.1 92.3±2.6 35.3±2.2 88.5±1.5
GPT-4 50.3±4.2 89.0±3.1 22.5±1.9 88.3±1.5
GPT-4o 56.1±4.2 91.3±2.8 41.4±2.3 80.5±1.9
Claude 3.5 Sonnet 45.7±4.2 92.6±2.6 60.0±2.3 90.2±1.4
Nemotron 4 340B – – – 86.5±1.6
g
Table 22 Zero-shot tool use benchmarks. We report function calling accuracy
across Nexus (Srinivasan et al., 2023), API-Bank (Li et al., 2023b), API-
Bench (Patil et al., 2023), and BFCL (Yan et al., 2024).
We compare Llama 3 405B to
GPT-4o using OpenAI’s Assis-
tants API10. The results are pro-
vided in Figure 16. On text-only
code execution tasks and plots gen-
eration, Llama 3 405B signiﬁcantly
beats GPT-4o. However, it lags
behind on the ﬁle upload use case.
5.3 Human Evaluations
In addition to evaluations on stan-
dard benchmark sets, we also per-
form a series of human evaluations.
These evaluations allow us to mea-
sure and optimize more subtle as-
pects of model performance, such
as our model’s tone, verbosity, and
understanding of nuances and cul-
tural contexts. Well-designed hu-
man evaluations closely reﬂect the
user experience, providing insights
into how the model performs in real-world scenarios.
Prompt collection. We collected high-quality prompt spanning a wide range of categories and diﬃculties. To do
so, we ﬁrst developed a taxonomy with categories and subcategories capturing as many model capabilities as
possible. We used this taxonomy to collect about7, 000 prompts spanning six individual capabilities (English,
reasoning, coding, Hindi, Spanish, and Portuguese), and three multiturn capabilities11 (English, reasoning,
and coding). We ensured that within each category, prompts are uniformly distributed across subcategories.
We also categorized each prompt into one of three diﬃculty levels and ensured that our prompt collection
10https://platform.openai.com/docs/assistants/overview
11For multiturn human evaluations, the number of turns is between 2 and 11 in each prompt. We assess the model response in
the ﬁnal turn.
39
Figure 16 Human evaluation results for Llama 3 405B vs. GPT-4o on code execution tasks including plotting and file uploads.
Llama 3 405B outperforms GPT-4o on code execution (without plotting or ﬁle uploads) as well as plot generation, but
lags behind in ﬁle upload use cases.
contains roughly 10% easy prompts, 30% medium prompts, and60% hard prompts. All the human evaluation
prompt sets were subject to a thorough quality assurance process. Modeling teams did not have access to our
human-evaluation prompts to prevent accidental contamination or overﬁtting on the test set.
Evaluation process. To perform a pairwise human evaluation of two models, we ask human annotators which
of two model responses (produced by diﬀerent models) they prefer. Annotators use a 7-point scale for their
ratings, enabling them to indicate whether one model response is much better than, better than, slightly
better than, or about the same as the other model response. When an annotator indicates that one model
response is better or much better than the other model response, we consider this a “win” for that model. We
perform pairwise comparisons between models in which we report win rates per capability in the prompt set.
Results. We use our human evaluation process to compare Llama 3 405B with GPT-4 (0125 API version),
GPT-4o (API version), and Claude 3.5 Sonnet (API version). The results of these evaluations are presented
in Figure 17. We observe that Llama 3 405B performs approximately on par with the 0125 API version of
GPT-4, while achieving mixed results (some wins and some losses) compared to GPT-4o and Claude 3.5
Sonnet. On nearly all capabilities, the win rates of Llama 3 and GPT-4 are within the margin of error. On
multiturn reasoning and coding tasks, Llama 3 405B outperforms GPT-4 but it underperforms GPT-4 on
multilingual (Hindi, Spanish, and Portuguese) prompts. Llama 3 performs on par with GPT-4o on English
prompts, on par with Claude 3.5 Sonnet on multilingual prompts, and outperforms Claude 3.5 Sonnet on
single and multiturn English prompts. However, it trails Claude 3.5 Sonnet in capabilities such as coding
and reasoning. Qualitatively, we ﬁnd that model performance in human evaluations is heavily inﬂuenced by
nuanced factors such as model tone, response structure, and verbosity – factors that we are optimizing for
in our post-training process. Overall, our human evaluation results are consistent with those on standard
benchmark evaluations: Llama 3 405B is very competitive with leading industry models, making it the
best-performing openly available model.
Limitations. All human evaluation results underwent a thorough data quality assurance process. However,
since it is challenging to deﬁne objective criteria for evaluating model responses, human evaluations can still
be inﬂuenced by personal biases, backgrounds, and preferences of human annotators, which may lead to
inconsistent or unreliable results.
5.4 Safety
We focus our study on assessing Llama 3’s ability to generate content in a safe and responsible way, while still
maximizing helpful information. Our safety work begins in the pre-training stage, primarily in the form of
40


*(Extracción truncada en la página 40 de 92 totales)*