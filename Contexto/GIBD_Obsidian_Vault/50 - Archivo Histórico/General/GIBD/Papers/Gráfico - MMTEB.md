---
aliases: [Gráfico - MMTEB]
tags:
  - grupo/general
  - estado/archivo
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2025-07-01
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Papers/Índices/Gráfico - MMTEB.pdf"
tamanio_bytes: 1375509
---

# Gráfico - MMTEB

Ruta interna: `GIBD/Papers/Índices/Gráfico - MMTEB.pdf`

---

Primer on Evaluation Embeddings
MMTEB — Massive Multilingual Text Embedding BenchmarkKenneth Enevoldsen, Isaac Chung, Imene Kerboua, Márton Kardos, Ashwin Mathur, David Stap, Jay Gala, Wissam Siblini, Dominik Krzemiński, Genta Indra Winata, Saba Sturua, Saiteja Utpala, Mathieu Ciancone, Marion Schaeffer, Gabriel Sequeira, Diganta Misra, Shreeya Dhakal, Jonathan Rystrøm, Roman Solomatin, Ömer Çağatan, Akash 
Kundu, Martin Bernstorff, Shitao Xiao, Akshita Sukhlecha, Bhavish Pahwa, Rafał Poświata, Kranthi Kiran GV, Shawon Ashraf, Daniel Auras, Björn Plüster, Jan Philipp Harries, Loïc Magne, Isabelle Mohr, Mariya Hendriksen, Dawei Zhu, Hippolyte Gisserot-Boukhlef, Tom Aarsen, Jan Kostkan, Konrad Wojtasik, Taemin Lee, Marek Šuppa, Crystina 
Zhang, Roberta Rocca, Mohammed Hamdy, Andrianos Michail, John Yang, Manuel Faysse, Aleksei Vatolin, Nandan Thakur, Manan Dey, Dipam Vasani, Pranjal Chitale, Simone Tedeschi, Nguyen Tai, Artem Snegirev, Michael Günther, Mengzhou Xia, Weijia Shi, Xing Han Lù, Jordan Clive, Gayatri Krishnakumar, Anna Maksimova, Silvan Wehrli, 
Maria Tikhonova, Henil Panchal, Aleksandr Abramov, Malte Ostendorff, Zheng Liu, Simon Clematide, Lester James Miranda, Alena Fenogenova, Guangyu Song, Ruqiya Bin Safi, Wen-Ding Li, Alessia Borghini, Federico Cassano, Hongjin Su, Jimmy Lin, Howard Yen, Lasse Hansen, Sara Hooker, Chenghao Xiao, Vaibhav Adlakha, Orion Weller, 
Siva Reddy, Niklas Muennighoff
OVERVIEW &  
RATIONALEEVALUATING EMBEDDINGS 
Want to see more 
performance scores? 
Check out the public 
leaderboard with >200 
models
ICLR
OPTIMIZATIONS
Target to 
label
Nearest neighbours
Classification: Samples from each class is 
embeded documents and a classifer is fit 
and evaluated.
Given label
Determined Clusters
Clustering: We embed and cluster the 
documents and calculate agreement using a v-
measure.
Want a similar benchmark for image and 
text embeddings? 
Great news! We will soon release an 
upcoming work on images. See the results 
Rationale 
Text embeddings are often evaluated on a limited set of tasks, 
constrained by language, domain, and task diversity 
Text embeddings play an essential role in LLM inference, including 
RAG and few-shot classification, data curation, and more 
Contributions 
MMTEB covers >500 quality-controlled evaluation tasks across >250 
languages, making it the largest multilingual benchmark 
MMTEB introduces a diverse set of tasks, including instruction 
following, long document retrieval, code retrieval, and more 
We reveal a notable performance gap appearing already among 
mid-resource languages such as German and Polish 
Significantly speed up evaluation using only 2% of previous 
benchmark documents for comparable benchmarks
Task Curation 
All task in MMTEB was collected through an open-
science eﬀort using a point-system to determine co-
authorship. 
Each task has extensive metadata on annotation 
source, dataset source, license, dialect, citation 
information, etc. 
And was reviewed along various axes, including 
checking for performance ceilings, implementation 
bugs, and the ability to discriminate between models.
DATA CURATION
LANGUAGE GAP
Language Gap 
Multilingual 7b models 
outperformed by notably smaller 
models (560M) in low-resource 
settings. This is likely due to pre-
training of the base model. 
In truly low-resource settings, the 
smaller XLM-R-based multilingual-e5-
large-instruct consistently 
outperforms larger models. 
This suggests the need for better 
multilingual base models as XLM-R-
based models still outperform Mistral 
or Llama-based encoders.
Want know more? 
Check out the paper!
MULTILINGUAL 
EVALUATION
Results and Findings 
Models trained with instruction-
tuning perform significantly better 
compared to those without. The two 
large multilingual e5 models are a clear 
example of this. 
Discrepancies in multilingual 
benchmarks stem from diﬀerences in 
the pre-training. This suggests that 
multilingual pre-training of the base 
models will likely lead to performance 
gains.
Speed Optimizations 
To keep the benchmark accessible for low-
resource communities, we optimize  by: 
Encouraging smaller dataset submission, typically 
~2048 samples is enough to diﬀerentiate between 
models 
For clustering tasks, we used bootstrapping-based 
downsampling to reduce the number of 
documents by ~16x 
For retrieval, we use hard-negative mining across 
diverse models, keeping the top 250 ranked 
documents pr. query 
We perform task downsampling to remove highly 
correlated tasks while maintaining benchmark 
sensitivity
Largest 
Multilingual 
Benchmark for 
Embeddings
Encoder
“Is Creole a pidgin of 
French?”
“When did Marxism 
develop?”
…
MiraclRetrieval
Query
Corpus 
documents
Top-k Retrieved 
Candidates
Retrieval:  Queries and documents are 
embedded. For each query, the documents are 
ranked and performance estimated.
1) Select a task
2) Encode texts
3) Evaluate
Same  or better 
performance 
estimate using 
50x fewer 
documents
Highly multilingual 
pre-training 
compensates for size
>500 tasks 
>250 Languages
