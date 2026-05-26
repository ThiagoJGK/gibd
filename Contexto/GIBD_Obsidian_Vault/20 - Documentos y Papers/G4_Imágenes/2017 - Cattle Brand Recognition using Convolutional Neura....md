---
aliases: [2017 - Cattle Brand Recognition using Convolutional Neura...]
tags:
  - grupo/g4_imágenes
  - estado/util
  - tipo/documento
formato_original: pdf
fecha_modificacion: 2022-08-16
origen_zip: GIBD-20260521T205218Z-3-001.zip
ruta_interna: "GIBD/Trabajos parecidos/Brasileros/2017 - Cattle Brand Recognition using Convolutional Neural Network and Support Vector Machines.pdf"
tamanio_bytes: 810801
---

# 2017 - Cattle Brand Recognition using Convolutional Neural Network and Support Vector Machines

Ruta interna: `GIBD/Trabajos parecidos/Brasileros/2017 - Cattle Brand Recognition using Convolutional Neural Network and Support Vector Machines.pdf`

---

1Abstract— The recognition images of cattle brand in an 
automatic way is a necessity to governmental organs responsible  
for this activity. To help this process, this work presents a 
method that consists in using Convolutional Neural Network for 
extracting of characteristics fr om images of cattle brand and 
Support Vector Machines for classification. This method consist s 
of six stages: a) select database of images; b) select pre-trai ned 
CNN; c) pre-process the images and apply CNN; d) extract 
images of features; e) train and sort images (SVM); f) evaluate  
the classification results. The accuracy of the method was test ed 
on database of municipal city hall, where it achieved satisfact ory 
results, comparable to other methods from the literature, 
reporting 93.28% of accuracy and 12.716 seconds of processing 
time, respectively.  
 
Keywords— Recognition Images, Cattle Brands, 
Convolutional Neural Network, Support Vector Machines, Deep 
Learning. 
 
 
I. INTRODUÇÃO 
 
 DESENVOLVIMENTO de ferramentas computacionais 
para auxiliar a análise e reconhecimento de imagens é 
alvo de interesse dos mais renomados centros de pesquisa do 
mundo. O uso da computação para análise e reconhecimento 
de imagens está em constante desenvolvimento, gerando 
vários benefícios à sociedade nas mais diversificadas áreas do 
conhecimento. No que se refere particularmente ao 
reconhecimento de marcas de gado, atividade esta tão 
tradicional e de grande relevância socioeconômica para países 
da América Latina, incluindo o Brasil, não há um método 
específico e devidamente consolidado para este fim. Para se 
ter uma dimensão da importância desta atividade, de acordo 
com a Food and Agriculture Organization  – FAO, entre os 
países produtores, o Brasil e a Índia possuem os maiores 
rebanhos, sendo que o Brasil encontra-se em 1º lugar, com 
uma média de 209.215.666 cabeças [1]. 
A produção pecuária possui um papel relevante na 
formação social, e que ainda hoje se mantém como uma 
atividade de grande importância nas expressões culturais 
associadas a ela, uma vez que é  associada à cultura e ao modo 
de vida do campo, além do papel na afirmação ou construção 
                                                            
1C. Silva, Universidade Federal do Pampa (UNIPAMPA), Alegrete, R io 
Grande do Sul, Brasil, carlossilva@ieee.org 
1D. Welfer, Universidade Federal de Santa Maria (UFSM), Departam ento 
de Computação Aplicada – DCOM, Santa Maria, Rio Grande do Sul, Brasil, 
daniel.welfer@ufsm.br 
1F. P. Gioda, Prefeitura Municipal de São Francisco de Assis, Rio Grande 
do Sul, Brasil, administracao@saofranciscodeassis.rs.gov.br 
1C .  D o r n e l l e s ,  P r e f e i t u r a  M u n i c i p a l  d e  S ã o  F r a n c i s c o  d e  A s s i s ,  Rio 
Grande do Sul, Brasil, administracao@saofranciscodeassis.rs.gov.br 
de identidades individuais ou de grupos [2].  
A utilização das marcas ou sinais no gado pressupõe o 
reconhecimento público de sua propriedade por um indivíduo 
ou grupo. Utilizadas desde o i nício da colonização ibérica na 
América, o início de sua institucionalização se deu a partir do  
registro em órgãos, sendo oficiais, reconhecidamente 
portadores de legitimação púb lica [2]. A esses registros 
seguem-se as regulamentações que procuram, além de 
oficializar a marcação em si, orientar a forma e o período a ser 
feita, discriminar a forma do registro, instituir valores a est e, a 
construção dos ferros e a taxação governamental. Em geral, os 
registros das marcas de gado constituem-se de livros com os 
desenhos das marcas e com a identificação de seu proprietário. 
No Brasil as tentativas e os investimentos de aprimoramento 
no sistema de registro de marcas de gado sempre foi alvo de 
polêmica, em função da resistência dos pecuaristas. Boa parte 
desse receio está associado ao temor de perda das marcas 
familiares e o significado que elas adquiriram com o tempo. 
Atualmente, os registros de ma rcas no Brasil são realizados 
nos municípios, em geral sem uma sistematização mais efetiva 
e sem a necessidade de serem renovadas. 
Perante o contexto apresentado, este trabalho tem como 
objetivo apresentar e avaliar uma ferramenta que realiza o 
reconhecimento automático de marcas de gado, com o 
propósito de substituir o controle manual de marcas de gado 
que é realizado atualmente, de forma a diminuir 
potencialmente a possibilidade de registros duplicados, reduzir 
o tempo de espera para registro de novas marcas, aprimorar a 
gestão governamental no que diz respeito ao acervo de marcas 
sob sua responsabilidade e auxiliar as autoridades de 
segurança no combate a crimes de abigeato.  
O trabalho concentrou-se no desenvolvimento dos 
algoritmos responsáveis pelo reconhecimento das marcas de 
gado da Prefeitura Municipal de São Francisco de Assis, Rio 
Grande do Sul, Brasil. Desta maneira, os funcionários do Setor 
de Registro de marcas de gado e do Centro de Processamento 
de Dados do município validaram a ferramenta proposta. 
O restante do trabalho está organizado da seguinte 
maneira: na Seção 2 são descritos os trabalhos relacionados. 
Na Seção 3 são apresentados os materiais e métodos utilizados 
para este trabalho. A Seção 4 descreverá os resultados e 
discussões obtidos através da aplicação dos métodos 
propostos. Por fim, na Seção 5 serão descritas as 
considerações finais. 
 
 
II.
 TRABALHOS RELACIONADOS 
 
De maneira geral, não foram encontrados trabalhos na 
revisão da literatura que reportem a utilização de redes neurais 
O 
Cattle Brand Recognition using Convolutional 
Neural Network and Support Vector Machines 
 
C. Silva, Member, IEEE, D. Welfer, F. P. Gioda and C. Dornelles 
310 IEEE LATIN AMERICA TRANSACTIONS, VOL. 15, NO. 2, FEB. 2017
convolucionais para reconhecimento de imagens de marcas de 
gado.  
Sanchez et al  [ 3 ]   a p r e s e n t a  u m a  f e r r a m e n t a  p a r a  
reconhecimento de marcas de gado utilizando momentos de 
Hu e Legendre para extração de características de imagens em 
escala de cinza, e um classi ficador de k-Vizinhos mais 
Próximos (k-NN). Os autores utilizaram momentos de Hu e 
Legendre com o propósito de extrair características que não 
fossem suscetíveis a transform ações de rotação, translação e 
escala. O percentual máximo de classificação correta 
apresentado pelos autores foi de 99,3 %, porém com uma 
diminuição significativa de acurácia à medida que aumentava 
o número de imagens classificadas. Outro resultado 
apresentado foi o tempo de processamento da classificação. 
Pelo fato de ter sido utilizado um classificador k-NN, a cada 
novo objeto que se quer classificar, utiliza-se os dados de 
treinamento para verificar quais são os objetos nesta base de 
dados que mais se assemelham ao novo objeto que se quer 
classificar. O objeto é classi ficado dentro da classe mais 
comum a que pertencem os objetos mais similares a ele. 
Assim, a classificação é feita por analogia. Nenhum modelo 
de classificação é criado. Ao invés disto, a cada novo objeto a  
ser classificado, os dados de treinamento são escaneados, 
dessa forma o classificador proposto se torna 
computacionalmente dispendioso.  
Diferentemente do trabalho apresentado por Sanchez et al 
[3], o trabalho aqui proposto pretende apresentar resultados 
que possam ser generalizados ou replicados, utilizando 
técnicas no estado da arte no que diz respeito à extração de 
características e classificação estatística de imagens digitais , 
tais como Redes Neurais Convolucionais (CNN) e Máquinas 
de Vetores de Suporte (SVM), objetivando desempenhos 
superiores no reconhecimento de imagens em bases de dados  
com número maior de registros, porém com custo 
computacional e tempo de processamento menores.  
Os trabalhos encontrados na literatura para reconhecimento 
e classificação de imagens que utilizam descritores ou filtros 
para extração de características, seguida de uma etapa de 
quantização e agrupamento e, por último, uma etapa de 
classificação são divididos em duas categorias: algoritmos de 
estágio de extração de características único e algoritmos de 
dois ou mais estágios [4].  
A possibilidade de treinar redes neurais com múltiplas 
camadas intermediárias desperta o surgimento de diversos 
algoritmos agrupados em um área conhecida como deep 
learning. 
O objetivo principal dos algoritmos que utilizam dois ou 
mais estágios é aprender não apenas a distinguir as classes 
com base em descritores artificiais, mas aprender os próprios 
descritores com base nos dados brutos, no caso de imagens, os 
próprios valores dos pixels [5].  
Redes neurais convolucionais vêm sendo utilizadas a vários 
anos no reconhecimento de imagens, tendo obtido grande 
sucesso no reconhecimento de caracteres no trabalho de Cun 
et al [6].  
Estudos mais recentes u tilizando redes neurais 
convolucionais conhecidas também como Deep Convolutional 
Neural Networks (CNN) obtiveram o novo estado da arte no 
reconhecimento de objetos em bases CIFAR-10 e NORB [7]. 
De forma geral as CNN são treinadas de forma 
supervisionada, mas trabalhos sugerem que o pré-treinamento 
da CNN com filtros obtidos de forma não supervisionada 
apresentam um melhor resultado [8]. 
No trabalho realizado por Sermanet et al [9] é apresentado 
um framework utilizando CNN para efetuar o reconhecimento, 
localização e detecção de imagens sendo o vencedor do 
campeonato ImageNet Large Scale Visual Recognition 
Challenge 2013 (ILSVRC2013). A base ILSVRC2013 
consiste em 1,2 milhões de imagens divididas em 1000 
categorias. 
Uma característica importante da CNN é a capacidade de 
serem reutilizadas e refinadas para diferentes bases de 
imagens. No trabalho realizado por Razavian et al  [10] foi 
utilizada uma CNN pré-treinada chamada Overfeat [ 9 ]  p a r a  
realizar extração de um descritor de diferentes bases de 
imagens na qual a CNN não foi originalmente treinada. Nesse 
caso, os descritores são então classificados utilizando um 
classificador linear SVM. Os resultados demonstram um 
desempenho compatível com o estado da arte, mesmo 
comparando com algoritmos que utilizam imagens 
segmentadas manualmente procedimento que não é necessário 
quando utilizada a CNN, e treinados especificamente na base 
analisada.  
A utilização de deep learning  também é descrita no 
trabalho de Constante et al [11], os quais utilizaram uma rede 
neuronal de três camadas com entrada usando 
backpropagation. Nesse trabalho, o método foi utilizado para 
realizar classificação de morangos, e chegou a resultados de 
reconhecimento de 92,5% na categoria “Extra”; 90% na 
categoria “Consumo”; 90% na categoria “Matéria prima”; e 
100% na categoria “Objetos estranhos”.  
 
III. MATERIAIS E MÉTODOS 
 
As imagens de marcas de gado utilizadas nesse trabalho 
foram fornecidas pela Prefeitura de São Francisco de Assis –
RS. Foram utilizadas 12 imagens de marca de gado, sendo 
cada uma dessas constituídas por 45 subimagens (amostras), 
totalizando 540 amostras oriundas das imagens originais, 
porém com variações de tamanho e orientação, com o objetivo 
de identificar padrões com a maior independência possível 
desses fatores. As imagens fo ram disponibilizadas em alta 
resolução no formato Portable Network Graphics  c o m  
tamanho 600 x 600 pixels.  
Para implementação da ferramenta proposta, assim como o 
armazenamento do banco de imagens, processamento dos 
algoritmos e visualização dos resultados foi utilizado um 
computador pessoal com placa de vídeo com suporte à 
plataforma de computação paralela CUDA com compute 
capability version  5.0. Além disso, foi utilizado o software 
MatLab com as bibliotecas Neural Network, Parallel 
Computing e Statistics and Machine Learning  e o modelo de 
rede convolucional pré-treinado obtido junto à biblioteca open 
source da VLFeat.org [12]. 
SILVA  et al.: CATTLE BRAND RECOGNITION USING 311
O método proposto é composto por seis etapas, que são: 
seleção do banco de imagens; seleção do modelo de CNN pré-
treinado; pré-processamento das imagens e aplicação da CNN; 
extração de características das imagens; treinamento e 
classificação das imagens por m eio de Máquinas de Vetores 
de Suporte; e, por fim, avaliação dos resultados da 
classificação. A Fig. 1 ilustra o fluxograma sumarizado do 
método proposto. 
 
 
 
Figura 1. Fluxograma sumarizado do método proposto. 
 
A.Seleção do banco de imagens 
As marcas utilizadas no trabalho são ilustradas na Fig. 2. O 
código das marcas, os proprietários e a quantidade de amostras 
de cada marca podem ser obtidos na Tabela I. 
 
 
Figura 2. Imagens de marcas de gado utilizadas no trabalho. 
 
 
TABELA I 
MARCAS, PROPRIETÁRIOS E TOTAL DE AMOSTRAS POR MARCA  
Marca Proprietário Total de 
amostras 
802 Proprietário “A” 45 
803 Proprietário “B” 45 
804 Proprietário “C” 45 
805 Proprietário “D” 45 
811 Proprietário “E” 45 
812 Proprietário “F” 45 
813 Proprietário “G” 45 
814 Proprietário “H” 45 
815 Proprietário “I” 45 
821 Proprietário “J” 45 
822 Proprietário “K” 45 
1093 Proprietário “L” 45 
 
 
B. Seleção do modelo de CNN pré-treinado 
Redes neurais convolucionais (CNN) são arquiteturas 
biologicamente inspiradas capazes de serem treinadas e 
aprenderem representações invariantes à escala, translação, 
rotação e transformações afi ns [13]. As CNN compõem um 
dos tipos de algoritmos da área conhecida como deep leaning 
e são projetadas para uso com dados em duas dimensões 
tornando-as uma boa candidata  para a solução de problemas 
envolvendo reconhecimento de imagens [14]. Por definição, 
uma arquitetura profunda é um a estrutura hierárquica de 
múltiplas etapas, onde cada et apa é formada por uma rede 
neuronal de pelo menos 3 camadas, e cada etapa é treinada por 
backpropagation. 
A Fig. 3 ilustra o modelo geral de uma arquitetura CNN. 
 
 
 
Figura 3. Modelo geral de uma arquitetura CNN. 
312 IEEE LATIN AMERICA TRANSACTIONS, VOL. 15, NO. 2, FEB. 2017
As redes neurais convolucionais dispõem de uma 
propriedade ainda pouco explorada conhecida como 
transferência de conhecimento.  Esta propriedade remete ao 
fato de uma CNN poder ser treinada em uma base de imagens 
A (sendo assim, seus pesos são ajustados para classificação da 
base A) sendo os pesos de aprendizado (e filtros nas CNN) 
considerados genéricos o suficiente para serem usados no 
treinamento de uma nova base B. Utilizando-se desse conceito 
foi selecionado o modelo CNN da biblioteca open source  
VLFeat, cujos dados utilizados para pré-treinamento da rede 
neural convolucional utilizada no trabalho apresentado foram 
obtidos junto ao arquivo “imagenet-caffe-alex.mat”. A 
utilização do modelo de CNN pr é-treinado supracitado não 
influenciou diretamente na taxa de reconhecimento das 
imagens de marcas de gado. 
 
C. Pré-processamento das imagens e aplicação da CNN 
 
O método adotado consiste em uma rede neural com cinco 
camadas convolucionais. Caso a imagem seja em tons de cinza 
é realizado um pré-processamento, no qual a imagem é 
replicada 3 vezes para criar uma imagem RGB. A 1ª camada 
convolucional tem como entrad a os 3 canais de cores da 
imagem (RGB). Cada convolução realiza a aplicação da 
função de ativação não linear ReLu e a redução através do 
Maxpooling. As últimas camadas são compostas por neurônios 
totalmente conectados. Para o treinamento da CNN foi 
utilizada a função Softmax e o algoritmo backpropagation.  
A Fig. 4 apresenta a arquitetura da rede neural 
convolucional proposta.  
 
 
Figura 4. Arquitetura da rede neural convolucional proposta. 
D. Extração de características das imagens 
O conjunto de filtros aprendidos pela CNN durante o 
treinamento é responsável pela detecção das características na 
nova imagem no momento de uma consulta. No primeiro nível 
de filtros é possível observar algumas linhas e orientações 
utilizadas para essa detecção. A Fig. 5 mostra os filtros 
aprendidos na primeira camada convolucional utilizando 
espaço de cores RGB. Existem 96 conjuntos individuais que 
representam os 96 filtros utilizad os nessa camada. É possível 
observar como as áreas com saliências horizontais, verticais e 
diagonais são destacadas apó s a realização da primeira 
convolução. 
 
 
 
Figura 5. Filtros da 1ª camada convolucional do experimento realizado. 
 
 
Após a realização das convoluçõe s foi possível realizar a 
extração das características das imagens de forma que o 
classificador pudesse ser treinado. A Fig. 6 apresenta uma 
ilustração do algoritmo de e xtração de características 
desenvolvido. 
 
 
 
Figura 6. Ilustração do algoritmo de extração de características desenvolvido. 
 
 
E. Treinamento e classificação das imagens com a utilização 
de Máquinas de Vetores de Suporte 
O modelo de aprendizagem automática adotado no 
t r a b a l h o  a p r e s e n t a d o  f o i  o  classificador supervisionado 
Support Vector Machine  ( S V M ) .  Support Vector Machine  é  
um algoritmo de classificação conhecido por ser bem sucedido 
numa grande variedade de aplicações. As SVM são uma das 
abordagens mais populares para modelagem e classificação de 
SILVA  et al.: CATTLE BRAND RECOGNITION USING 313
dados. As suas vantagens inclu em a excelente capacidade de 
generalização, que diz respeito à capacidade de classificar 
corretamente as amostras que n ão estão dentro do espaço da 
característica usado para o trein o [15]. Dadas duas classes e 
u m  c o n j u n t o  d e  p o n t o s  a  e s s a s  c l a s s e s ,  o  S V M  d e t e r m i n a  o  
hiperplano que separa os pontos de forma a colocar o maior 
número de pontos da mesma classe do mesmo lado, 
maximizando a distância de cada classe a esse hiperplano, 
sendo consequentemente denominado classificador de 
margem máxima [16]. Com efeito, uma larga margem entre os 
valores correspondentes aos pontos dos dois subconjuntos de 
dados implica um risco de generalização minimizado do 
classificador.  
As SVMs são utilizadas para classificar e reconhecer 
padrões em diversos tipos de dados, sendo utilizadas em 
diversas aplicações, tais com o reconhecimento de faces, 
diagnósticos clínicos, supervisão de processos industriais, 
processamento e análise de imagens [17]. 
Na ferramenta proposta nesse trabalho o classificador foi 
utilizado após a extração de características das marcas 
pertencentes a diferentes conjuntos de amostras. Na 
aprendizagem supervisionada dado um conjunto de exemplos 
(X
1, X 2) em X 1 representa um exemplo e X 2 a  s u a  
classificação, deve-se reproduz ir um classificador capaz de 
prever a classe a que pertencem novos dados, efetuando assim 
o processo de treino. Nesse contexto, utilizou-se o método 
cross-validation [ 1 5 ] ,  n a  q u a l  a s  i m a g e n s  f o r a m  d i v i d i d a s  
aleatoriamente em 2 partes, onde uma dessas partes foi 
utilizada para treinamento e a o utra para validação, de forma 
que não houvesse polarização dos resultados. O resultado final 
é a média do resultado obtido na validação. A divisão da 
percentagem utilizada foi de 30% para treinamento e 70% para 
validação.  
 
F. Avaliação dos resultados da classificação - Matriz de 
Confusão 
 
A matriz de confusão contém informações relativas a 
classificações efetuadas através da aplicação de um 
classificador. O desempenho dos classificadores é 
frequentemente avaliado através dos dados retirados desta 
matriz [18]. 
A matriz de confusão de um classificador indica o número 
de classificações corretas versus as previsões efetuadas para 
cada caso, sobre um conjunto de exemplos. Nesta matriz as 
linhas representam os casos reais e as colunas as previsões 
efetuadas pelo modelo. Através da matriz de confusão é 
possível obter informação rela tiva ao número de imagens 
corretamente classificadas e incorretamente classificadas, para 
cada conjunto de amostras. Esta  matriz é do tipo AxA, sendo 
A o número de categorias ao qual se aplica o classificador, no 
caso do experimento efetuad o existem 12 marcas, sendo 
portanto a matriz de confusão de tamanho 12x12.  
 
IV. RESULTADOS E DISCUSSÕES 
 
Através dos resultados obtidos foi possível avaliar o 
método proposto. A avali ação dos resultados dos 
experimentos foi realizada com base na taxa de 
reconhecimento obtida da matriz de confusão gerada a partir 
da classificação realizada na etapa de validação. Além disso, o  
tempo total de processamento do método proposto também foi 
verificado. 
A Fig. 7 apresenta a matriz de confusão para o melhor 
resultado obtido nos experimentos, cuja taxa de 
reconhecimento chegou a 93,28%. Percebe-se através da 
análise da diagonal principal que a taxa de classificação 
correta se destaca em quatro marcas, “802”, “812”, “815” e 
“821”, nas quais o percentual de acerto chega a 100%, 
correspondendo a 31 acertos. É possível observar também que 
as marcas que tiveram as menores taxas de classificação 
correta foram a “813” e a “805” , com percentual de 77,42% e 
80,64%, nessa ordem. As demais marcas, “803”; “804”; 
“811”; “814”; “822” e “109 3”, alcançaram taxa de 
classificação correta de 96,77% ; 93,55%; 90,32%; 90,3%; 
96,77% e 93,55%, respectivamente. 
A hipótese de classificação errada das marcas de gado 
conforme mostrada na matriz de confusão pode estar também 
associada à complexidade das amostras, pois algumas imagens 
possuem características similares entre si.  
Em geral, as amostras de imagens de marcas com maior 
poder descritivo e melhor qualidade classificaram 
corretamente mais imagen s, pois abrangem mais 
características quando comparadas às marcas com menor 
qualidade das amostras, e, consequentemente, menos 
características extraídas. A capacidade de reconhecer padrões 
de uma imagem sobre um conjunto de imagens depende da 
quantidade de informações que se conhece a priori do objeto 
em questão. 
Outro fato importante a ser apresentado, fruto da análise da 
Fig. 7, é o número baixo de falsos positivos e falsos negativos , 
observados fora da diagonal principal da matriz de confusão, o 
que se deve principalmente à capacidade do método proposto 
de extrair características de imagens, mesmo em situações 
adversas, em que as imagens apresentem diferentes tamanhos, 
formas, escalas, orientações, distorções, ruídos, cores 
diferentes e diferente contexto de fundo. Nesse tipo de 
experimento os ruídos nas imagens podem prejudicar a 
acurácia da classificação.  
 
 
 
Figura 7. Matriz de confusão obtida na etapa de validação. 
314 IEEE LATIN AMERICA TRANSACTIONS, VOL. 15, NO. 2, FEB. 2017
Tendo em consideração a matriz de confusão apresentada 
na Fig. 7, observa-se que através do somatório da diagonal 
principal obtém-se o total de marcas de gado classificadas 
corretamente, mais precisamente 347, ao passo que a soma 
dos demais valores equivale às marcas classificadas 
incorretamente, ou seja, 25. 
A Fig. 8 mostra um gráfico comparativo entre as marcas 
classificadas corretamente em função do número de amostras 
utilizadas na validação. 
 
 
 
Figura 8. Proporção de marcas corretamente classificadas em fun ção do 
número de amostras. 
 
As barras em azul apresentam as marcas de gado 
classificadas corretamente (acurácia), e as barras em vermelho 
apresentam o total de amostras de cada marca. Das 12 marcas 
analisadas, 4 apresentaram 100% de classificações corretas, no 
caso as marcas “802”; “812”; “815” e “821”.  As marcas de 
gado que tiveram a menor taxa de acerto foram a “813” e 
“805”, com 24 e 25 imagens classificadas corretamente, 
respectivamente. Observa-se t ambém o elevado índice de 
classificações corretas, mesmo nas marcas que não alcançaram 
100% de acerto, como no caso das marcas “803”; “804”; 
“811”; “814”; “822” e “1093”. 
O tempo de processamento do algoritmo em função do 
número de amostras de marcas de gado é apresentado na Fig. 
9.  
Foram medidos os tempos de processamento do método 
proposto para a classificação e m cinco grupos de amostras. 
Cada grupo contendo 108, 216, 324, 432 e 540 imagens. Os 
tempos de processamento para classificação das imagens em 
cada grupo foram de 6,206s; 8,01s; 9,488s; 11,295s e 12,716s, 
nessa ordem.  
A o  a n a l i s a r  o  g r á f i c o  i l u s t r a d o  n a  F i g .  9  é  p o s s í v e l  
perceber que o tempo de processamento observado no eixo y, 
varia de forma diretamente p roporcional ao aumento do 
número de amostras de marcas de gado classificadas do eixo 
x, ou seja, observa-se um padrão de crescimento linear da 
função, mesmo que a taxa de crescimento não seja exatamente 
um valor constante.    
 
 
Figura 9. Tempo de processamento do algoritmo em função do núme ro de 
amostras. 
 
Os resultados obtidos com os experimentos realizados com 
12 marcas de gado e 540 amostras de imagens, utilizadas tanto 
para treinamento quanto para validação, alcançaram taxa de 
reconhecimento de 93,28%, taxa de erro de 6,72% e tempo 
total de processamento de 12,716 segundos.  
A taxa de reconhecimento foi obtida mediante o cálculo da 
média aritmética das marcas classificadas corretamente da 
matriz de confusão, e o tempo total de processamento foi 
obtido mediante a utilização do so ftware MatLab, que ao final 
do processamento do código realiza o detalhamento da 
velocidade de processamento do algoritmo. 
 
V. CONCLUSÃO 
 
Nesse trabalho foi apresentado um método automatizado 
para reconhecimento de marcas de gado. O projeto foi 
desenvolvido entre duas instituições; a Prefeitura Municipal 
de São Francisco de Assis e UNIPAMPA.  
Os experimentos realizados neste trabalho utilizaram uma 
Rede Neural Convolucional (CNN) para extração de 
características e um classificador supervisionado SVM. Na 
CNN foi criada uma rede convolucional completa utilizando 
como entrada imagens transformadas para o formato de cores 
RGB. Todos os experimentos fo ram realizados na base de 
marcas de gado fornecida pela Prefeitura Municipal de São 
Francisco de Assis. 
A utilização do método proposto apresentou uma acurácia 
média de 93,28% e tempo de processamento do algoritmo de 
12,716 segundos para 12 marcas avaliadas, num total de 540 
amostras utilizadas para treino e validação. 
O método utilizado realizou eficaz e eficientemente o 
reconhecimento de diferentes marcas de gado, mesmo 
utilizando uma CNN pré-treinada, porém sua principal 
limitação foi a necessidade de uma grande quantidade de 
imagens de amostra para trei namento do classificador, uma 
vez que a CNN necessita dessas  imagens para extração de 
características nas camadas convolucionais. 
Como trabalho futuro objetiva-se a ampliação da base de 
imagens tomadas para treinamen to e validação, com o intuito 
de se obter novas medições de acurácia e desempenho do 
método proposto, visando a disponibilização de uma 
SILVA  et al.: CATTLE BRAND RECOGNITION USING 315
ferramenta devidamente validada e consolidada para os órgãos 
governamentais responsáveis pelo registro e controle das 
marcas de gado. 
 
REFERÊNCIAS 
 
[1] SEPLAN. Secretaria do Planejamento e Desenvolvimento Regional – 
Governo do Estado do Rio Grande do Sul, Brasil . 2015. URL: 
http://www.scp.rs.gov.br/atlas/conteudo.asp?cod_menu_filho=819&cod_men
u=817&tipo_menu=ECONOMIA&cod_conteudo=1580 
[2] ARNONI, R. “Os Registros e Catálogos de Marcas de Gado da R egião 
Platina.” Pelotas: Revista Memória em Rede da UFPEL, 2013. 
[3] SANCHEZ, G.; RODRIGUEZ, M. “Cattle Marks Recognition by Hu and 
Legendre Invariant Moments”. ARPN Journal of Engineering and Applied  
Sciences, Vol. 11, Nº 1, 2016. 
[4] JARRETT, K.; KAVUKCUOGLU, K.; LECUN, Y. “What Is The Best 
Multi-Stage Architecture for Object Recognition?”  IEEE 12th International 
Conference on Computer Vision, 2009.    
[5] JURASZEK, G. “Reconhecimento de Produtos por Imagem Utiliza ndo 
Palavras Visuais e Redes Neurais Convolucionais”. Joinville: UDESC, 2014. 
[6] CUN, L.; BOSER, B.; DENKER, J. S.; HENDERSON, D.; HOWARD, R . 
E.; HUBBARD, W.; JACKEL, L. D. “Handwritten Digit Recognition w ith a 
Back-Propagation Network”. In: Advances in Neural Information Processing 
Systems. [S.l.]: Morgan Kaufmann, 1990. p. 396-404.  
[7] CIRESAN, D.; MEIER, U.; SCHMIDHUBER, J. “Multi-Columm Deep 
Neural Networks for Image Classification”. In: Proceedings of the 25th IEEE 
Conference on Computer Visi on and Pattern Recognition  (CVPR 2012). [S.l.: 
s.n.], 2012. p. 3642-3649. 
[8] KAVUKCUOGLU, K.; SERMANET, P.; BOUREAU, Y. lan; GREGOR, 
K.; MATHIEU, M.; LECUN, Y. Learning Convolutional Feature Hierarchies 
for Visual Recognition. 2012. 
[9] SERMANET, P.; EIGEN, D.; ZHANG, X.; MATHIEU, M.; FERGUS, R. ; 
LECUN, Y. Overfeat: Integrated Recognition, Localization and Detection  
Using Convolutional Networks. CoRR, abs/1312.6229, 2013.  
[10] RAZAVIAN, A. S.; AZIZPOUR, H.; SULLIVAN, J.; CARLSSON, S. 
CNN Features Off-the-Shelf: An Astounding Baseline for Recognition.  CoRR, 
abs/1403.6382, 2014. 
[11] CONSTANTE, P.; GORDÓN, A.; CHANG, O.; PRUNA, E.; 
ESCOBAR, I.; ACUÑA, F. “Artificial Vision Techniques for Strawb erry's 
Industrial Classification”. IEEE Latin America Transactions, Vol. 14, Nº 6,  
2016. 
[12] VLFEAT. Biblioteca Open Source VLFeat . 2016. URL: 
http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat 
[13] LECUN, Y.; KAVUKCUOGLU, K.; FARABET, C. “Convolutional 
Networks and Applications in Vision”. In: Circuits and Systems (ISCAS), 
Proceedings of 2010 IEEE International Symposium on. [S.l.: s.n.], 2010. p. 
253-256.  
[ 1 4 ]  A R E L ,  I . ;  R O S E ,  D. ;  K A R NO W SK I ,  T .  “D e e p  M a c h i n e  L e a r n i n g  - A 
New Frontier in Artificial Inte llingence Research [research fro ntier]”. 
Computational Intelligence Magazine, IEEE , v. 5, n. 4, p. 13-18, 2010. ISSN 
1556-603X.  
[15] TEIXEIRA, A. “Desenvolviment o de uma Interface Gráfica par a 
Classificadores de Imagem”. 2016. URL: 
https://repositorio.ipcb.pt/bitstream/10400.11/1155/1/disserta%C3%A7ao.pdf 
[16] LU, H.; HUANG, Y. CHEN, Y.; YANG, D. “Real-Time Facial 
Expression Recognition Based on Pixel Pattern-Based Texture Fea ture”. In: 
Proc. Electronic Letters, pp. 916-918, 2007. 
[17] TCHANGANI, A. “Support Vector Machines: A Tool for Pattern 
Recognition and Classification” . Studies in Informatics & Control Journal  
14: 2. 99-110, 2005. 
[18] KOHAVI, R.; PROVOST, F. “Glossary of Terms” . Machine Learning , 
30(2-3), 271-274, 1998. 
[19] VAPINIK, N. “The Nature of Statistical Learning Theory”. N ew York: 
Springer, 1995. 
[20] KIM, K. I.; JUNG, K.; PARK, S. H.;  KIM, H. J. “Support Ve ctor 
Machines for Texture Classification”. IEEE Trans, PAMI, 2002. 
[21] LU, H.; HUANG, Y. CHEN, Y.; YANG, D. “Real-Time Facial 
Expression Recognition Based on Pixel Pattern-Based Texture Fea ture”. In: 
Proc. Electronic Letters, pp. 916-918, 2007. 
[22] LOVELL, B.; WALDER, C. “Support Vector Machines for Busine ss 
Applications”. Business Applications and Com putational Intelligence, Idea 
Group Publishers, 2006. 
 
Carlos Silva é analista de sistemas da Prefeitura Municipal de 
São Francisco de Assis, Rio Grande dos Sul, Brasil. Graduado 
em informática pela Universidade da Região da Campanha 
(URCAMP) em 2010 e especialista em engenharia de sistemas 
pela (ESAB) em 2012. Atualme nte é mestrando em Engenharia 
Elétrica pela Universidade Federal do Pampa (UNIPAMPA). 
Suas pesquisas concentram-se em análise e processamento de 
imagens, aprendizado de má quina e visão computacional. 
Membro ativo da SBC e IEEE. 
 
Daniel Welfer é professor da Universidade Federal de Santa 
Maria (UFSM) lotado no Departamento de Computação 
Aplicada (DCOM).  Concluiu seu doutorado em Ciência da 
Computação pela  Universidade Federal do Rio Grande do Sul 
(UFRGS) em 2011. Suas áreas de interesse são processamento e 
análise de imagens médicas, morfologia matemática e sistemas 
de informação hospitalar. Atualmente é membro permanente do 
Programa de Pós-Graduação em Informática (PPGI)  da UFSM. 
 
Francisco Paulo Gioda possui graduação em medicina 
veterinária pela Universidade Federal de Santa Maria (UFSM) 
em 1978. É o médico veterinário responsável do Sindicato Rural 
de São Francisco de Assis, Rio Grande do Sul, Brasil. 
Atualmente é secretário de adm inistração e planejamento da 
cidade de São Francisco de Assi s e coordenador do setor de 
registro de marcas de gado do município. 
 
Claudia Dornelles possui graduação em administração pela 
Universidade Norte do Paraná (UNOPAR) em 2014. É auxiliar 
administrativo da Prefeitura de São Francisco de Assis, Rio 
Grande do Sul, Brasil e responsáv el pelo registro e controle de  
marcas de gado do município. 
 
316 IEEE LATIN AMERICA TRANSACTIONS, VOL. 15, NO. 2, FEB. 2017