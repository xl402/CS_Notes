# Machine Learning Notes
[^1]:Created by: Tom Xiaoding  Lu on 09/19/18

## Contents[^1]
* [Natural Language Processing](#natural-language-processing)
  * [TF-IDF](#tf-idf)
  * [Markov Property of Language Model](#markov-property-of-language-model)
  * [Smoothing Techniques](#smoothing-techniques)
    * [Laplacian Smoothing](#laplacian-smoothing)
    * [Katz Backoff](#katz-backoff)
    * [Interpolation Smoothing](#interpolation-smoothing)
    * [Kneser-Ney Smoothing](#kneser-ney-smoothing)
  * [Hidden Markov Model](#hidden-markov-model)
  * [Probabilistic Neural Language Model](#probabilistic-neural-language-model-legacy)
  * [Log-Bilinear Language Model](#log-bilinear-language-model)
  * [Vector Space Models of Semantics](#vector-space-models-of-semantics)
    * [SVD On Point Mutual Information](#svd-on-point-mutual-information)
    * [Global Vectors for Word Representation](#global-vectors-for-word-representation-glove)
    * [Skip-gram Model (Word2Vec)](#skip-gram-model-word2vec)
  * [Topic Modelling](#topic-modelling)
    * [Probabilistic Latent Semantic Analysis](#probabilistic-latent-semantic-analysis-plsa)
* [Tensor Flow](#tensor-flow)
  * [Running Sessions](#running-sessions)
  * [Visualizing Graphs](#visualizing-graphs)
* [Google Cloud](#google-cloud)
  * [Launch Cloud Datalab](#launch-cloud-datalab)
  * [Apache Beam](#apache-beam)
    * [Local Example](#local-example)
    * [Cloud Job Example](#cloud-job-example)
    * [Map VS FlatMap](#mal-vs-flatmap)


## Natural Language Processing
### TF-IDF
Stands for Term Frequency-Inverse Document Frequency, used to reflect how important a word is to a document in a collection or corpus. This value **increases proportionally to the number of times a word appears in the document** and is offset by the number of documents containing the word.
For a given term (or n-gram) $t$ in a document $d$, define the tern frequency $tf(t,d)$ to be:
```Math
tf(t,d) = \dfrac{f_{t,d}}{\sum_{t'\in d}f_{t',d}}
```
In other words, the count of the term in document over total term count in document, it has the log normalization of
```Math
\overline{f}_{t,d}=1+\log(f_{t,d})
```
The **inverse document requency** is a measure of how rare the term is across all documents, it is logarithmically scaled inverse fraction of the documents that contain the word, obtained by dividing the total number of documents by the number of documents containing the term:
```Math
idf(t, D)=\log\dfrac{N}{|\{d \in D : t \in d\}|}
```

In python, use the TfidfVectorizer package:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_features(X_train, X_val, X_test):

  # min, max frequency arguments: integer for number of occurrences in document, float between 0
  # and 1 for percentage of occurrences
  tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')
  X_train = tfidf_vectorizer.fit_transform(X_train)
  X_val = tfidf_vectorizer.transform(X_val)
  X_test = tfidf_vectorizer.transform(X_test)

  return X_train, X_val, X)test, tfidf_vectorizer.vocabulary
```
### Markov Property of Language Model
First, predict the probability of a sequence of words:
$$
\bm{w}=(w_1, w_2, w_3, ..., w_k)
$$
According to chain rule:
$$
p(\bm{w})=p(w_1)p(w_2|w_1)p(w_3|w_1,w_2)...p(w_k|w_1...w_{k-1})
$$
This can however be simplified with Markov property which states that **we only care about up to $n$ previous steps** when computing the probability of the occurrence of the current state:
$$
p(w_i|w_1...w_{i-1}) \approx p(w_{i-n+1}...w_{i-1})
$$
This in scope of a **bigram language model**, by applying Markov's property, using $n=2$ we get:
Toy corpus:
* This is the malt
* That lay in the house that Jack built
$$
p(\textsf{this is the house})=p(\textsf{this}|\textbf{Start})p(\textsf{is}|\textsf{this})p(\textsf{the}|\textsf{is})p(\textsf{house}|\textsf{the})p(\textbf{End}|\textsf{House})
$$
The special **Start** and **End** points are added to the sentence to ensure correct normalization of sentences of different lengths. Putting all of these together, we have the formalised Markov likelihood equation:
$$
\mathcal{L} = p(\bm{w})=\prod_{i=1}^{k+1}p(w_i|w_{i-n+1}^{i-1})
$$
With estimated probabilities being the count of occurrences of ensemble including the current term divided by the occurrences of ensemble excluding the current term:
$$
p(w_i|w_{i-n+1}^{i-1}) = \dfrac{c(w_{i-n+1},...,w_{i-1}, w_i)}{c(w_{i-n+1},...,w_{i-1})} = \dfrac{c(w_{i-n+1}^{i})}{c(w_{i-n+1}^{i-1})}
$$
As it turned out, equation above is the solution of the log-likelihood maximization of the Markov equation above.

**Perplexity** is a measurement of how well a probability model predicts a sample. It may be used to compare probability models. A low perplexity indicates the probability distribution is good at predicting a sample. It is defined as:

$$
\mathcal{P} =p(\bm{W_{\textsf{test}}})^{-1/N}
$$
Where $N$ is the length of the test corpus (all words concatenated). Intuitively, it normalizes the likelihood with how long the actual text is, so that low likelihood due to long text will be of higher quality than slightly higher likelihood but with much shorter text.

### Smoothing Techniques:
Sometimes, we may end up with infinite perplexity even when we have a reasonable set of testing and training data, for example, given the training corpus: *This is the house that Jack Built* and the test corpus *This is Jack*, we have:
$$
p(\textsf{Jack}|\textsf{is}) = \dfrac{c(\textsf{is Jack})}{c(\textsf{is})}=0
$$
This will results in $\mathcal{L}=0$ therefore the perplexity $\mathcal{P}= \inf $. In order to combat this, several smoothing techniques are applied.
#### Laplacian Smoothing
Idea: pull some probability from frequent bigrams to infrequent ones, either just add 1 to the counts (*add one smoothing*):
$$
\hat{p}(w_i|w_{i-n+1}^{i-1}) = \dfrac{c(w_{i-n+1}^{i})+1}{c(w_{i-n+1}^{i-1})+V}
$$
where $V$ is the total vocabulary/term size (i.e. unique n-grams within text ensemble). In order to gain more control over the effect of smoothing, we can tune a parameter (*add-k smoothing*):
$$
\hat{p}(w_i|w_{i-n+1}^{i-1}) = \dfrac{c(w_{i-n+1}^{i})+k}{c(w_{i-n+1}^{i-1})+kV}
$$
#### Katz Backoff
Longer n-grams are better, but data is not always enough. Hence we try to use longer n-grams, when it fails we simply back off to a shorter one:
```math
\hat{p}(w_i|w_{i-n+1}^{i-1}) =
\begin{cases}
\bar{p}(w_i|w_{i-n+1}^{i-1}) & \iff c(w_{i-n+1}^{i})>0\\
\alpha (w_{i-n+1}^{i-1})  \hat{p}(w_i|w_{i-n+2}^{i-1}) & \textrm{, otherwise}
\end{cases}
```
Note the $\alpha (w_{i-n+1}^{i-1})$ is simply the normalizing constant term used to ensure that the $\hat{p}$ is bounded between 0 and 1.


#### Interpolation Smoothing
Idea: similar to [Katz Backoff](#katz-backoff) method, just make $\hat{p}$ a superposition of different n-gram probabilities:
$$
\hat{p} (  w_i | w_{i-n+1}^{i-1} )  = \lambda_1 p(  w_i  |  w_{i-2},  w_{i-1}  ) + \lambda_2 p(  w_i  |  w_{i-1}  ) + \lambda_3 p(  w_i )
$$

Where the parameters must satisfy $\lambda_1 + \lambda_2 + \lambda_3 = 1$. The weights are optimized on a test set. optionally they can also depend on the context.

#### Kneser-Ney Smoothing
From the concept of absolute discounting, we can write the conditional probability of the current term given its previous term to be an "interpolation" of the unigram offset-ed by some number:

$$
\hat{p} (  w_i | w_{i-1})  = \lambda(w_{i-1}) p(w_i)  + \dfrac{c(w_{i-1}, w_i) - d } {\sum_x c(w_{i-1}, x)}
$$
As it turned out, this offset can be represented by the proportion of the bigrams containing the current word over the count of bigrams, this is also offset by a magic number "d" being 0.75 in English. This is termed **"Absolute Discounting"**. Note that within the unigram term, we can do better than just purely using the frequency of the current word, for example:

*"Why did Anne get off with another ____"*

Within our corpus, the occurrence of "pig" is higher than "guy", but it does not make sense for Anne to shag a pig, therefore we use how likely is the word to appear as a novel continuation defined as:
$$
P_{\textrm{continuation}}(w_i) \propto |{w_{i-1} : c(w_{i-1}, w_i) > 0}|
$$
Here we are just counting how many times the current word forms a unique bigram with its previous term. In this case, we may then select "guy" over "pig". This may seem like a bad example, but think of weird terms like "Francisco" and how it normally appears after the word "San".

The actual $P_{\textrm{continuation}}$ is simply the normalized version of the equation above:

$$
P_{\textrm{continuation}}(w_i) = \dfrac{|{w_{i-1} : c(w_{i-1}, w_i) > 0}|} {\sum_{\mathcal{w'}}|{\mathcal{w'}_{i-1} : c(\mathcal{w'}_{i-1}, \mathcal{w'}_i) > 0}|}
$$
Putting all these together, we have:
$$
\hat{p} (  w_i | w_{i-1})  = \lambda(w_{i-1}) P_{\textrm{continuation}}(w_i)  + \dfrac{c(w_{i-1}, w_i) - d } {\sum_x c(w_{i-1}, x)}
$$
which is the Kneser-Ney smoothing, it is widely considered the most effective method of smoothing.

### Hidden Markov Model
One major usage of NLP is text tagging, this is also used for text generation. The idea is, given a sequence of input text *"Isn't it funny"* we wish to generate the next most probable text. To do this however, we first map the input sequence to a **most probable sequence of tags**, we then use this to predict what the next most likely text is. Formally we define $\bm{x}=x_1, ..., x_T$ being a sequence of words and $\bm{y}=y_1, ..., y_t$ being a sequence of its corresponding tags, and we wish to find:
$$
\bm{y}=\textrm{argmax}_\bm{y}p(\bm{y|x})=\textrm{argmax}_\bm{y}p(\bm{y,x})
$$
What is shown above is simply reproduced as a diagram here (taken from Winnie the Pooh):
![markov_model](/assets/markov_model.png)
From the model shown above, the tags above are the hidden layer with the text being the output layer, we can construct the following transition diagram:
<img align="left" src="/assets/markov_list.png">
The transitions can be represented by two probability matrices **transition** matrix and **emission** matrix, in this case, they are denoted as follows (denoting *AUX* as *au*, *PART* as *pa* and *PRON* as *pr*):
$$
\mathcal{T}=
  \begin{bmatrix}
    p(\textrm{au|au}) & p(\textrm{pa|au}) & p(\textrm{pro|au})\\
    p(\textrm{au|pa}) & p(\textrm{pa|pa}) & p(\textrm{pro|pa})\\
    p(\textrm{au|pr}) & p(\textrm{pa|pr}) & p(\textrm{pr|pr})\\
  \end{bmatrix}
$$
$$
\mathcal{E}=
\begin{bmatrix}
  p(\textrm{is|au}) & p(\textrm{not|au}) & p(\textrm{it|au})\\
  p(\textrm{is|pa}) & p(\textrm{not|pa}) & p(\textrm{it|pa})\\
  p(\textrm{is|pr}) & p(\textrm{not|pr}) & p(\textrm{it|pr})\\
\end{bmatrix}
$$
Of course the model denoted above is over simplified, in fact the matrix grows exponentially as the length of the tags increase, fortunately, we can use Markov's assumption to simplify this problem ($\bm{x}$ is observable, $\bm{y}$ is hidden, $i$ being the index of the tags/text within the sequence):
$$
p(\bm{x,y}) = p(\bm{x|y})p(\bm{y})\approx \prod_{i=1}^{n}p(x_i|y_i)p(y_i|y_{i-1})
$$
which states that the probability of the current state is only dependent on the previous $T$ number of states.
In order to find the probabilities of these hidden states ($\mathcal{T}_{i,j}$ and $\mathcal{E}_{i,j}$), provided that we have labelled tags as training data, we compute a **trigram** Hidden Markov Model which can be defined using:
* A finite set of states.
* A sequence of observations.
* $t(s|u,v)$ **Transition probability** defined as the probability of a state “s” appearing right after observing “u” and “v” in the sequence of observations.
* $e(x|s)$ **Emission probability** defined as the probability of making an observation x given that the state was s.

Then, the generative model for the probability of a sequence of labels given a sequence of observations over “n” time steps would be estimated as
$$
p(x_1, ..., x_n, y_1, ..., y_{n+1}) = \prod_{i=1}^{n}e(x_i|y_i)\prod_{i=1}^{n+1}q(y_i|y_{i-1},y_{i-2})
$$
With each individual probability defined as:
$$
t(s|u,v) =\dfrac{c(u, v, s)}{c(u,v)}
$$
$$
e(x|s)=\dfrac{c(s \leadsto x)}{c(s)}
$$

Finally, we are going to solve the problem of finding the most likely sequence of labels given a set of observations $x_1, x_2, ..., x_n$. That is we wish to find out:
$$
\textrm{argmax}_{\bm{y}}p(\bm{x},\bm{y})
$$
Before looking at an optimized algorith, lets see what the brute force method produces for a small set of example, given the following text sequence: *"the dog barks"* and a finite set of tags *{D, N, V}*, we have the following possible combinations:
```python
Input:    they   dog   barks

Tags:      D      D     D
           D      D     N
           D      N     N
           N      N     N
           D      V     N  ,etc
```
Totalling $3^3=27$ examples. The possible tags sequence grow exponentially w.r.t both the text sequence length and the number of possible tags! Computing $p(\bm{x},\bm{y})$ for each combination would be impossible.
Instead of this brute force approach, we will see that we can find the highest probable tag sequence efficiently using a dynamic programming algorithm known as the **Viterbi Algorithm**.
Let us refine a truncated version of the markov model of generated probability and let us call this the cost of a sequence of length k.:
$$
r(y_1, ..., y_k)=\prod_{i=1}^k q(y_i|y_{i-1}, y_{i-2})\prod_{i=1}^k e(x_i|y_i), k\in\{1, ..., n\}
$$
Next we have the set $S(k, u, v)$ which is basically the set of all label sequences of length $k$ that end with the bigram $(u, v)$. Figure below shows the pseudo code for finding the most likely next two words given a sequence of words:
<div style="text-align: center"><img src="/assets/viterbi.png" width="600" /></div>

The algorithm first fills in the $\pi (k,u,v)$ values in using the recursive definition. It then uses the identity described before to calculate the highest probability for any sequence.

Let us practice this on a actual sequence tagging example, say we have both the trained transition and emission matricies, both our vocabulary and number of different tags are very small, our transition matrix is denoted as:

<div style="text-align: center"><img src="/assets/DeepinScreenshot_select-area_20180926134150.png" width="300" /></div>

and our emission matrix is denoted as:

<div style="text-align: center"><img src="/assets/DeepinScreenshot_select-area_20180926134207.png" width="400" /></div>

At some point during the sequence, we may face the state looking like figure below:

<div style="float: left; text-align: right"><img src="/assets/DeepinScreenshot_select-area_20180926134639.png" height="250" /></div>

<div style="text-align: right"><img src="/assets/DeepinScreenshot_select-area_20180926135006.png" height="250" /></div>

by refering to the table above, we fill in all the edges from tags for *"bear"* to all tags for *"likes"*, and from all the tags for *"likes"* to the actual word *likes*. Doing this for the first tag pair, we see something like the picture on the left. Repeat the computation above and choose whichever path that maximizes the probability of *"likes"*, remember this path, and proceed to the following computation. We finally arrive at the terminating step:

<div style="text-align: center"><img src="/assets/DeepinScreenshot_select-area_20180926135207.png" width="350" /></div>

All we need to do is to trace backwards forming the most probable sequence of tags.



### Probabilistic Neural Language Model (Legacy)
First, we represent each word within the text as a vector (using BOM, TF-IDF,etc). Then the input feature vector is simply the n-grams vectors concatenated together:
$$
x = [C(w_{i-n+1}), ..., C(w_{i-1})]^T
$$
For example, to produce outputs for the sentence *Anne is a horrible person* using 3-grams approach we have the first input being $x_0 = [C(\textsf{Anne}), C(\textsf{is})]^T$.

Next, we have the output $y$ (tags, vocabulary,etc) being a linear combination of both the raw input feature and the non-linear transformed input feature through some hidden layer:
$$
y =b + Wx + U\tanh (d+Hx)
$$
We can interpret $W$ as "literal" interpretation from the previous two words, $H$ being the "context" provided by the two words, $U$ being the "literal" interpretation of our learnt "concept".

Finally, we transfer the output $y$ into a probability distribution of different tags/vocabulary outputs. This is a simple *softmax* transformation:
$$
p(w_i| w_{i-n+1}, ..., w_{i-1}) =\dfrac{\exp y_{w_i}}{\sum_{w \in V} \exp (y_w)}
$$

**Dimensional analysis:** $[y] = [b] = |V|$ where  $|V|$ is the size of the vocabulary. $[W] = |V \times m*(n-1)|$ where $m$ is the dimension of the vector representation of one word, $n$ is the number of grams. The problem with this method is it is **overly complicated.**

### Log-Bilinear Language Model
After studying the probabilistic neural language model in the previous section, we can effectively simplify the the learnt parameters of $W$ and $H$ through a simple step. Note since the *softmax* function is intrinsically non-linear, we also do not need the *tanh* function.

let $y = \hat{r}^Tr_{w_i}+b_{w_i}$, where $\hat{r}$ is the **context representation** of the n-grams, $r$ represents the literal representation $r_{w_i}=C(w_i)^T$ of the current word. The context representation is denoted as a superposition of some *linear transformation* of all words within the n-gram:

$$
\hat{r} = \sum_{k=1}^{n-1}W_k C(w_{i-k})^T
$$

Putting everything together, $y$ is simply the dot product between the literal and context representation of the n-gram, transform this into probability distribution using softmax we have:

$$
p(w_i| w_{i-n+1}, ..., w_{i-1}) =\dfrac{\exp (\hat{r}^Tr_{w_i}+b_{w_i})}{\sum_{w \in V} \exp (\hat{r}^Tr_{w}+b_{w})}
$$

Obviously now we only have parameters $W$ and $b$ to learn. Which is effectively $1/3$ of the computational complexity compared with method described in the previous section.

### Vector Space Models of Semantics
In order to measure how similar two words are, we use **second order co-occurrences** termed *paradigmatic parallels.*

For example, how do we know that *KFC* and *McDonald's* are similar? We take its surrounding nearest neighbours i.e for *KFC* its most frequent neighbours may be *burgers*, *chicken*, *nuggets*, *coke*, etc. Each with their own probability of occurrence. We would expect this distrubution to be similar to one for *McDonald's*.

This section devises different methods on mapping the semantics similarity of two wards onto a vector space.

#### SVD On Point Mutual Information
Let two words for which their semantics is compared be denoted as $W_i$ and $W_j$. Their features are represented as their $n$ most frequent neighbouring words: $W_i: \{u_{i-n}, ..., u_{i+n}\} n\neq 0$. i.e. for the word *KFC* the 3 most frequent words to the right may be *is*, *a*, *fast*. To its left may be *resturant*, *food*, *like*, therefore the feature for *KFC* is *{resturant, food, like, is, a, fast}*

One classic way of measuring the similarity of two words' semantics is through Point Mutual Information (PMI), measured as:
$$
\textrm{PMI}(u,v)=\log\dfrac{p(u,v)}{p(u)p(v)} = \log \dfrac{n_{uv} n}{n_u n_v}
$$
Where $n$ represents the total count, $n_{uv}$ represents the count where $u$ and $v$ occur together. From which we can form a characteristic matrix $\bm{X}$ where $\bm{X}_{u,v} = \textrm{PMI}(u,v)$. In case that $n_{uv}=0$, we have the positive Pointwise Mutual Information denoted as:
$$
\textrm{pPMI}=\max (0, \textrm{PMI})
$$
By using Singular Value Decomposition on the sparse characteristic matrix $X$ described above. We write down $\bm{X = U\Omega V}$. Since $\Omega$ is the square root of the eigenvalues of matrix $\bm{X}$, and is ordered from top left down to bottom right in descending magnitude, we can take an approximation keeping only $k$ terms of the eigenvalues. Therefore, the truncated SVD is denoted:
$$
\bm{X} \approx \bm{\hat{X}_k} = \bm{U_k\Omega_kV_k^T}
$$
When selecting the term $k$, we use Frobenius norm which is basically the $l2$ norm in matrix form:
$$
||\bm{X-\hat{X}}||_F= \sqrt{\sum_{i=1}^n \sum_{j=1}^m (x_{i,j}-\hat{x}_{i,j})^2}
$$

These three matrices can be combined in the following way to produce $\Phi$ and $\Theta$ matrices:
$$
\Phi=U_k \Omega_k, \Theta = V_k^T
$$
$$
\Phi = U_k \sqrt{\Omega_k}
, \Theta = \sqrt{\Omega_k} V_k^T$$
#### Global Vectors for Word Representation (GloVe)
There is another way of representing co-occurrences using $\Theta$ and $\Phi$ matrices, one of which is minimizing the following loss function:

$$
\sum_{u\in W}\sum_{v\in W} f(n_{uv}) (\langle \phi_u, \theta_v \rangle + b_u + b'_v - \log n_{uv})^2 \rightarrow \min_{\phi_u, \theta_v, b_u, b_v}
$$

Where $f(n_{uv})$ is an increasing function up to some threshold $x_{max}$ bounded between 0 and 1:
```math
f(n_{uv})=
\begin{cases}
\dfrac{n_{uv} }{x_{max}} \; \qquad ,\forall n_{uv} \leq x_{max} \\
1 \qquad  \qquad  \, ,\textrm{otherwise}
\end{cases}
```
#### Skip-gram Model (Word2Vec)
This section aims to learn the "context" ($\bm{e_c}$) of a word $\bm{x}$ through learning the transformation matrix $\Theta$, in otder to achieve this, we map a one hot encoded word $\bm{x}$ to a contect $\bm{y}$ which is a **probability distribution** of all vocabularies. i.e. $\bm{y} = [p(\textsf{car}), p(\textsf{glass}), p(\textsf{juice}), ..., p(\textsf{winter})]^T$ for $\bm{x_i} = \textsf{orange}$, $\bm{y_i}=[0.04, 0.13, 0.14, 0, 0, ..., 0]^T$. **But we are not interested in $\bm{y}$**, it is only used in trainign, all we want to know it the weights of the hidden layer!

This model forms the foundation of Word2Vec algorithm which maps a input text $x$ to a target $y$ in the form of either class labels or related words. The algorithm is trained on a large corpus. For example: we wish to input *time* and output words such as *zone*.
Let's say that we are given a sentence *"I want a glass of orange juice to go along with my cereal"*, we pick a random context word *orange*, we set up a supervised learning model which maps the **one hot encoded word** $\bm{x}$ (i.e. *orange*) to a range of targets words within a window (say of size 5), i.e. *glass, juice, along, cereal*.
In order to achieve this, we use one hidden and one output layer to achieve this, denoting hidden layer output as $e_c$ (context):
$$
e_c=\Theta x
$$
For each output, we have its corresponding weights being $\Phi_t$, we use softmax to convert output of the output layer into a probability distribution:
$$
p(y|\bm{x})=\dfrac{e^{\Phi_y^T e_c}}{\sum_{j=1}^{|V|}e^{\Phi_j^T e_c}}
$$
All parameters in matrices $\Theta$ and $Phi$ are learnt through backpropogation using the cross entropy loss function:
$$
\mathcal{L}(\bm{\hat{y}, y)} = -\sum_{i=1}^{|V|}\bm{y_i\log \hat{y}_i}
$$
Note this is very easy to compute as the target vector $y_i$ is one hot encoded hence the summation is more like a fake sum as there is only non zero value.
As softmax is very inefficient to compute, either more efficient storage system like huffman tree is used, or, more commonly, a **Skip-gram Negative Sampling (SGNS)** method is used. The idea behind it is we can greatly reduce the number of output vector space by limiting it to a length of 2, i.e. instead of predicting a word for another word, we compute **yes or no** for word pairs, trained on examples of word pairs $v$ and non word pairs (\bar{v}), we use sigmoid function $\sigma$ to produce the logistic regression output, we wish the maximize the function below:.
$$
\sum_{u\in W} \sum_{v \in W} n_{u,v} \log \sigma(\langle \Phi_u, \Theta_v\rangle) + k\mathbb{E}(\bar{v})\log\sigma(-\langle \Phi_u, \Theta_{\bar{v}}\rangle)    
$$
Where $k$ is the number of samples taken in the negative examples. Note that the whole expectation and $k$ bussiness are literally mathematical notations, in reality, for each training step, we just select at random a negative example and feed it into the evaluation network. So now, we would be feeding in word pairs to train such as *"orange_glass"* for positive example and *"orange_hindrance"* for negative example.
## Topic Modelling
### Probabilistic Latent Semantic Analysis (PLSA)
This is the most basic method for discovering topics within a corpus. We can marginalise the probability distribution of words within a document $p(w|d)$ into $\sum_{t\in T}p(w|t, d)p(t|d)$, we then assume that topics are independent of documents and finally write down:
$$
p(w|d) =\sum_{t\in T}p(w|t)p(t|d)
$$
In matrix form, we think of each topic $\Phi_t$ being a probability distribution of words in our vocabulary with matrix $\Phi$ with dimension $|V|\times |T|$ (numbers of vocabulary and topics) representing the probability distribution of words over all topics. Each document $\Theta_d$ consists of a probability distribution of topics, matrix $\Theta$ with dimension $|T|\times |D|$ (numbers of topics and documents) representing the probability distribution of topics over documents. We now have:
$$
p(w|d)=\sum_{t\in T}\Phi_{wt}\Theta_{td}
$$
Or in matrix multiplication form the matrix $\bm{X}$ representing the probability distribution of words in all documents:
$$
\bm{X}=\Phi \Theta
$$
Note $\Theta$ contains information on the distribution of words over each topic (vectorized topics), and $\Theta$ represents the distribution of topics over each document.

There are two methods to find the two matrices, one may use log-liklihood optimization. Firstly, we compute the probability of generating the completed version of our corpus of documents:
$$
p(D) = \prod_{d\in D} p(d) \prod_{w\in d} p(w|d)^{n_{wd}}
$$
Note the exponent $n_{wd}$ denotes the number of occurrences of the word $w$ within the document $d$, obviously as we are expecting $n_{wd}$ occurrences within the document, the total probability of the word being formed within the document should multiply each time it is generated.
Substituting the matrix representation of $p(w|d)$ in and taking the log, we obtain the log-liklihood optimization of PLSA model:
$$
\log \: \left(\prod_{d\in D} p(d) \prod_{w\in d}(\sum_{t\in T}\Phi_{wt}\Theta_{td})^{n_{dw}}\right) \rightarrow \max_{\Phi, \Theta}
$$
Assuming uniform distribution of documents $p(d)$, the expression above can be simplified into:
$$
\sum_{d\in D} \sum_{w\in D} n_{wd}\log \sum_{t\in T}\Phi_{wt}\Theta_{td} \rightarrow \max_{\Phi, \Theta}
$$
Note that we have constraints on these variables that need to optimized, since they cannot be negative and must be bounded to a probability distribution:
```math
\begin{cases}
\Phi_{wt}\geq 0 \qquad \qquad \& \qquad  \Theta_{td} \geq 0 \\ \\
\sum_{w\in W}\Phi_{wt}=1 \; \; \; \; \& \qquad \sum_{t\in T}\Theta_{td}=1
\end{cases}
```
This optimization problem is not easily solved, instead in reality **Expectation-Maximization Algorithm** is used.
## Tensor Flow

### Running Sessions
Running a session after "building":
```python
with tf.Session() as sess:

  result = sess.run(c)
  print(result)

  # Alternatively
  print c.eval()

  # run() accepts list of tensors
  a1, a3 = sess.run([z1, z3])
```
During development:
```python
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution
```


### Visualizing Graphs
Write the session graph to a summary directory
```python
with tf.Session() as sess:
  with tf.summary.FileWriter('summaries', sess.graph) as writer:
    a1, a3 = sess.run([z1, z3])
```
To visualize the graph with TensorBoard:
```python
from google.datalab.ml import TensorBoard
TensorBoard().start('./summaries')
```

## Google Cloud
### Launch Cloud Datalab
In Cloud Shell type:
```bash
gcloud compute zones list
datalab create mydatalabvm --zone <ZONE>
```
### Apache Beam
#### Local Example
```python
import apache_beam as beam
import sys

def my_grep(line, term):
  if line.startswith(term):
    yield line

if __name__ == '__main__':
  p = beam.Pipeline(argv=sys.argv)
  input ='../../data/*.java'
  output_prefix = '/tmp/output'
  searchTerm = 'import'

  (p
    | 'GetJava' >> beam.io.ReadFromText(input)
    | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm))
    | 'write' >> beam.io.WriteToText(output_prefix)
  )

  p.run().wait_until_finish()
```
#### Cloud Job Example

```python
import apache_beam as beam
import sys

def my_grep(line, term):
  if line.startswith(term):
    yield line

PROJECT = 'cloud-trainig-demos'
BUCKET = 'tomlu_1'


def run():

  argv =[
    '--project={0}'.format(PROJECT),
    '--job_name=examplejob2',
    '--save_main_session',
    '--staging_location=gs://{0}/staging/'.format(BUCKET),
    '--temp_locations=gs:/{0}/staging'.format(BUCKET),
    '--runner=DataflowRunner'
  ]


  p = beam.Pipeline(argv=sys.argv)
  input = 'gs://{0}/javahelp/*.java'.format(BUCKET)
  output_prefix = 'gs://{0}/javahelp/output'.format(BUCKET)
  searchTerm = 'import'

  (p
    | 'GetJava' >> beam.io.ReadFromText(input)
    | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm))
    | 'write' >> beam.io.WriteToText(output_prefix)
  )

  p.run()

if __name__ == '__main__':

  run()

```

#### Map VS FlatMap
Use Map for 1:1 relationship between input & output
```python
'WordLengths' >> beam.Map(lambda word: (word, len(word)))
```
FlatMap for non 1:1 relationships, usually with a generator:
```python
def vowels(word):
  for ch in word:
    if ch in ['a', 'e', 'i', 'o', 'u']:
      yield ch

'WordVowels' >> beam.FlatMap(lambda word: vowels(word))
```
