### Convolutional Neural Networks  for Sentence Classification

improve upon the state of art on 4 out of 7 tasks( sentiment analysis and question classification)

code：

- https://github.com/yoonkim/CNN_sentence（Code is written in Python (2.7) and requires Theano (0.7)）
- https://github.com/dennybritz/cnn-text-classification-tf ， in TensorFlow python3

#### introduction

- Word vectors:  1-of-V encoding (V is the vocabulary size )  transformed into lower dimensional vector space via a hidden layer, which can be an essentially feature extractors.

  >  **one-hot vector**: 将所有单词排序，排序之后每个单词就会有一个位置，然后用一个与单词数量等长的数组表示某单词，该单词所在的位置数组值就为1，而其他所有位置值都为0.
  >
  >  **Word embedding**: 将一个词映射为一个空间向量，通过向量之间的相似度（欧氏距离）来度量词与词之间的相似度。 
  >
  >  https://www.jianshu.com/p/7864843880e5

- three types of input : pre-trained, task-specific, both of  pre-trained and task-specific

- the work is philosophically similar to Razavian which showed that for image classification, feature extractors obtained from a pre-trained deep learning model perform well on a variety of tasks

#### Model

<img src="C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570752664515.png" alt="1570752664515" style="zoom:100%;" />

##### input

- $x_i \in R^k$ be the k-dimensional word vector corresponding to the $i$-th word in the sentence, $i=1,2,3,\cdots,n$，$n$ is the max-length of all sentences.

- the sentence " wait for the video and do not rent it"  can be transformed  a matrix with dimension $n\times k$,   $x_{i:i+j} $ refer to the concatenation of words $x_i,x_{i+1},x_{i+2},\cdots,x_{i+j}$

##### convolution

- filter $w \in R^{hk} $ which is applied to a window of $h$ words to produce a new feature

   ![img](https://upload-images.jianshu.io/upload_images/5118838-b30cf5e96d669504.png?imageMogr2/auto-orient/strip|imageView2/2/w/1045/format/webp) 

- $c_i = f(w \cdot x_{i:i+j}+b)$ ; $c=[c_1,c_2,c_3,\cdots,c_{n-h+1}]$

##### max-pooling

- $\hat{c}=max\{c\}$

- to capture the most important feature—one with the highest value—for each feature map, This pooling scheme naturally deals with variable sentence lengths.

- one feature is extracted from one filter to obtain multiple features, These features form
  the penultimate layer and are passed to a fully connected softmax layer whose output is the probability distribution over labels.

  

##### dropout

- given the penultimate layer $z=[\hat{c_1},\hat{c_2},\hat{c_3},\cdots,\hat{c_m}]$
- for output unit $y$ in forward propagation, dropout uses $y=w\cdot(z\cdot r)+b$, $r \in R^m$ is a 'masking' vector of random variables with probability $p$ of being 1. Gradients are backpropagated only through the unmasked units.

- the learned weight vectors are scaled by $p$ such that $\hat{w}=p\cdot w$ 
  $$
  E(\hat{y})=E(w\cdot(z\cdot r)+b)=w\cdot E[(z\cdot r)]+b=w\cdot[z\cdot E(r)]+b
  \\=w\cdot [z \cdot [\begin{matrix}0.8\times1+0.2\times0&\cdots&\\\cdots&&\\\end{matrix}]]+b\\=w\cdot [z \cdot [\begin{matrix}0.8&\cdots&\\\cdots&&\\\end{matrix}]]+b=0.8\cdot E(y))
  $$

  > 

- Dropout consistently added 2%–4% relative performance

##### $l_2-norms$

- constrain $l_2-norms$ of the weight vectors by rescaling $w$ to have $||w||_2=s$ whenever $||w||_2 >s$ after a gradient descent step.

#### Dataset

![1570752726035](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570752726035.png)

use 7 datasets with different amounts of classes. All of them are short text

#### Parameters tunning

- filter windows $h$ of  3, 4, 5 with 100 feature maps each, dropout rate (p) of 0.5, $l_2$  constraint (s) of 3, and
  mini-batch size of 50.
- Training is done through stochastic gradient descent over shuffled mini-batches with the Adadelta update rule.https://www.cnblogs.com/neopenx/p/4768388.html
- Pre-trained Word Vectors from Google news, The vectors have dimensionality of 300 and were trained using the continuous bag-of-words architecture







#### Model Variations

- **CNN-rand**: all words are randomly initialized and then modified during training.
- **CNN-static**:  All words— including the unknown ones that are randomly initialized—are kept static from Google news and only the other parameters of the model are learned.
- **CNN-no-static**: Same as above but the pretrained vectors are fine-tuned for each task.
- **CNN-mutichannel** : Both channels are initialized with word2vec,  the model is able to fine-tune
  one set of vectors while keeping the other static.

#### Result

![1570752762266](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570752762266.png)

![1570752781139](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\1570752781139.png)

- the pretrained vectors are good, ‘universal’ feature extractors and can be utilized across datasets
- Finetuning the pre-trained vectors for each task gives still further improvements



#### Further observation

- When randomly initializing words not in word2vec, we obtained slight improvements by sampling each dimension from  $U[-a,a] $ where $a$ was chosen such that the randomly initialized vectors have the same variance as the pre-trained ones.

  > if employing more sophisticated methods to mirror the distribution of pre-trained vectors in the initialization process gives further improvements.

- experimented with another set of publicly available word vectors trained on Wikipedia and found that word2vec gave far superior performance. It is not clear whether this is due to architecture or the 100 billion word Google News dataset.
  
