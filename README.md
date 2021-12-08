# Dependency-Parser

Implemented Stanford's Transition Dependency Parse based on  https://www.emnlp2014.org/papers/pdf/EMNLP2014082.pdf. 
</br>
For more info: 
https://nlp.stanford.edu/software/nndep.html

## Model Components

#### Embeddimgs : Word2vec / Random initialised Embeddings

#### Features : POS embedding and Arc Label Embeddings

In detail, Sw contains nw = 18 elements: 
</br>
(1) The top 3 words on the stack and buffer: s1, s2, s3, b1, b2, b3
</br>
(2) The first and second leftmost / rightmost children of the top two words on the stack:lc1(si), rc1(si), lc2(si), rc2(si), i = 1, 2. 
</br>
(3) The leftmost of leftmost / rightmost of rightmost children of the top two words on the stack: lc1(lc1(si)), rc1(rc1(si)), i = 1, 2.
</br>
(4) We use the corresponding POS tags for St(nt = 18), and the corresponding arc labels of words excluding those 6 words on the stack/buffer for Sl (nl = 12).

#### Hidden Layers : 1 

#### Loss Function : Cross Entropy Loss

#### Optimiser : AdaGrad

#### Activation function : Cubic Activation
             

## Experiments and Command to run them
### Basic Neural Net

DependencyParser.py
</br>
run - python DependencyParser.py
               
### Neural net with two hidden Layers:

DependencyParser_hidden_layer_2.py
</br>
run - python DependencyParser_hidden_layer_2.py
         
### Neural net with three hidden Layers: 

DependencyParser_hidden_layer_3.py
</br>
run - python DependencyParser_hidden_layer_3.py

### Neural Net with three parallel hidden layers for POS, Labels and Tags:

DependencyParser_parallel.py
</br>
run python DependencyParser_parallel.py
    

### Neural Net without training Embeddings:
DependencyParser_fixed.py
</br>
run python DependencyParser_fixed.py


