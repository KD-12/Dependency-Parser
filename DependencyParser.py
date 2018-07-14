import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            """
            ===================================================================

            Define the computational graph with necessary variables.

            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()

            self.train_inputs =
            self.train_labels =
            self.test_inputs =
            ...


            2) Call forward_pass and get predictions

            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam

            ...
            self.loss =

            ===================================================================
            """


            numTrans = parsing_system.numTransitions()

            self.train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size,Config.n_Tokens])
            self.train_labels = tf.placeholder(tf.int32, shape=[Config.batch_size,numTrans])


            #self.prob = tf.placeholder_with_default(1.0, shape=())

            self.test_inputs = tf.placeholder(tf.int32, shape = None)

            # look up on embeds with train inputs to get embeddings of the train_input sentences. These are 3D
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)

            embed = tf.reshape(embed, [Config.batch_size, -1])

            ############################################################ NN with 1 Hidden Layer ###########################################################

            # initialise  input weight and output weight in a particular range with truncated normal Function, inside tf.variable so that they will be learnt.
            weights_input = tf.Variable(tf.truncated_normal([Config.embedding_size*Config.n_Tokens,Config.hidden_size ],stddev=1.0 / math.sqrt(Config.hidden_size)))


            weights_output = tf.Variable(tf.truncated_normal([Config.hidden_size,numTrans],stddev=1.0 / math.sqrt(Config.hidden_size)))

            # initialise biases as zeros inside tf.variable so that they will be learnt.
            biases_input = tf.Variable(tf.zeros([Config.hidden_size]))


            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            #sparse_softmax_cross_entropy_with_logits requires the data with a specific rank thats why taking argmax on the train labels and for filtering we use tf.maximum
            #which will convert all labels with value less than zero to zero.

            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction, labels=tf.argmax(tf.to_int64(tf.maximum(self.train_labels, 0)),axis=1))

            # regularizing three input weights, biases, output weights and the embeddings.
            ip_weight_regularizer = tf.nn.l2_loss(weights_input)
            h_weight_regularizer = tf.nn.l2_loss(weights_output)
            h_bias_regularizer = tf.nn.l2_loss(biases_input)
            embed_regularizer  = tf.nn.l2_loss(embed)

            # create loss by adding  cross entopy loss and l2_penalties.
            self.loss = tf.reduce_mean(ce_loss + (Config.lam) *  ip_weight_regularizer
                                               + (Config.lam) *  h_weight_regularizer
                                               + (Config.lam) *  h_bias_regularizer
                                               + (Config.lam) *  embed_regularizer  )


            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            ##################### without gradient clipping ###########################
            #self.app = optimizer.apply_gradients(grads)
            # For test data, we only need to get its prediction

            ###### testing phase for NN with one layer #####################

            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized with truncated normal"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]
            len(batch_inputs)
            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)

                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights_input, biases_inpu, weights_output):

        W_E = tf.matmul(embed,weights_input)

        h_temp = tf.add(W_E, biases_inpu)

        ##################### Cubic Function #############################################
        cubing_tensor = tf.constant([3.0])
        h = tf.pow(h_temp, cubing_tensor)
        #################### Sigmoid Function ############################################
        #h = tf.nn.sigmoid(h_temp)
        ###################### tanh function ####################################################
        #h = tf.nn.tanh(h_temp)
        ###################### ReLu Function ################################################
        #h = tf.nn.relu(h_temp)


        return tf.matmul(h,weights_output)






def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1
    print labelDict
    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """


    feature_list = []
    lc_rc_list = []
    w1 = c.getStack(0)
    w2 = c.getStack(1)
    w3 = c.getStack(2)
    b1 = c.getBuffer(0)
    b2 = c.getBuffer(1)
    b3 = c.getBuffer(2)
    for i in [w1, w2]:   #12
        lc = c.getLeftChild(i,1) # 1 st left child of the word on the stack.
        rc = c.getRightChild(i,1) # 1 st right child of the word on the stack.
        lc_rc_list.append(lc)
        lc_rc_list.append(rc)
        lc_rc_list.append(c.getLeftChild(i,2))  # 2 nd left child of the word on the stack
        lc_rc_list.append(c.getRightChild(i,2)) # 2 nd right child of the word on the stack
        lc_rc_list.append(c.getLeftChild(lc,1)) # 1 st left child of the left child of the word on the stack
        lc_rc_list.append(c.getRightChild(rc,1)) # 1 st right child of the right child of the word on the stack
    ########################### 18 Word Features ###########################
    for i in [w1,w2,w3,b1,b2,b3]:

        feature_list.append(getWordID(c.getWord(i))) # 6 words of the stack and buffer

    for i in lc_rc_list:    #12 words of the tree
        feature_list.append(getWordID(c.getWord(i)))

    ########################### 18 Tag Features ###########################
    for i in [w1,w2,w3,b1,b2,b3]:

        feature_list.append(getPosID(c.getPOS(i))) # 6 tags of the owrds on the stack and the buffer

    for i in lc_rc_list:
        feature_list.append(getPosID(c.getPOS(i)))  #12 tags of the words onthe stack and the buffer.
    ########################### 12 label Features ###########################
    for i in lc_rc_list:
        feature_list.append(getLabelID(c.getLabel(i)))  #12 labels of the words on the stack and the buffer.


    return feature_list




def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):

                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)

    print len(features)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))


    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)


    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)

    print "Done."

    # Build the graph model
    graph = tf.Graph()

    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)
