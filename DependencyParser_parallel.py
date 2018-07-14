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


            ############################################################ NN with parallel hidden layers #############################################################

            print " Training with the parallel hidden layers"


            numTrans = parsing_system.numTransitions()

            self.train_inputs = tf.placeholder(tf.int32, shape=[Config.batch_size, Config.n_Tokens])
            self.train_labels = tf.placeholder(tf.int32, shape = [Config.batch_size, numTrans])



            self.test_inputs = tf.placeholder(tf.int32, shape = [Config.n_Tokens, ])

            # look up on embeds with train inputs to get embeddings of the train_input sentences. These are 3D
            embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)


            # slice the embeds to get three separate embeds for words, pos and labels.
            embed_words = tf.slice(embed, [0, 0, 0], [Config.batch_size, 18, 50])
            embed_pos = tf.slice(embed, [0, 18, 0], [Config.batch_size, 18, 50])
            embed_labels = tf.slice(embed, [0, 36, 0], [Config.batch_size, 12, 50])

            # reshape  the 3d embeds to get 2d embeds.
            embed_words = tf.reshape(embed_words, [Config.batch_size, -1])
            embed_pos = tf.reshape(embed_pos, [Config.batch_size, -1])
            embed_labels = tf.reshape(embed_labels, [Config.batch_size, -1])


            # initialise three input weights in a particular range with truncated normal Function, inside tf.variable so that they will be learnt.
            weights_input_words = tf.Variable(tf.truncated_normal([Config.embedding_size * Config.n_Tokens_words, Config.hidden_size], stddev = 1.0 / math.sqrt(Config.hidden_size)))
            weights_input_pos = tf.Variable(tf.truncated_normal([Config.embedding_size * Config.n_Tokens_pos, Config.hidden_size], stddev = 1.0 / math.sqrt(Config.hidden_size)))
            weights_input_labels = tf.Variable(tf.truncated_normal([Config.embedding_size * Config.n_Tokens_labels, Config.hidden_size], stddev = 1.0 / math.sqrt(Config.hidden_size)))

            # initialise three output weights in a particular range with truncated normal Function, inside tf.variable so that they will be learnt.
            weights_output_words = tf.Variable(tf.truncated_normal([Config.hidden_size,numTrans], stddev = 1.0 / math.sqrt(Config.hidden_size)))
            weights_output_pos = tf.Variable(tf.truncated_normal([Config.hidden_size,numTrans],stddev=1.0 / math.sqrt(Config.hidden_size)))
            weights_output_labels = tf.Variable(tf.truncated_normal([Config.hidden_size,numTrans],stddev=1.0 / math.sqrt(Config.hidden_size)))

            # initialise three biases for three parallel layers with zero inside tf.variable so that they will be learnt.
            biases_input_words = tf.Variable(tf.zeros([Config.hidden_size]))
            biases_input_pos = tf.Variable(tf.zeros([Config.hidden_size]))
            biases_input_labels = tf.Variable(tf.zeros([Config.hidden_size]))


            # three forward pass for each word, pos and label embeds.
            self.prediction_words = self.forward_pass(embed_words, weights_input_words, biases_input_words, weights_output_words)
            self.prediction_pos = self.forward_pass(embed_pos, weights_input_pos, biases_input_pos, weights_output_pos)
            self.prediction_labels = self.forward_pass(embed_labels, weights_input_labels, biases_input_labels, weights_output_labels)

            # three cross entropy loss for above three predictions
            ce_loss_words = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction_words, labels=tf.argmax(tf.to_int64(tf.maximum(self.train_labels, 0)),axis=1))
            ce_loss_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction_pos, labels=tf.argmax(tf.to_int64(tf.maximum(self.train_labels, 0)),axis=1))
            ce_loss_labels = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction_labels, labels=tf.argmax(tf.to_int64(tf.maximum(self.train_labels, 0)),axis=1))


            # regularizing three input weights, biases, output weights and the embeddings.
            ip_weight_regularizer_words = tf.nn.l2_loss(weights_input_words)
            ip_weight_regularizer_pos = tf.nn.l2_loss(weights_input_pos)
            ip_weight_regularizer_labels = tf.nn.l2_loss(weights_input_labels)

            h_weight_regularizer_words = tf.nn.l2_loss(weights_output_words)
            h_weight_regularizer_pos = tf.nn.l2_loss(weights_output_pos)
            h_weight_regularizer_labels = tf.nn.l2_loss(weights_output_labels)

            h_bias_regularizer_words = tf.nn.l2_loss(biases_input_words)
            h_bias_regularizer_pos = tf.nn.l2_loss(biases_input_pos)
            h_bias_regularizer_labels = tf.nn.l2_loss(biases_input_labels)

            embed_regularizer_words  = tf.nn.l2_loss(embed_words)
            embed_regularizer_pos  = tf.nn.l2_loss(embed_pos)
            embed_regularizer_labels  = tf.nn.l2_loss(embed_labels)


            # create loss by adding  cross entopy loss and l2_penalties.
            self.loss = tf.reduce_mean(ce_loss_words + ce_loss_pos + ce_loss_labels + (Config.lam) *  ip_weight_regularizer_words
            			                             + (Config.lam) *  ip_weight_regularizer_pos
            			                             + (Config.lam) *  ip_weight_regularizer_labels
                                                     + (Config.lam) *  h_weight_regularizer_words
                                                     + (Config.lam) *  h_weight_regularizer_pos
                                                     + (Config.lam) *  h_weight_regularizer_labels
                                               	     + (Config.lam) *  h_bias_regularizer_words
                                               	     + (Config.lam) *  h_bias_regularizer_pos
                                               	     + (Config.lam) *  h_bias_regularizer_labels
                                               	     + (Config.lam) *  embed_regularizer_words
                                               	     + (Config.lam) *  embed_regularizer_pos
                                               	     + (Config.lam) *  embed_regularizer_labels  )


            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)

            ##################### without gradient clipping ###########################
            #self.app = optimizer.apply_gradients(grads)
            # For test data, we only need to get its prediction


            ###### testing phase for NN with 3 parallel layers #####################
            # For test data, we only need to get its prediction

            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            #print("test embed: ", test_embed)

            test_embed_words = tf.slice(test_embed, [0, 0], [18, test_embed.get_shape()[1]])
            test_embed_pos = tf.slice(test_embed, [18, 0], [18, test_embed.get_shape()[1]])
            test_embed_labels = tf.slice(test_embed, [36, 0], [12, test_embed.get_shape()[1]])


            test_embed_words = tf.reshape(test_embed_words, [1, -1])
            test_embed_pos = tf.reshape(test_embed_pos, [1, -1])
            test_embed_labels = tf.reshape(test_embed_labels, [1, -1])


            test_pred_words = self.forward_pass(test_embed_words, weights_input_words, biases_input_words, weights_output_words)
            test_pred_pos = self.forward_pass(test_embed_pos, weights_input_pos, biases_input_pos, weights_output_pos)
            test_pred_labels = self.forward_pass(test_embed_labels, weights_input_labels, biases_input_labels, weights_output_labels)


            self.test_pred = (test_pred_words + test_pred_pos + test_pred_labels) / 3

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
        #print "h_temp", h_temp
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
        lc = c.getLeftChild(i,1)
        rc = c.getRightChild(i,1)
        lc_rc_list.append(lc)
        lc_rc_list.append(rc)
        lc_rc_list.append(c.getLeftChild(i,2))
        lc_rc_list.append(c.getRightChild(i,2))
        lc_rc_list.append(c.getLeftChild(lc,1))
        lc_rc_list.append(c.getRightChild(rc,1))
    ########################### 18 Word Features ###########################
    for i in [w1,w2,w3,b1,b2,b3]:

        feature_list.append(getWordID(c.getWord(i))) # 6 words

    for i in lc_rc_list:    #12
        feature_list.append(getWordID(c.getWord(i)))

    ########################### 18 Tag Features ###########################
    for i in [w1,w2,w3,b1,b2,b3]:

        feature_list.append(getPosID(c.getPOS(i))) # 6 words

    for i in lc_rc_list:
        feature_list.append(getPosID(c.getPOS(i)))  #12
    ########################### 12 label Features ###########################
    for i in lc_rc_list:
        feature_list.append(getLabelID(c.getLabel(i)))  #12


    return feature_list




def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])
            #print "#######################",c

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
    #print len(wordDict)
    #print len(posDict)
    #print len(labelDict)
    #print wordDict

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
    #print "embedding_array shape",embedding_array.shape

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
