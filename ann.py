#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 01:23:33 2016

@author: safeer
"""
import numpy as np

import tensorflow as tf
from sklearn.preprocessing import scale
import sklearn.metrics as met
#%%
print("Processing Training Data")

Xtr = np.loadtxt("OHE_KD99_f.csv",delimiter=',')[:,:86]
Ytr = np.loadtxt("OHE_KD99_l.csv",delimiter=',')
Xtr = scale(Xtr,axis=0)
test_indices = np.arange(len(Ytr))
np.random.shuffle(test_indices)
X = Xtr[test_indices]
Y = Ytr[test_indices]
Xte = X[:10000]
Xtr = X[10000:]
Yte = Y[:10000]
Ytr = Y[10000:]

#%%
print("Starting Training")

def init_weights(shape, sn):
    return tf.Variable(tf.random_normal(shape, stddev=0.1),name=sn)

def init_bias(shape,sn):
    return tf.Variable(tf.abs(tf.random_normal(
                                    shape,stddev=0.01),name=sn))
    
def model(X, wh1, wh2, wh3, wo, bh1, bh2, bh3, drop):

    X = tf.nn.dropout(X, drop)

    h = tf.nn.relu(tf.matmul(X, wh1) + bh1)

    h = tf.nn.dropout(h, drop)

    h2 = tf.nn.relu(tf.matmul(h, wh2) + bh2)

    h2 = tf.nn.dropout(h2, drop)
    
    h3 = tf.nn.relu(tf.matmul(h2, wh3) + bh3)

    h3 = tf.nn.dropout(h3, drop)

    return tf.matmul(h3, wo)
    
n_input = 86
n_output = 5
hlayer_1 = 100
hlayer_2 = 100
hlayer_3 = 50
n_epoch = 20
lr = 0.003
dropout_rate = 0.5
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
drop = tf.placeholder("float")

wh1 = init_weights([n_input, hlayer_1], "wh1")
wh2 = init_weights([hlayer_1, hlayer_2], "wh2")
wh3 = init_weights([hlayer_2, hlayer_3], "wh3")
wo = init_weights([hlayer_3, n_output], "wo")

bh1 = init_bias([hlayer_1], "bh1")
bh2 = init_bias([hlayer_2], "bh2")
bh3 = init_bias([hlayer_3], "bh3")

saver = tf.train.Saver([wh1, wh2, wh3, wo, bh1, bh2, bh3])

pY = model(X, wh1, wh2, wh3, wo, bh1, bh2, bh3, drop)


#pYV = modelV(X, wh1, wh2, wo, bh1, bh2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pY,Y))
train_op = tf.train.RMSPropOptimizer(lr, 0.9).minimize(cost)

pOP = tf.argmax(pY, 1)

#pOPV = tf.argmax(pYV, 1)


with tf.Session() as sess:
    tf.initialize_all_variables().run()
#    #saver.restore(sess,"model_norm_relu.ckpt")
    for i in range(n_epoch):
        for start, end in zip(range(0, len(Xtr), 400), range(400, len(Xtr)+1, 400)):
            sess.run(train_op, feed_dict={X:Xtr[start:end], Y:Ytr[start:end],
                                          drop:dropout_rate})
            
        #saver.save(sess, "model_norm_relu.ckpt")
        print(i,np.mean(np.argmax(Ytr,axis=1)==sess.run(pOP, feed_dict={X: Xtr,
                        Y: Ytr, drop:1.0})),
              np.mean(np.argmax(Yte,axis=1) ==sess.run(pOP, feed_dict={X: Xte,
                      Y: Yte, drop:1.0})))
    #saver.restore(sess,"model_norm_relu.ckpt")

    pred = sess.run(pOP, feed_dict={X:Xte,Y:Yte,drop:1.0})
    true = np.argmax(Yte, axis=1)    
    print "Precision ", met.precision_score(true,pred)
    print "Recall ", met.recall_score(true,pred)
    print "F1_Scare ", met.f1_score(true,pred)
    print met.confusion_matrix(true,pred)
                  
        