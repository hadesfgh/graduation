import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from unknown_data import UnknownData
from problem_data import ProblemData
from fill_normalization import fill,normalization
from kmeansByAndrew import runKmeans,plotData
from K_means改 import kpp_centers

def encode(data):
    learning_rate = 0.012
    # r1 = -3*np.random.rand()
    # learning_rate = np.power(10,r1)

    training_epochs = 1000
    n_input = 336

    X = tf.placeholder("float", [None, n_input])

    n_hidden_1 = 128
    n_hidden_2 = 64
    n_hidden_3 = 10
    n_hidden_4 = 2

    weights = {
        'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], )),
        'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], )),
        'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], )),
        'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], )),
        'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3], )),
        'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2], )),
        'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], )),
        'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], ))
    }

    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b4': tf.Variable(tf.random_normal([n_input]))
    }

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                       biases['encoder_b3']))
        layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                         biases['encoder_b4'])
        return layer_4

    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                       biases['decoder_b3']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                       biases['decoder_b4']))
        return layer_4

    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    y_pred = decoder_op
    y_true = X

    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    """Saver会保存在其之前的所有变量"""
    saver = tf.train.Saver(max_to_keep=1)

    with tf.Session() as sess:
        sess.run(init)

        # for epoch in range(training_epochs):
        #     _, c = sess.run([optimizer, cost], feed_dict={X: data})
        #     if epoch % 100 == 0:
        #         print("cost={:.9f}".format(c))
        # encoder_result = sess.run(encoder_op, feed_dict={X: data})
        # plt.scatter(encoder_result[:, 0], encoder_result[:, 1])
        # saver.save(sess, 'net/net-4-/encode.ckpt')

        model_file = 'net/net-4-80%/encode.ckpt'
        saver.restore(sess, model_file)
        encoder_result = sess.run(encoder_op, feed_dict={X: data})
        plt.scatter(encoder_result[:, 0], encoder_result[:, 1])
    return encoder_result

unknowndata = UnknownData()
unknowndata = fill(unknowndata, 1778, 336, 254)
unknowndata = normalization(unknowndata, 1778, 336)
result = encode(unknowndata)
# # plt.show()

# tf.reset_default_graph()
# problemdata = ProblemData()
# problemdata = fill(problemdata, 2093, 336, 299)
# problemdata = normalization(problemdata, 2093, 336)
# result_ = encode(problemdata)

init_centroids = np.array(kpp_centers(result, 2))
idx, centroids_all = runKmeans(result, init_centroids, 100)
centroids = centroids_all[-1]
plotData(result, centroids_all, idx)

plt.show()