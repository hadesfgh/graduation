import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from unknown_data import UnknownData
from problem_data import ProblemData
from fill_normalization import fill,normalization
from kmeansByAndrew import runKmeans,plotData
from K_means改 import kpp_centers
from test_data_new import TestData


def encode(data):
    learning_rate = 0.014
    training_epochs = 2000
    n_input = 336

    X = tf.placeholder("float", [None, n_input])

    n_hidden_1 = 128
    n_hidden_2 = 64
    n_hidden_3 = 32
    n_hidden_4 = 16
    n_hidden_5 = 2

    weights = {
        'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1], )),
        'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], )),
        'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], )),
        'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4], )),
        'encoder_h5': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5], )),
        'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_5, n_hidden_4], )),
        'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3], )),
        'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2], )),
        'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1], )),
        'decoder_h5': tf.Variable(tf.truncated_normal([n_hidden_1, n_input], ))
    }


    biases = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_4])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_3])),
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b4': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b5': tf.Variable(tf.random_normal([n_input]))
    }

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                       biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                       biases['encoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                       biases['encoder_b3']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                       biases['encoder_b4']))
        layer_5 = tf.add(tf.matmul(layer_4, weights['encoder_h5']),
                         biases['encoder_b5'])
        return layer_5

    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                       biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                       biases['decoder_b2']))
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                       biases['decoder_b3']))
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                       biases['decoder_b4']))
        layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h5']),
                                       biases['decoder_b5']))
        return layer_5

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
        # print(sess.run(weights['encoder_h1']))

        # for epoch in range(training_epochs):
        #     _, c = sess.run([optimizer, cost], feed_dict={X: data})
        #     if epoch % 100 == 0:
        #         print("cost={:.9f}".format(c))
        # encoder_result = sess.run(encoder_op, feed_dict={X: data})
        # # plt.scatter(encoder_result[:, 0], encoder_result[:, 1])
        # saver.save(sess, 'net/net-5-6/encode.ckpt')

        # model_file = 'net/net-5-0-71.22%/encode.ckpt'
        # model_file = 'net/net-5-2-78.15%/encode.ckpt'
        model_file = 'net/net-5-3-72.69%/encode.ckpt'
        # model_file = 'net/net-5-4-80.46%/encode.ckpt'
        # model_file = 'net/net-5-5-86.76%/encode.ckpt'
        saver.restore(sess, model_file)
        # print(sess.run(weights['encoder_h1']))
        encoder_result = sess.run(encoder_op, feed_dict={X: data})
        # # plt.scatter(encoder_result[:, 0], encoder_result[:, 1])
    return encoder_result


if __name__ == '__main__':

    unknowndata = UnknownData()
    unknowndata = fill(unknowndata, 1778, 336, 254)
    unknowndata = normalization(unknowndata, 1778, 336)
    result = encode(unknowndata)

    # tf.reset_default_graph()
    # problemdata = ProblemData()
    # problemdata = fill(problemdata, 2093, 336, 299)
    # problemdata = normalization(problemdata, 2093, 336)
    # result_ = encode(problemdata)

    tf.reset_default_graph()
    testdata = TestData()
    testdata = fill(testdata, 476, 336, 68)
    testdata = normalization(testdata, 476, 336)
    result__ = encode(testdata)

    # init_centroids = np.array(kpp_centers(result, 2))
    # idx, centroids_all = runKmeans(result, init_centroids, 100)
    # centroids = centroids_all[-1]
    # plotData(result, centroids_all, idx)
    # # plt.savefig('F:\\导出的图片1.png')

    # plt.scatter(result_[:, 0], result_[:, 1])
    plt.scatter(result__[:, 0], result__[:, 1], c='red')

    y_pred = KMeans(n_clusters=2).fit_predict(result)
    plt.scatter(result[:, 0], result[:, 1], c=y_pred)
    # plt.savefig('F:\\导出的图片3.png')
    plt.show()

    # y_pred = DBSCAN(eps = 0.3, min_samples = 8).fit_predict(result)
    # plt.scatter(result[:, 0], result[:, 1], c=y_pred)
    # plt.show()

    # plt.show()