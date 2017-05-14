import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import  input_data

def xavier_init(fan_in,fan_out ,constant=1):
    low=-constant*np.sqrt(6.0/(fan_in+fan_out))
    high=constant*np.sqrt(6.0/(fan_in+fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low,
                             maxval=high,
                             dtype=tf.float32)

class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.transfer=transfer_function
        self.scale=tf.placeholder(tf.float32)
        self.training_scale=scale
        network_weights=self._initialize_weights()
        self.weights=network_weights

        self.x=tf.placeholder(tf.float32,[None,self.n_input])

        # 编码过程
        self.hidden=self.transfer(tf.add(tf.matmul(
            self.x+scale * tf.random_normal(n_input,mean=0.0,stddev=1.0,),self.weights['w1'])
            ,self.weights['b1']))

        # 解码重构过程
        self.reconstruction=tf.add(tf.matmul(self.hidden,self.weights['w2']),self.weights['b2'])

        # 损失函数
        self.cost=0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction.self.x),2.0))

        self.optimizer=optimizer.minimize(self.cost)


        init=tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)

    # 初始化参数函数
    def _initialize_weights(self):
        all_weights = dict()

        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))

        all_weights['b1'] = tf.Variable(tf.zeros(self.n_hidden), dtype=tf.float32)

        all_weights['w2'] = tf.Variable(tf.zeros(self.n_hidden, self.n_input), dtype=tf.float32)

        all_weights['b2'] = tf.Variable(tf.zeros(self.n_input), dtype=tf.float32)

        return all_weights

    # 只有cost和训练过程optimizer
    def partial_fit(self,X):
        cost,opt =self.sess.run((self.cost,self.optimizer),
                                feed_dict={self.x:X, self.scale:self.training_scale})

        return cost

    # 只有cost计算
    def calc_total_cost(self,X):
        return self.sess.run(self.cost, feed_dict={self.x:X, self.scale:self.training_scale})



    # 返回编码器隐含层的结果，即编码之后的结果
    def transform(self,X):
        return self.sess.run(self.hidden,feed_dict={self.x:X, self.scale:self.training_scale})

    # 将隐含层输出的结果，通过解码重构复原为原始数据
    def generate(self,hidden=None):
        if hidden is None:
            hidden=np.random.normal(size=self.weights['b1'])

            return  self.sess.run(self.reconstruction,feed_dict={self.x:self.hidden})

    # 从原始数据到复原数据，即transform过程和generate过程合二为一
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X, self.scale:self.training_scale})


    # 获取隐含层权重w1
    def getWeight(self):
        return self.sess.run(self.weights['w1'])

    #获取隐含层的偏置系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])



mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 数据标准化处理
def standard_scale(X_train,X_test):
    preprocessor=prep.StandardScaler().fit(X_train)
    X_train=preprocessor.transform(X_train)
    X_test=preprocessor.transform(X_test)
    return X_train,X_test

def get_random_block_from_data(data,batch_size):
    start_index=np.random.randint(0,len(data)-batch_size)
    return data[start_index:(start_index+batch_size)]

X_train,X_test=standard_scale(mnist.train.images,mnist.test.images)

autoencoder=AdditiveGaussianNoiseAutoencoder(n_input=784,
                                             n_hidden=200,
                                             transfer_function=tf.nn.softplus,
                                             optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                             scale=0.01)

