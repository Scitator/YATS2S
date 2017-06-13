import numpy as np
import tensorflow as tf


from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn

def create_embedding_matrix(vocab_size, embedding_size, scope=None, reuse=False):
    with tf.variable_scope(scope or "embeddings", reuse=reuse) as scope:
        # Uniform(-sqrt(3), sqrt(3)) has variance ~= 1.
        sqrt3 = np.sqrt(3)
        initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

        embedding_matrix = tf.get_variable(
            name="embedding_matrix",
            shape=[vocab_size, embedding_size],
            initializer=initializer,
            dtype=tf.float32)
        return embedding_matrix


class Embeddings(object):
    def __init__(self, vocab_size, embedding_size, special):
        self.loss = None
        self.train_op = None

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.special = special

        self.scope = self.special.get("scope", "Embeddings")
        self.reuse_scope = self.special.get("reuse_scope", False)
        with tf.variable_scope(self.scope, self.reuse_scope):
            self.embedding_matrix = create_embedding_matrix(
                self.vocab_size, self.embedding_size, scope=self.scope)
            self.global_step = tf.contrib.framework.get_global_step()
            # self.global_step = tf.get_variable(
            #     "global_step", [],
            #     trainable=False,
            #     dtype=tf.int64,
            #     initializer=tf.constant_initializer(
            #         0, dtype=tf.int64))
