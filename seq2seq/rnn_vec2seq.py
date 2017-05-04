import tensorflow as tf
from seq2seq.embeddings import Embeddings
from seq2seq.rnn_decoder import DynamicRnnDecoder
from rstools.tf.optimization import build_model_optimization, build_scope_optimization


class DynamicVec2Seq(object):
    def __init__(self,
                 vocab_size, embedding_size,
                 encoder_args, decoder_args,
                 embeddings_optimization_args=None,
                 encoder_optimization_args=None,
                 decoder_optimization_args=None):
        self.embeddings = Embeddings(
            vocab_size,
            embedding_size,
            scope="embeddings")

        self.inputs_vec = tf.placeholder(
            tf.float32, shape=(None, ) + encoder_args["input_shape"])

        with tf.variable_scope("encoder"):
            state = tf.layers.dense(self.inputs_vec, encoder_args["state_shape"])
            self.encoder_state = tf.transpose(state, [1, 0])

        self.decoder = DynamicRnnDecoder(
            encoder_state=self.encoder_state,
            encoder_outputs=None,
            encoder_inputs_length=None,
            embedding_matrix=self.embeddings.embedding_matrix,
            **decoder_args)

        build_model_optimization(self.decoder, decoder_optimization_args)
        build_model_optimization(self.embeddings, embeddings_optimization_args, self.decoder.loss)
        self.encoder_optimizer, self.encoder_train_op = build_scope_optimization(
            var_list=tf.get_collection(tf.GraphKeys, "encoder"),
            optimization_params=encoder_optimization_args,
            loss=self.decoder.loss)
