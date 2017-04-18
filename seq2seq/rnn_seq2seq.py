import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
from seq2seq.embeddings import Embeddings
from seq2seq.rnn_encoder import DynamicRnnEncoder
from seq2seq.rnn_decoder import DynamicRnnDecoder
from seq2seq.optimization import build_optimization
from seq2seq.os_utils import create_if_need
from seq2seq.batch_utils import time_major_batch


class DynamicSeq2Seq(object):
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

        self.encoder = DynamicRnnEncoder(
            embedding_matrix=self.embeddings.embedding_matrix,
            **encoder_args)

        self.decoder = DynamicRnnDecoder(
            encoder_state=self.encoder.state,
            encoder_outputs=self.encoder.outputs,
            encoder_inputs_length=self.encoder.inputs_length,
            embedding_matrix=self.embeddings.embedding_matrix,
            **decoder_args)

        build_optimization(self.encoder, encoder_optimization_args, self.decoder.loss)
        build_optimization(self.decoder, decoder_optimization_args)
        build_optimization(self.embeddings, embeddings_optimization_args, self.decoder.loss)
