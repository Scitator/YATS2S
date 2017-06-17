import tensorflow as tf
from seq2seq.embeddings import Embeddings
from seq2seq.rnn_encoder import DynamicRnnEncoder
from seq2seq.rnn_decoder import DynamicRnnDecoder


def build_model_optimization(model, optimization_args=None, loss=None):
    loss = model.loss if model.loss is not None else loss
    assert loss is not None
    optimization_args = optimization_args or {}

    # global_step = tf.contrib.framework.get_global_step()

    learning_rate_decay_fn = lambda learning_rate, global_step: \
        tf.train.exponential_decay(
            learning_rate, global_step,
            decay_steps=optimization_args.get("decay_steps", 100000),
            decay_rate=optimization_args.get("decay_rate", 0.99))

    # very magic fn!
    model.train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=model.global_step,
        learning_rate=optimization_args.get("learning_rate", 1e-4),
        optimizer=tf.train.AdamOptimizer,
        learning_rate_decay_fn=learning_rate_decay_fn,
        clip_gradients=optimization_args.get("clip_gradients", 10.0),
        variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model.scope))


class DynamicSeq2Seq(object):
    def __init__(self,
                 vocab_size, embedding_size,
                 encoder_args, decoder_args,
                 embeddings_optimization_args=None,
                 encoder_optimization_args=None,
                 decoder_optimization_args=None,
                 mode="train"):
        self.embeddings = Embeddings(
            vocab_size,
            embedding_size,
            special={"scope": "embeddings"})

        self.encoder = DynamicRnnEncoder(
            embedding_matrix=self.embeddings.embedding_matrix,
            **encoder_args)

        self.decoder = DynamicRnnDecoder(
            encoder_state=self.encoder.state,
            encoder_outputs=self.encoder.outputs,
            embedding_matrix=self.embeddings.embedding_matrix,
            mode=mode,
            **decoder_args)

        build_model_optimization(self.encoder, encoder_optimization_args, self.decoder.loss)
        build_model_optimization(self.decoder, decoder_optimization_args)
        build_model_optimization(self.embeddings, embeddings_optimization_args, self.decoder.loss)
