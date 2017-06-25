import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from seq2seq.embeddings import create_embedding_matrix
from seq2seq.dynamic_decode import dynamic_rnn_decode, dynamic_targets


class DynamicRnnEncoder(object):
    def __init__(self, cell, bidirectional=False,
                 embedding_matrix=None, vocab_size=None, embedding_size=None,
                 lm_regularization=False, output_layer=None,
                 special=None, defaults=None, mode="train"):
        assert embedding_matrix is not None \
               or (vocab_size is not None and embedding_size is not None)
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.mode = mode

        self.loss = None
        self.train_op = None
        self.cell = cell
        self.bidirectional = bidirectional
        self.lm_regularization = lm_regularization
        self.output_layer = output_layer
        self.special = special or {}
        self.PAD = self.special.get("PAD", 0)
        self.EOS = self.special.get("EOS", 1)

        self.scope = self.special.get("scope", "DynamicRnnEncoder")
        self.reuse_scope = self.special.get("reuse_scope", False)
        with tf.variable_scope(self.scope, self.reuse_scope):
            self._build_embeddings()
            self._build_graph(defaults)

            self.global_step = tf.contrib.framework.get_global_step()

            if (self.mode == tf.estimator.ModeKeys.TRAIN or
                        self.mode == tf.estimator.ModeKeys.EVAL):
                self._build_loss()

    def _build_embeddings(self):
        if self.embedding_matrix is not None:
            self.vocab_size, self.embedding_size = self.embedding_matrix.get_shape().as_list()
        else:
            self.embedding_matrix = create_embedding_matrix(
                self.vocab_size, self.embedding_size)

    def _build_graph(self, defaults):
        if defaults is None:
            self.inputs = tf.placeholder(
                shape=(None, None),
                dtype=tf.int32,
                name="encoder_inputs")
            self.inputs_length = tf.placeholder(
                shape=(None,),
                dtype=tf.int32,
                name="encoder_inputs_length")
        else:
            self.inputs = tf.placeholder_with_default(
                defaults["inputs"],
                shape=(None, None),
                name="encoder_inputs"
            )
            self.inputs_length = tf.placeholder_with_default(
                defaults["inputs_length"],
                shape=(None,),
                name="encoder_inputs_length")

        with tf.variable_scope("embedding"):
            self.inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.inputs)

        if self.bidirectional:
            with tf.variable_scope("BidirectionalEncoder"):
                ((encoder_fw_outputs,
                  encoder_bw_outputs),
                 (encoder_fw_state,
                  encoder_bw_state)) = (
                    tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=self.cell[0] if isinstance(self.cell, tuple) else self.cell,
                        cell_bw=self.cell[1] if isinstance(self.cell, tuple) else self.cell,
                        inputs=self.inputs_embedded,
                        sequence_length=self.inputs_length,
                        time_major=False,
                        dtype=tf.float32))

                self.outputs = tf.concat(
                    (encoder_fw_outputs, encoder_bw_outputs), 2,
                    name="bidirectional_output_concat")

                if isinstance(encoder_fw_state, rnn.LSTMStateTuple):  # LstmCell
                    state_c = tf.concat(
                        (encoder_fw_state.c, encoder_bw_state.c), 1, name="bidirectional_concat_c")
                    state_h = tf.concat(
                        (encoder_fw_state.h, encoder_bw_state.h), 1, name="bidirectional_concat_h")
                    self.state = rnn.LSTMStateTuple(c=state_c, h=state_h)
                elif isinstance(encoder_fw_state, tuple) \
                        and isinstance(encoder_fw_state[0], rnn.LSTMStateTuple):  # MultiLstmCell
                    self.state = tuple(map(
                        lambda fw_state, bw_state: rnn.LSTMStateTuple(
                            c=tf.concat((fw_state.c, bw_state.c), 1,
                                        name="bidirectional_concat_c"),
                            h=tf.concat((fw_state.h, bw_state.h), 1,
                                        name="bidirectional_concat_h")),
                        encoder_fw_state, encoder_bw_state))
                else:
                    self.state = tf.concat(
                        (encoder_fw_state, encoder_bw_state), 1,
                        name="bidirectional_state_concat")
        else:
            with tf.variable_scope("Encoder"):
                outputs, state = \
                    tf.nn.dynamic_rnn(
                        cell=self.cell,
                        inputs=self.inputs_embedded,
                        sequence_length=self.inputs_length,
                        time_major=False,
                        dtype=tf.float32)

                self.outputs = outputs
                self.state = state

        if self.lm_regularization:
            if self.output_layer is None:
                with tf.variable_scope("Decoder"):
                    self.output_layer = Dense(
                        self.vocab_size,
                        name="output_layer")

            if self.mode == tf.estimator.ModeKeys.TRAIN or \
                            self.mode == tf.estimator.ModeKeys.EVAL:
                with tf.name_scope("DecoderTrainFeed"):
                    self.train_inputs, self.train_targets, self.train_length, self.loss_weights = \
                        dynamic_targets(
                            targets=self.inputs,
                            targets_length=self.inputs_length,
                            pad_token=self.PAD,
                            end_token=self.EOS)

                with tf.variable_scope("embedding"):
                    self.train_inputs_embedded = tf.nn.embedding_lookup(
                        self.embedding_matrix, self.train_inputs)

                forward_cell = self.cell[0] if isinstance(self.cell, tuple) else self.cell
                batch_size, _ = tf.unstack(tf.shape(self.inputs))

                with tf.variable_scope("Decoder"):
                    ((self.train_outputs, self.train_sampled_ids),
                     self.train_state, self.train_lengths) = dynamic_rnn_decode(
                        mode=self.mode,
                        decode_mode="greedy",
                        cell=forward_cell,
                        initial_state=forward_cell.zero_state(
                            batch_size=batch_size,
                            dtype=tf.float32),
                        embeddings=self.embedding_matrix,
                        output_layer=self.output_layer,
                        inputs=self.train_inputs_embedded,
                        inputs_length=self.train_length)
                    self.train_logits = self.train_outputs

    def _build_loss(self):
        if self.lm_regularization:
            self.unreg_loss = self.loss = seq2seq.sequence_loss(
                logits=self.train_logits, targets=self.train_targets,
                weights=self.loss_weights)
        else:
            self.unreg_loss = self.loss = 0.0
