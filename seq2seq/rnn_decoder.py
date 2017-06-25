import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import seq2seq
from seq2seq.embeddings import create_embedding_matrix
from seq2seq.dynamic_decode import dynamic_rnn_decode, dynamic_targets
from tensorflow.contrib.rnn import LSTMStateTuple


class DynamicRnnDecoder(object):
    def __init__(self, cell, encoder_state, encoder_outputs, maximum_length=150,
                 attention=None, encoder_inputs_length=None,
                 training_mode="greedy", scheduled_sampling_probability=0.0,
                 inference_mode="greedy", beam_width=1,
                 embedding_matrix=None, vocab_size=None, embedding_size=None,
                 output_layer=None,
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
        self.encoder_state = encoder_state

        self.training_mode = training_mode
        self.scheduled_sampling_probability = tf.constant(
            scheduled_sampling_probability, shape=(), dtype=tf.float32,
            name="scheduled_sampling_probability")
        self.inference_mode = inference_mode
        self.beam_width = beam_width

        # @TODO: should be optimal
        self.encoder_outputs = encoder_outputs
        self.maximum_length = maximum_length
        self.attention = attention or False
        self.encoder_inputs_length = encoder_inputs_length
        self.output_layer = output_layer

        self.special = special or {}
        self.PAD = self.special.get("PAD", 0)
        self.EOS = self.special.get("EOS", 1)

        self.scope = self.special.get("scope", "DynamicRnnDecoder")
        self.reuse_scope = self.special.get("reuse_scope", False)
        with tf.variable_scope(self.scope, self.reuse_scope):
            self._build_embeddings()
            self._build_graph(defaults)

            self.global_step = tf.contrib.framework.get_global_step()
            if (self.mode == tf.estimator.ModeKeys.TRAIN or
                        self.mode == tf.estimator.ModeKeys.EVAL):
                self._build_loss()

    @property
    def decoder_hidden_units(self):
        if isinstance(self.cell.output_size, tuple):
            return self.cell.output_size[0]
        else:
            return self.cell.output_size

    @property
    def decoder_batch_size(self):
        if isinstance(self.encoder_state, tuple):
            real_cell = self.encoder_state[0]
        else:
            real_cell = self.encoder_state

        if isinstance(real_cell, LSTMStateTuple):
            batch_size, _ = tf.unstack(tf.shape(real_cell[0]))
            return batch_size
        else:
            batch_size, _ = tf.unstack(tf.shape(real_cell))
            return batch_size

    def _build_embeddings(self):
        if self.embedding_matrix is not None:
            self.vocab_size, self.embedding_size = self.embedding_matrix.get_shape().as_list()
        else:
            self.embedding_matrix = create_embedding_matrix(
                self.vocab_size, self.embedding_size)

    def _build_graph(self, defaults=None):
        if (self.mode == tf.estimator.ModeKeys.TRAIN or
                    self.mode == tf.estimator.ModeKeys.EVAL):
            if defaults is None:
                self.targets = tf.placeholder(
                    shape=(None, None),
                    dtype=tf.int32,
                    name="decoder_inputs")
                self.targets_length = tf.placeholder(
                    shape=(None,),
                    dtype=tf.int32,
                    name="decoder_inputs_length")
            else:
                self.targets = tf.placeholder_with_default(
                    defaults["targets"],
                    shape=(None, None),
                    name="decoder_inputs")
                self.targets_length = tf.placeholder_with_default(
                    defaults["targets_length"],
                    shape=(None,),
                    name="decoder_inputs_length")

            with tf.name_scope("DecoderTrainFeed"):
                self.train_inputs, self.train_targets, self.train_length, self.loss_weights = \
                    dynamic_targets(
                        self.targets,
                        self.targets_length,
                        pad_token=self.PAD,
                        end_token=self.EOS)

            with tf.variable_scope("embedding"):
                self.inputs_embedded = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.train_inputs)

        with tf.variable_scope("Decoder"):
            if self.output_layer is None:
                self.output_layer = Dense(
                    self.vocab_size,
                    name="output_layer")

            if (self.mode == tf.estimator.ModeKeys.TRAIN or
                        self.mode == tf.estimator.ModeKeys.EVAL):
                ((self.train_outputs, self.train_sampled_ids),
                 self.train_state, self.train_lengths) = dynamic_rnn_decode(
                    mode=self.mode,
                    decode_mode=self.training_mode,
                    cell=self.cell,
                    initial_state=self.encoder_state,
                    embeddings=self.embedding_matrix,
                    output_layer=self.output_layer,
                    scheduled_sampling_probability=self.scheduled_sampling_probability,
                    inputs=self.inputs_embedded,
                    inputs_length=self.train_length,
                    attention=self.attention,
                    attention_num_units=self.decoder_hidden_units,
                    attention_memory=self.encoder_outputs,
                    attention_memory_sequence_length=self.encoder_inputs_length,
                    attention_layer_size=self.decoder_hidden_units)
                self.train_logits = self.train_outputs
                self.train_prediction = self.train_sampled_ids
            elif self.mode == tf.estimator.ModeKeys.PREDICT:
                (final_outputs, self.inference_state,
                 self.inference_lengths) = dynamic_rnn_decode(
                    mode=self.mode,
                    decode_mode=self.inference_mode,
                    cell=self.cell,
                    initial_state=self.encoder_state,
                    embeddings=self.embedding_matrix,
                    output_layer=self.output_layer,
                    maximum_length=self.maximum_length,
                    attention=self.attention,
                    attention_num_units=self.decoder_hidden_units,
                    attention_memory=self.encoder_outputs,
                    attention_memory_sequence_length=self.encoder_inputs_length,
                    attention_layer_size=self.decoder_hidden_units,
                    start_token=self.EOS,
                    end_token=self.EOS,
                    beam_width=self.beam_width)

                if self.inference_mode == "greedy":
                    (self.inference_outputs, self.inference_sampled_ids) = \
                        (final_outputs.rnn_output, final_outputs.sample_id)
                    self.inference_scores = tf.ones(shape=[self.decoder_batch_size, 1])
                    self.inference_sampled_ids = tf.expand_dims(self.inference_sampled_ids, -1)
                elif self.inference_mode == "beam":
                    (self.inference_outputs, self.inference_sampled_ids) = \
                        (final_outputs.beam_search_decoder_output,
                         final_outputs.predicted_ids)
                    self.inference_scores = self.inference_outputs.scores
                self.inference_logits = self.inference_outputs
                # [batch_size, time_len, beam_size]
                self.inference_prediction = self.inference_sampled_ids
            else:
                raise NotImplementedError()

    def _build_loss(self):
        self.unreg_loss = self.loss = seq2seq.sequence_loss(
            logits=self.train_logits, targets=self.train_targets,
            weights=self.loss_weights)
