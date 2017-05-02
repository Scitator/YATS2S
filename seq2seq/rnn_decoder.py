import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib import seq2seq
from seq2seq.embeddings import create_embedding_matrix


class DynamicRnnDecoder(object):
    def __init__(self, cell, encoder_state, encoder_outputs, encoder_inputs_length,
                 attention=False, training_mode="greedy", decoding_mode="basic",
                 embedding_matrix=None, vocab_size=None, embedding_size=None,
                 special=None):
        assert embedding_matrix is not None \
               or (vocab_size is not None and embedding_size is not None)
        # @TODO: should work without all encoder stuff ?
        if embedding_matrix is not None:
            self.vocab_size, self.embedding_size = embedding_matrix.get_shape().as_list()
            self.embedding_matrix = embedding_matrix
        else:
            self.vocab_size = vocab_size
            self.embedding_size = embedding_size
            self.embedding_matrix = create_embedding_matrix(
                self.vocab_size, self.embedding_size)

        self.loss = None
        self.optimizer = None
        self.train_op = None
        self.cell = cell
        self.encoder_state = encoder_state

        self.training_mode = training_mode
        self.decoding_mode = decoding_mode

        # @TODO: should be optimal
        self.encoder_outputs = encoder_outputs
        self.encoder_inputs_length = encoder_inputs_length
        self.attention = attention

        self.special = special or {}
        self.PAD = self.special.get("PAD", 0)
        self.EOS = self.special.get("EOS", 1)

        self.scope = self.special.get("scope", "DynamicRnnDecoder")
        self.reuse_scope = self.special.get("reuse_scope", False)
        with tf.variable_scope(self.scope, self.reuse_scope):
            self._build_graph()
            self._build_loss()

    @property
    def decoder_hidden_units(self):
        if isinstance(self.cell.output_size, tuple):
            return self.cell.output_size[0]  # LSTM support? need to test
        else:
            return self.cell.output_size

    def _build_graph(self):
        # required only for training
        self.targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name="decoder_inputs")
        self.targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name="decoder_inputs_length")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        with tf.name_scope("DecoderTrainFeed"):
            sequence_size, batch_size = tf.unstack(tf.shape(self.targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.train_inputs = tf.concat([EOS_SLICE, self.targets], axis=0)
            self.train_length = self.targets_length + 1

            train_targets = tf.concat([self.targets, PAD_SLICE], axis=0)
            train_targets_seq_len, _ = tf.unstack(tf.shape(train_targets))
            train_targets_eos_mask = tf.one_hot(
                self.train_length - 1,
                train_targets_seq_len,
                on_value=self.EOS, off_value=self.PAD,
                dtype=tf.int32)
            train_targets_eos_mask = tf.transpose(train_targets_eos_mask, [1, 0])

            # hacky way using one_hot to put EOS symbol at the end of target sequence
            train_targets = tf.add(
                train_targets, train_targets_eos_mask)

            self.train_targets = train_targets

            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.train_length)],
                dtype=tf.float32, name="loss_weights")

        with tf.variable_scope("embedding") as scope:
            self.inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.train_inputs)

        with tf.variable_scope("Decoder") as scope:

            def logits_fn(outputs):
                return layers.dense(outputs, self.vocab_size, name="logits_fn")

            decoder_output_layer = None
            # layers.Dense(self.vocab_size, name="decoder_output_layer")

            if self.attention:
                # attention_states: size [batch_size, max_time, num_units]
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

                create_attention_mechanism = seq2seq.BahdanauAttention

                attention_mechanism = create_attention_mechanism(
                    num_units=self.decoder_hidden_units,
                    memory=attention_states,
                    memory_sequence_length=self.encoder_inputs_length)  # @TODO: needed?

                self.cell = seq2seq.DynamicAttentionWrapper(
                    cell=self.cell,
                    attention_mechanism=attention_mechanism,
                    attention_size=self.decoder_hidden_units)  # @TODO: attention size?

                # @TODO: hack or not a hack? why this attention size?
                self.encoder_state = seq2seq.DynamicAttentionWrapperState(
                    cell_state=self.encoder_state,
                    attention=tf.zeros(
                        shape=(batch_size, self.decoder_hidden_units),
                        dtype=tf.float32))

            if self.training_mode == "greedy":
                train_helper = seq2seq.TrainingHelper(
                    inputs=self.inputs_embedded,
                    sequence_length=self.train_length,
                    time_major=True)
            elif self.training_mode == "scheduled_sampling_embedding":
                self.scheduled_sampling_probability = tf.placeholder(dtype=tf.float32, shape=())
                train_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs=self.inputs_embedded,
                    sequence_length=self.train_length,
                    embedding=self.embedding_matrix,
                    sampling_probability=self.scheduled_sampling_probability,
                    time_major=True)
            elif self.training_mode == "scheduled_sampling_output":
                # @TODO: what the difference?
                self.scheduled_sampling_probability = tf.placeholder(dtype=tf.float32, shape=())
                train_helper = seq2seq.ScheduledOutputTrainingHelper(
                    inputs=self.inputs_embedded,
                    sequence_length=self.train_length,
                    sampling_probability=self.scheduled_sampling_probability,
                    time_major=True)
            else:
                raise NotImplemented()

            inference_helper = seq2seq.GreedyEmbeddingHelper(
                embedding=self.embedding_matrix,
                start_tokens=tf.ones([batch_size], dtype=tf.int32) * self.EOS,
                end_token=self.EOS)

            train_decoder = seq2seq.BasicDecoder(
                cell=self.cell,
                helper=train_helper,
                initial_state=self.encoder_state,
                output_layer=decoder_output_layer)

            inference_decoder = seq2seq.BasicDecoder(
                cell=self.cell,
                helper=inference_helper,
                initial_state=self.encoder_state,
                output_layer=decoder_output_layer)

            # @TODO: undocumented, need to check, what is sampled ids?
            ((self.train_outputs, self.train_sampled_ids), self.train_state) = \
                seq2seq.dynamic_decode(
                    decoder=train_decoder,
                    output_time_major=True)
            self.train_logits = logits_fn(self.train_outputs)

            self.train_prediction = tf.argmax(
                self.train_logits, axis=-1,
                name="train_prediction")
            self.train_prediction_probabilities = tf.nn.softmax(
                self.train_logits, dim=-1,
                name="train_prediction_probabilities")

            scope.reuse_variables()

            ((self.inference_outputs, self.inference_sampled_ids), self.inference_state) = \
                seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=True)
            self.inference_logits = logits_fn(self.inference_outputs)

            self.inference_prediction = tf.argmax(
                self.inference_logits, axis=-1,
                name="inference_prediction")
            self.inference_prediction_probabilities = tf.nn.softmax(
                self.train_logits, dim=-1,
                name="inference_prediction_probabilities")

    def _build_loss(self):
        self.train_logits_seq = tf.transpose(self.train_logits, [1, 0, 2])
        self.train_targets_seq = tf.transpose(self.train_targets, [1, 0])
        self.unreg_loss = self.loss = seq2seq.sequence_loss(
            logits=self.train_logits_seq, targets=self.train_targets_seq,
            weights=self.loss_weights)
