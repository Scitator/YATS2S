import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import seq2seq
from seq2seq.embeddings import create_embedding_matrix
from tensorflow.contrib.rnn import LSTMStateTuple


class DynamicRnnDecoder(object):
    def __init__(self, cell, encoder_state, encoder_outputs, maximum_length=150,
                 attention=None, encoder_inputs_length=None,
                 training_mode="greedy", scheduled_sampling_probability=0.0,
                 inference_mode="greedy", beam_width=1,
                 embedding_matrix=None, vocab_size=None, embedding_size=None,
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
                target_batch_size, target_sequence_size = tf.unstack(tf.shape(self.targets))

                EOS_SLICE = tf.ones([target_batch_size, 1], dtype=tf.int32) * self.EOS
                PAD_SLICE = tf.ones([target_batch_size, 1], dtype=tf.int32) * self.PAD
                self.train_inputs = tf.concat([EOS_SLICE, self.targets], axis=1)
                self.train_length = self.targets_length + 1

                train_targets = tf.concat([self.targets, PAD_SLICE], axis=1)
                _, train_targets_seq_len = tf.unstack(tf.shape(train_targets))
                train_targets_eos_mask = tf.one_hot(
                    self.train_length - 1,
                    train_targets_seq_len,
                    on_value=self.EOS, off_value=self.PAD,
                    dtype=tf.int32)

                # hacky way using one_hot to put EOS symbol at the end of target sequence
                train_targets = tf.add(train_targets, train_targets_eos_mask)

                self.train_targets = train_targets
                # @TODO: make something interesting with sequential loss
                self.loss_weights = tf.ones([
                    target_batch_size,
                    tf.reduce_max(self.train_length)],
                    dtype=tf.float32, name="loss_weights")

            with tf.variable_scope("embedding"):
                self.inputs_embedded = tf.nn.embedding_lookup(
                    self.embedding_matrix, self.train_inputs)

        with tf.variable_scope("Decoder"):
            decoder_output_layer = Dense(
                self.vocab_size,
                name="decoder_output_layer")

            if self.attention:
                attention_dct = {
                    "bahdanau": seq2seq.BahdanauAttention,
                    "luong": seq2seq.LuongAttention,
                    True: seq2seq.BahdanauAttention
                }
                if isinstance(self.attention, str):
                    self.attention = self.attention.lower()
                create_attention_mechanism = attention_dct[self.attention]

            if (self.mode == tf.estimator.ModeKeys.TRAIN or
                        self.mode == tf.estimator.ModeKeys.EVAL):
                if self.attention:
                    inference_attention_mechanism = create_attention_mechanism(
                        num_units=self.decoder_hidden_units,
                        memory=self.encoder_outputs,
                        memory_sequence_length=self.encoder_inputs_length)

                    # @TODO: alignment_history?
                    inference_cell = seq2seq.AttentionWrapper(
                        cell=self.cell,
                        attention_mechanism=inference_attention_mechanism,
                        attention_layer_size=self.decoder_hidden_units)

                    inference_initial_state = inference_cell.zero_state(
                        dtype=tf.float32, batch_size=self.decoder_batch_size)
                    inference_initial_state = inference_initial_state.clone(
                        cell_state=self.encoder_state)
                else:
                    inference_cell = self.cell
                    inference_initial_state = self.encoder_state

                if self.training_mode == "greedy":
                    train_helper = seq2seq.TrainingHelper(
                        inputs=self.inputs_embedded,
                        sequence_length=self.train_length,
                        time_major=False)
                elif self.training_mode == "scheduled_sampling_embedding":
                    train_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
                        inputs=self.inputs_embedded,
                        sequence_length=self.train_length,
                        embedding=self.embedding_matrix,
                        sampling_probability=self.scheduled_sampling_probability,
                        time_major=False)
                elif self.training_mode == "scheduled_sampling_output":
                    train_helper = seq2seq.ScheduledOutputTrainingHelper(
                        inputs=self.inputs_embedded,
                        sequence_length=self.train_length,
                        sampling_probability=self.scheduled_sampling_probability,
                        time_major=False)
                else:
                    raise NotImplementedError()

                train_decoder = seq2seq.BasicDecoder(
                    cell=inference_cell,
                    helper=train_helper,
                    initial_state=inference_initial_state,
                    output_layer=decoder_output_layer)

                ((self.train_outputs, self.train_sampled_ids),
                 self.train_state, self.train_lengths) = \
                    seq2seq.dynamic_decode(
                        decoder=train_decoder,
                        output_time_major=False,
                        maximum_iterations=self.maximum_length)
                self.train_logits = self.train_outputs
                self.train_prediction = self.train_sampled_ids

            if self.mode == tf.estimator.ModeKeys.PREDICT:
                if self.inference_mode == "greedy":
                    if self.attention:
                        inference_attention_mechanism = create_attention_mechanism(
                            num_units=self.decoder_hidden_units,
                            memory=self.encoder_outputs,
                            memory_sequence_length=self.encoder_inputs_length)

                        inference_cell = seq2seq.AttentionWrapper(
                            cell=self.cell,
                            attention_mechanism=inference_attention_mechanism,
                            attention_layer_size=self.decoder_hidden_units)

                        inference_initial_state = inference_cell.zero_state(
                            dtype=tf.float32, batch_size=self.decoder_batch_size)
                        inference_initial_state = inference_initial_state.clone(
                            cell_state=self.encoder_state)
                    else:
                        inference_cell = self.cell
                        inference_initial_state = self.encoder_state

                    inference_helper = seq2seq.GreedyEmbeddingHelper(
                        embedding=self.embedding_matrix,
                        start_tokens=tf.ones([self.decoder_batch_size], dtype=tf.int32) * self.EOS,
                        end_token=self.EOS)

                    inference_decoder = seq2seq.BasicDecoder(
                        cell=inference_cell,
                        helper=inference_helper,
                        initial_state=inference_initial_state,
                        output_layer=decoder_output_layer)
                elif self.inference_mode == "beam":
                    if isinstance(self.encoder_state, LSTMStateTuple):
                        inference_initial_state = LSTMStateTuple(
                            seq2seq.tile_batch(self.encoder_state[0], multiplier=self.beam_width),
                            seq2seq.tile_batch(self.encoder_state[1], multiplier=self.beam_width))
                    elif isinstance(self.encoder_state, tuple) \
                        and isinstance(self.encoder_state[0], LSTMStateTuple):
                        inference_initial_state = tuple(map(
                            lambda state: LSTMStateTuple(
                                seq2seq.tile_batch(state[0], multiplier=self.beam_width),
                                seq2seq.tile_batch(state[1], multiplier=self.beam_width)),
                            self.encoder_state))
                    else:
                        inference_initial_state = seq2seq.tile_batch(
                            self.encoder_state, multiplier=self.beam_width)

                    if self.attention:
                        attention_states = seq2seq.tile_batch(
                            self.encoder_outputs, multiplier=self.beam_width)

                        attention_memory_length = seq2seq.tile_batch(
                            self.encoder_inputs_length, multiplier=self.beam_width)

                        inference_attention_mechanism = create_attention_mechanism(
                            num_units=self.decoder_hidden_units,
                            memory=attention_states,
                            memory_sequence_length=attention_memory_length)

                        inference_cell = seq2seq.AttentionWrapper(
                            cell=self.cell,
                            attention_mechanism=inference_attention_mechanism,
                            attention_layer_size=self.decoder_hidden_units)

                        zero_initial_state = inference_cell.zero_state(
                            dtype=tf.float32, batch_size=self.decoder_batch_size * self.beam_width)
                        inference_initial_state = zero_initial_state.clone(
                            cell_state=inference_initial_state)
                    else:
                        inference_cell = self.cell

                    inference_decoder = seq2seq.BeamSearchDecoder(
                        cell=inference_cell,
                        embedding=self.embedding_matrix,
                        start_tokens=[self.EOS] * self.decoder_batch_size * self.beam_width,
                        end_token=self.EOS,
                        initial_state=inference_initial_state,
                        output_layer=decoder_output_layer,
                        beam_width=self.beam_width)
                else:
                    raise NotImplementedError()

                (final_outputs, self.inference_state, self.inference_lengths) = \
                    seq2seq.dynamic_decode(
                        decoder=inference_decoder,
                        output_time_major=False,
                        maximum_iterations=self.maximum_length)

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

    def _build_loss(self):
        self.unreg_loss = self.loss = seq2seq.sequence_loss(
            logits=self.train_logits, targets=self.train_targets,
            weights=self.loss_weights)
