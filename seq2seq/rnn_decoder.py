import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow import layers
from tensorflow.contrib import seq2seq
from seq2seq.embeddings import create_embedding_matrix
from tensorflow.contrib.rnn import LSTMStateTuple


class DynamicRnnDecoder(object):
    def __init__(self, cell, encoder_state, encoder_outputs, maximum_length=150,
                 attention=False, training_mode="greedy", decoding_mode="greedy", beam_width=5,
                 embedding_matrix=None, vocab_size=None, embedding_size=None,
                 special=None, defaults=None):
        assert embedding_matrix is not None \
               or (vocab_size is not None and embedding_size is not None)
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.loss = None
        self.train_op = None
        self.cell = cell
        self.encoder_state = encoder_state

        self.training_mode = training_mode
        self.decoding_mode = decoding_mode
        self.beam_width = beam_width

        # @TODO: should be optimal
        self.encoder_outputs = encoder_outputs
        self.maximum_length = maximum_length
        self.attention = attention

        self.special = special or {}
        self.PAD = self.special.get("PAD", 0)
        self.EOS = self.special.get("EOS", 1)

        self.scope = self.special.get("scope", "DynamicRnnDecoder")
        self.reuse_scope = self.special.get("reuse_scope", False)
        with tf.variable_scope(self.scope, self.reuse_scope):
            self._build_embeddings()
            self._build_graph(defaults)

            self.global_step = tf.get_variable(
                "global_step", [],
                trainable=False,
                dtype=tf.int64,
                initializer=tf.constant_initializer(
                    0, dtype=tf.int64))

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
        # required only for training
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
            # self.targets = tf.placeholder_with_default(
            #     defaults["targets"],
            #     shape=(None, None),
            #     name="decoder_inputs")
            self.targets = defaults["targets"]
            self.targets = tf.transpose(self.targets, [1, 0])
            # self.targets_length = tf.placeholder_with_default(
            #     defaults["targets_length"],
            #     shape=(None,),
            #     name="decoder_inputs_length")
            self.targets_length = defaults["targets_length"]

        with tf.name_scope("DecoderTrainFeed"):
            target_sequence_size, target_batch_size = tf.unstack(tf.shape(self.targets))

            EOS_SLICE = tf.ones([1, target_batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, target_batch_size], dtype=tf.int32) * self.PAD
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
                target_batch_size,
                tf.reduce_max(self.train_length)],
                dtype=tf.float32, name="loss_weights")

        with tf.variable_scope("embedding") as scope:
            self.inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.train_inputs)

        with tf.variable_scope("Decoder") as scope:
            # def logits_fn(outputs):
                # return layers.dense(outputs, self.vocab_size, name="logits_fn")

            # decoder_output_layer = None
            decoder_output_layer = Dense(
                self.vocab_size,
                name="decoder_output_layer")

            if self.attention:
                # attention_states: size [batch_size, max_time, num_units]
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
                create_attention_mechanism = seq2seq.BahdanauAttention

            if self.decoding_mode == "greedy":
                inputs_embedded = self.inputs_embedded
                train_length = self.train_length

                if self.attention:
                    inference_attention_mechanism = create_attention_mechanism(
                        num_units=self.decoder_hidden_units,
                        memory=attention_states)
                        # memory_sequence_length=self.maximum_length)  # @TODO: needed?

                    inference_cell = seq2seq.AttentionWrapper(
                        cell=self.cell,
                        attention_mechanism=inference_attention_mechanism,
                        attention_layer_size=self.decoder_hidden_units)  # @TODO: attention size?

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
            elif self.decoding_mode == "beam":
                beam_width = self.beam_width  # @TODO: need to refactor
                # just...I don't know
                inputs_embedded = tf.transpose(self.inputs_embedded, [1, 0, 2])
                inputs_embedded = seq2seq.tile_batch(inputs_embedded, multiplier=beam_width)
                inputs_embedded = tf.transpose(inputs_embedded, [1, 0, 2])

                train_length = seq2seq.tile_batch(self.train_length, multiplier=beam_width)

                if isinstance(self.encoder_state, LSTMStateTuple):
                    inference_initial_state = LSTMStateTuple(
                        seq2seq.tile_batch(self.encoder_state[0], multiplier=beam_width),
                        seq2seq.tile_batch(self.encoder_state[1], multiplier=beam_width))
                else:
                    inference_initial_state = seq2seq.tile_batch(
                        self.encoder_state, multiplier=beam_width)

                if self.attention:
                    attention_states = seq2seq.tile_batch(attention_states, multiplier=beam_width)
                    # beam_sequence_length = seq2seq.tile_batch(
                    #     self.maximum_length, multiplier=beam_width)

                    inference_attention_mechanism = create_attention_mechanism(
                        num_units=self.decoder_hidden_units,
                        memory=attention_states)
                        # memory_sequence_length=beam_sequence_length)

                    inference_cell = seq2seq.AttentionWrapper(
                        cell=self.cell,
                        attention_mechanism=inference_attention_mechanism,
                        attention_layer_size=self.decoder_hidden_units)

                    # @TODO: bad code, need renaming
                    zero_initial_state = inference_cell.zero_state(
                        dtype=tf.float32, batch_size=self.decoder_batch_size * beam_width)
                    inference_initial_state = zero_initial_state.clone(
                        cell_state=inference_initial_state)
                else:
                    inference_cell = self.cell

                inference_decoder = seq2seq.BeamSearchDecoder(
                    cell=inference_cell,
                    embedding=self.embedding_matrix,
                    start_tokens=[self.EOS] * self.decoder_batch_size * beam_width,
                    end_token=self.EOS,
                    initial_state=inference_initial_state,
                    beam_width=beam_width)

            if self.training_mode == "greedy":
                train_helper = seq2seq.TrainingHelper(
                    inputs=inputs_embedded,
                    sequence_length=train_length,
                    time_major=True)
            elif self.training_mode == "scheduled_sampling_embedding":
                self.scheduled_sampling_probability = tf.placeholder(dtype=tf.float32, shape=())
                train_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs=inputs_embedded,
                    sequence_length=train_length,
                    embedding=self.embedding_matrix,
                    sampling_probability=self.scheduled_sampling_probability,
                    time_major=True)
            elif self.training_mode == "scheduled_sampling_output":
                self.scheduled_sampling_probability = tf.placeholder(dtype=tf.float32, shape=())
                train_helper = seq2seq.ScheduledOutputTrainingHelper(
                    inputs=inputs_embedded,
                    sequence_length=train_length,
                    sampling_probability=self.scheduled_sampling_probability,
                    time_major=True)
            else:
                raise NotImplemented()
            
            train_decoder = seq2seq.BasicDecoder(
                cell=inference_cell,
                helper=train_helper,
                initial_state=inference_initial_state,
                output_layer=decoder_output_layer)

            ((self.train_outputs, self.train_sampled_ids),
                self.train_state, self.train_lengths) = \
                seq2seq.dynamic_decode(
                    decoder=train_decoder,
                    output_time_major=True)
            self.train_logits = self.train_outputs
            self.train_prediction = self.train_sampled_ids

            scope.reuse_variables()

            (final_outputs, self.inference_state, self.inference_lengths) = \
                seq2seq.dynamic_decode(
                    decoder=inference_decoder,
                    output_time_major=True,
                    maximum_iterations=self.maximum_length,
                    scope="inference_decoder")

            if self.decoding_mode == "greedy":
                (self.inference_outputs, self.inference_sampled_ids) = \
                    (final_outputs.rnn_output, final_outputs.sample_id)
                self.inference_scores = tf.ones(shape=[self.decoder_batch_size, 1])
            elif self.decoding_mode == "beam":
                (self.inference_outputs, self.inference_sampled_ids) = \
                    (final_outputs.beam_search_decoder_output,
                     final_outputs.predicted_ids)
                self.inference_scores = tf.squeeze(self.inference_outputs.scores, axis=1)
                self.inference_sampled_ids = tf.squeeze(self.inference_sampled_ids, axis=1)
            self.inference_logits = self.inference_outputs
            self.inference_prediction = self.inference_sampled_ids

    def _build_loss(self):
        self.train_logits_seq = tf.transpose(self.train_logits, [1, 0, 2])
        self.train_targets_seq = tf.transpose(self.train_targets, [1, 0])
        if self.decoding_mode == "beam":
            self.train_targets_seq = seq2seq.tile_batch(
                self.train_targets_seq, 
                multiplier=self.beam_width)
            self.loss_weights = seq2seq.tile_batch(
                self.loss_weights, 
                multiplier=self.beam_width)
        self.unreg_loss = self.loss = seq2seq.sequence_loss(
                logits=self.train_logits_seq, targets=self.train_targets_seq,
                weights=self.loss_weights)
