import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
import os
from tqdm import trange, tqdm


def create_if_need(path):
    if not os.path.exists(path):
        os.makedirs(path)


def update_varlist(loss, optimizer, var_list, grad_clip=5.0, global_step=None):
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    capped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs]
    update_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    return update_step


def build_optimization(model, optimization_params=None, loss=None):
    optimization_params = optimization_params or {}

    initial_lr = optimization_params.get("initial_lr", 1e-4)
    decay_steps = int(optimization_params.get("decay_steps", 100000))
    lr_decay = optimization_params.get("lr_decay", 0.999)
    grad_clip = optimization_params.get("grad_clip", 10.0)

    lr = tf.train.exponential_decay(
        initial_lr,
        model.global_step,
        decay_steps,
        lr_decay,
        staircase=True)

    model.loss = model.loss or loss
    model.optimizer = tf.train.AdamOptimizer(lr)

    model.train_op = update_varlist(
        model.loss, model.optimizer,
        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model.scope),
        grad_clip=grad_clip,
        global_step=model.global_step)


def time_major_batch(inputs, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used

    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active
            time steps in each input sequence
    """

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_time_major, sequence_lengths


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
    def __init__(self, vocab_size, embedding_size, scope):
        self.loss = None
        self.optimizer = None
        self.train_op = None

        self.scope = scope
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.embedding_matrix = create_embedding_matrix(
            self.vocab_size, self.embedding_size, scope=self.scope)


class DynamicRnnEncoder(object):
    def __init__(self, cell, bidirectional=True,
                 embedding_matrix=None, vocab_size=None, embedding_size=None,
                 special=None):
        assert embedding_matrix is not None \
               or (vocab_size is not None and embedding_size is not None)
        if embedding_matrix is not None:
            # @TODO: add vocab_size and embe_size unpack
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
        self.bidirectional = bidirectional
        self.special = (special or {})
        self.PAD = self.special.get("PAD", 0)
        self.EOS = self.special.get("EOS", 1)

        self.scope = special.get("scope", "DynamicRnnEncoder")
        self.reuse_scope = special.get("reuse_scope", False)
        with tf.variable_scope(self.scope, self.reuse_scope):
            self._build_graph()

    def _build_graph(self):
        self.inputs = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='encoder_inputs')
        self.inputs_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='encoder_inputs_length')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope("embedding") as scope:
            self.inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.inputs)

        if self.bidirectional:
            with tf.variable_scope("BidirectionalEncoder") as scope:
                ((encoder_fw_outputs,
                  encoder_bw_outputs),
                 (encoder_fw_state,
                  encoder_bw_state)) = (
                    tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=self.cell,
                        cell_bw=self.cell,
                        inputs=self.inputs_embedded,
                        sequence_length=self.inputs_length,
                        time_major=True,
                        dtype=tf.float32)
                )

                self.outputs = tf.concat(
                    (encoder_fw_outputs, encoder_bw_outputs), 2,
                    name='bidirectional_output_concat')
                self.state = tf.concat(
                    (encoder_fw_state, encoder_bw_state), 1,
                    name='bidirectional_state_concat')
        else:
            with tf.variable_scope("Encoder") as scope:
                self.outputs, self.state = \
                    tf.nn.dynamic_rnn(
                        cell=self.cell,
                        inputs=self.inputs_embedded,
                        sequence_length=self.inputs_length,
                        time_major=True,
                        dtype=tf.float32)


class DynamicRnnDecoder(object):
    def __init__(self, cell, encoder_state, encoder_outputs, encoder_inputs_length,
                 attention=True,
                 embedding_matrix=None, vocab_size=None, embedding_size=None,
                 special=None):
        assert embedding_matrix is not None \
               or (vocab_size is not None and embedding_size is not None)
        if embedding_matrix is not None:
            # @TODO: add vocab_size and embe_size unpack
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
        self.encoder_output = encoder_outputs
        self.encoder_inputs_length = encoder_inputs_length
        self.attention = attention
        self.special = (special or {})
        self.PAD = self.special.get("PAD", 0)
        self.EOS = self.special.get("EOS", 1)

        self.scope = special.get("scope", "DynamicRnnDecoder")
        self.reuse_scope = special.get("reuse_scope", False)
        with tf.variable_scope(self.scope, self.reuse_scope):
            self._build_graph()
            self._build_loss()

    @property
    def decoder_hidden_units(self):
        return self.cell.output_size

    def _build_graph(self):
        # required only for training
        self.targets = tf.placeholder(
            shape=(None, None),
            dtype=tf.int32,
            name='decoder_inputs')
        self.targets_length = tf.placeholder(
            shape=(None,),
            dtype=tf.int32,
            name='decoder_inputs_length')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        with tf.variable_scope("embedding") as scope:
            self.inputs_embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, self.targets)

        with tf.name_scope('DecoderTrainFeeds'):
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

        with tf.variable_scope("Decoder") as scope:

            def output_fn(outputs):
                return layers.linear(outputs, self.vocab_size, scope=scope)

            if not self.attention:
                train_fn = seq2seq.simple_decoder_fn_train(
                    encoder_state=self.encoder_state)
                inference_fn = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size)
            else:

                # attention_states: size [batch_size, max_time, num_units]
                attention_states = tf.transpose(self.encoder_output, [1, 0, 2])

                (attention_keys,
                 attention_values,
                 attention_score_fn,
                 attention_construct_fn) = seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units,
                )

                train_fn = seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='decoder_attention'
                )

                inference_fn = seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size,
)

            (self.train_outputs,
             self.train_state,
             self.train_context_state) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.cell,
                    decoder_fn=train_fn,
                    inputs=self.inputs_embedded,
                    sequence_length=self.train_length,
                    time_major=True,
                    scope=scope,
                )
            )

            self.train_logits = output_fn(self.train_outputs)
            self.train_prediction = tf.argmax(
                self.train_logits, axis=-1,
                name='train_prediction')

            scope.reuse_variables()

            (self.inference_logits,
             self.inference_state,
             self.inference_context_state) = (
                seq2seq.dynamic_rnn_decoder(
                    cell=self.cell,
                    decoder_fn=inference_fn,
                    time_major=True,
                    scope=scope,
                )
            )
            self.inference_prediction = tf.argmax(
                self.inference_logits,
                axis=-1,
                name='inference_prediction')

    def _build_loss(self):
        self.train_logits_seq = tf.transpose(self.train_logits, [1, 0, 2])
        self.train_targets_seq = tf.transpose(self.train_targets, [1, 0])
        self.unreg_loss = self.loss = seq2seq.sequence_loss(
            logits=self.train_logits_seq, targets=self.train_targets_seq,
            weights=self.loss_weights)


class DynamicSeq2Seq(object):
    def __init__(self,
                 encoder_args, decoder_args,
                 encoder_optimization_args, decoder_optimization_args):
        self.embeddings = Embeddings(
            encoder_args["vocab_size"],
            encoder_args["embedding_size"],
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
        build_optimization(self.embeddings, decoder_optimization_args, self.decoder.loss)
