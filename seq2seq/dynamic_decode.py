import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.contrib.rnn import LSTMStateTuple


attention_dct = {
    "bahdanau": seq2seq.BahdanauAttention,
    "luong": seq2seq.LuongAttention,
    True: seq2seq.BahdanauAttention
}


def state_batch_size(state):
    if isinstance(state, tuple):
        real_cell = state[0]
    else:
        real_cell = state

    if isinstance(real_cell, LSTMStateTuple):
        batch_size, _ = tf.unstack(tf.shape(real_cell[0]))
        return batch_size
    else:
        batch_size, _ = tf.unstack(tf.shape(real_cell))
        return batch_size


def dynamic_targets(targets, targets_length, pad_token, end_token):
    target_batch_size, _ = tf.unstack(tf.shape(targets))

    EOS_SLICE = tf.ones([target_batch_size, 1], dtype=tf.int32) * end_token
    PAD_SLICE = tf.ones([target_batch_size, 1], dtype=tf.int32) * pad_token
    train_inputs = tf.concat([EOS_SLICE, targets], axis=1)
    train_targets = tf.concat([targets, PAD_SLICE], axis=1)

    train_length = targets_length + 1
    _, train_targets_seq_len = tf.unstack(tf.shape(train_targets))

    train_targets_eos_mask = tf.one_hot(
        train_length - 1,
        train_targets_seq_len,
        on_value=end_token,
        off_value=pad_token,
        dtype=tf.int32)

    train_targets = tf.add(train_targets, train_targets_eos_mask)

    loss_weights = tf.sequence_mask(
        train_length,
        dtype=tf.float32, name="loss_weights")

    return train_inputs, train_targets, train_length, loss_weights


def dynamic_rnn_decode(
        mode, decode_mode, cell, initial_state, embeddings,
        inputs=None, inputs_length=None, scheduled_sampling_probability=None,  # only for training
        output_layer=None, maximum_length=None,
        attention=False, attention_num_units=None, attention_memory=None,
        attention_memory_sequence_length=None, attention_layer_size=None,
        start_token=None, end_token=None, beam_width=None):  # only for inference
    batch_size = state_batch_size(initial_state)

    if attention:
        if isinstance(attention, str):
            attention = attention.lower()
        create_attention_mechanism = attention_dct[attention]

    if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
        if attention:
            inference_attention_mechanism = create_attention_mechanism(
                num_units=attention_num_units,
                memory=attention_memory,
                memory_sequence_length=attention_memory_sequence_length)

            # @TODO: alignment_history?
            inference_cell = seq2seq.AttentionWrapper(
                cell=cell,
                attention_mechanism=inference_attention_mechanism,
                attention_layer_size=attention_layer_size)

            inference_initial_state = inference_cell.zero_state(
                dtype=tf.float32, batch_size=batch_size)
            inference_initial_state = inference_initial_state.clone(
                cell_state=initial_state)
        else:
            inference_cell = cell
            inference_initial_state = initial_state

        if decode_mode == "greedy":
            train_helper = seq2seq.TrainingHelper(
                inputs=inputs,
                sequence_length=inputs_length,
                time_major=False)
        elif decode_mode == "scheduled_sampling_embedding":
            train_helper = seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=inputs,
                sequence_length=inputs_length,
                embedding=embeddings,
                sampling_probability=scheduled_sampling_probability,
                time_major=False)
        elif decode_mode == "scheduled_sampling_output":
            train_helper = seq2seq.ScheduledOutputTrainingHelper(
                inputs=inputs,
                sequence_length=inputs_length,
                sampling_probability=scheduled_sampling_probability,
                time_major=False)
        else:
            raise NotImplementedError()

        train_decoder = seq2seq.BasicDecoder(
            cell=inference_cell,
            helper=train_helper,
            initial_state=inference_initial_state,
            output_layer=output_layer)

        result = seq2seq.dynamic_decode(
            decoder=train_decoder,
            output_time_major=False,
            maximum_iterations=maximum_length)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        if decode_mode == "greedy":
            if attention:
                inference_attention_mechanism = create_attention_mechanism(
                    num_units=attention_num_units,
                    memory=attention_memory,
                    memory_sequence_length=attention_memory_sequence_length)

                inference_cell = seq2seq.AttentionWrapper(
                    cell=cell,
                    attention_mechanism=inference_attention_mechanism,
                    attention_layer_size=attention_layer_size)

                inference_initial_state = inference_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size)
                inference_initial_state = inference_initial_state.clone(
                    cell_state=initial_state)
            else:
                inference_cell = cell
                inference_initial_state = initial_state

            inference_helper = seq2seq.GreedyEmbeddingHelper(
                embedding=embeddings,
                start_tokens=tf.ones([batch_size], dtype=tf.int32) * start_token,
                end_token=end_token)

            inference_decoder = seq2seq.BasicDecoder(
                cell=inference_cell,
                helper=inference_helper,
                initial_state=inference_initial_state,
                output_layer=output_layer)
        elif decode_mode == "beam":
            if isinstance(initial_state, LSTMStateTuple):
                inference_initial_state = LSTMStateTuple(
                    seq2seq.tile_batch(initial_state[0], multiplier=beam_width),
                    seq2seq.tile_batch(initial_state[1], multiplier=beam_width))
            elif isinstance(initial_state, tuple) \
                    and isinstance(initial_state[0], LSTMStateTuple):
                inference_initial_state = tuple(map(
                    lambda state: LSTMStateTuple(
                        seq2seq.tile_batch(state[0], multiplier=beam_width),
                        seq2seq.tile_batch(state[1], multiplier=beam_width)),
                    initial_state))
            else:
                inference_initial_state = seq2seq.tile_batch(
                    initial_state, multiplier=beam_width)

            if attention:
                attention_states = seq2seq.tile_batch(
                    attention_memory, multiplier=beam_width)

                attention_memory_length = seq2seq.tile_batch(
                    attention_memory_sequence_length, multiplier=beam_width)

                inference_attention_mechanism = create_attention_mechanism(
                    num_units=attention_num_units,
                    memory=attention_states,
                    memory_sequence_length=attention_memory_length)

                inference_cell = seq2seq.AttentionWrapper(
                    cell=cell,
                    attention_mechanism=inference_attention_mechanism,
                    attention_layer_size=attention_layer_size)

                zero_initial_state = inference_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size * beam_width)
                inference_initial_state = zero_initial_state.clone(
                    cell_state=inference_initial_state)
            else:
                inference_cell = cell

            inference_decoder = seq2seq.BeamSearchDecoder(
                cell=inference_cell,
                embedding=embeddings,
                start_tokens=[start_token] * batch_size * beam_width,
                end_token=end_token,
                initial_state=inference_initial_state,
                output_layer=output_layer,
                beam_width=beam_width)
        else:
            raise NotImplementedError()

        result = seq2seq.dynamic_decode(
            decoder=inference_decoder,
            output_time_major=False,
            maximum_iterations=maximum_length)
    else:
        raise NotImplementedError()

    return result