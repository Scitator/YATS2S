import tensorflow as tf
from seq2seq.embeddings import Embeddings
from seq2seq.rnn_encoder import DynamicRnnEncoder
from seq2seq.rnn_decoder import DynamicRnnDecoder
from seq2seq.training.utils import get_rnn_cell


def build_model_optimization(model, optimization_args=None, loss=None):
    loss = model.loss if model.loss is not None else loss
    assert loss is not None
    optimization_args = optimization_args or {}

    learning_rate_decay_fn = lambda learning_rate, global_step: \
        tf.train.exponential_decay(
            learning_rate, global_step,
            decay_steps=optimization_args.get("decay_steps", 100000),
            decay_rate=optimization_args.get("decay_rate", 0.99))

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
                 mode=tf.estimator.ModeKeys.TRAIN):
        same_embeddings = (isinstance(vocab_size, int) and isinstance(embedding_size, int))
        different_embeddigns = (isinstance(vocab_size, tuple) and isinstance(embedding_size, tuple))
        assert (same_embeddings or different_embeddigns)

        if same_embeddings:
            self.embeddings = Embeddings(
                vocab_size,
                embedding_size,
                special={"scope": "embeddings"})
        elif different_embeddigns:
            # @TODO: projection ?
            self.embeddings_from = Embeddings(
                vocab_size[0],
                embedding_size[0],
                special={"scope": "embeddings_from"})
            self.embeddings_to = Embeddings(
                vocab_size[1],
                embedding_size[1],
                special={"scope": "embeddings_to"})
        else:
            raise NotImplementedError()

        self.encoder = DynamicRnnEncoder(
            embedding_matrix=self.embeddings.embedding_matrix
            if same_embeddings
            else self.embeddings_from.embedding_matrix,
            **encoder_args)

        self.decoder = DynamicRnnDecoder(
            encoder_state=self.encoder.state,
            encoder_outputs=self.encoder.outputs,
            embedding_matrix=self.embeddings.embedding_matrix
            if same_embeddings
            else self.embeddings_to.embedding_matrix,
            encoder_inputs_length=self.encoder.inputs_length,
            mode=mode,
            **decoder_args)

        if mode == tf.estimator.ModeKeys.TRAIN:
            build_model_optimization(
                self.encoder, encoder_optimization_args, self.decoder.loss)
            build_model_optimization(
                self.decoder, decoder_optimization_args)
            if same_embeddings:
                build_model_optimization(
                    self.embeddings, embeddings_optimization_args, self.decoder.loss)
                self.train_op = tf.group(
                    self.embeddings.train_op,
                    self.encoder.train_op,
                    self.decoder.train_op)
            else:
                build_model_optimization(
                    self.embeddings_from, embeddings_optimization_args, self.decoder.loss)
                build_model_optimization(
                    self.embeddings_to, embeddings_optimization_args, self.decoder.loss)
                self.train_op = tf.group(
                    self.embeddings_from.train_op,
                    self.embeddings_to.train_op,
                    self.encoder.train_op,
                    self.decoder.train_op)


def seq2seq_model(features, labels, mode, params, config):
    encoder_cell_params = dict(
        cell_class=params.cell,
        cell_params={"num_units": params.num_units},
        num_layers=params.num_layers,
        residual_connections=params.residual_connections,
        residual_dense=params.residual_dense)

    decoder_cell_params = dict(
        cell_class=params.cell,
        cell_params={"num_units": params.num_units + params.num_units * int(params.bidirectional)},
        num_layers=params.num_layers,
        residual_connections=params.residual_connections,
        residual_dense=params.residual_dense)

    if params.cell_num == 1:
        assert not params.bidirectional
        encoder_cell = decoder_cell = get_rnn_cell(**encoder_cell_params)
    elif params.cell_num == 2:
        encoder_cell = get_rnn_cell(**encoder_cell_params)
        decoder_cell = get_rnn_cell(**decoder_cell_params)
    elif params.cell_num == 3:
        assert params.bidirectional
        encoder_cell = (
            get_rnn_cell(**encoder_cell_params),
            get_rnn_cell(**encoder_cell_params))
        decoder_cell = get_rnn_cell(**decoder_cell_params)
    else:
        raise NotImplementedError()

    encoder_args = {
        "cell": encoder_cell,
        "bidirectional": params.bidirectional,
        "defaults": features
    }

    decoder_args = {
        "cell": decoder_cell,
        "attention": params.attention,
        "defaults": labels
    }

    if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
        decoder_args["training_mode"] = params.training_mode
        if "scheduled_sampling" in params.training_mode:
            decoder_args["scheduled_sampling_probability"] = params.scheduled_sampling_probability

    if mode == tf.estimator.ModeKeys.PREDICT:
        decoder_args["inference_mode"] = params.inference_mode
        if params.inference_mode == "beam":
            decoder_args["beam_width"] = params.beam_width

    model = DynamicSeq2Seq(
        params.vocab_size,
        params.embedding_size,
        encoder_args=encoder_args,
        decoder_args=decoder_args,
        mode=mode)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = model.train_op
    else:
        train_op = None

    if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
        loss = model.decoder.loss
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            # "inputs": model.encoder.inputs,
            "prediction": model.decoder.inference_prediction,
            "score": model.decoder.inference_scores
        }
    else:
        predictions = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)


def create_seq2seq_model(config=None, hparams=None):
    return tf.estimator.Estimator(
        model_fn=seq2seq_model,
        config=config,
        params=hparams)


def create_seq2seq_experiment(
        train_input_fn, val_input_fn,
        train_steps, eval_steps, min_eval_frequency):

    def seq2seq_experiment(run_config, hparams):
        return tf.contrib.learn.Experiment(
            estimator=create_seq2seq_model(
                config=run_config,
                hparams=hparams),
            train_input_fn=train_input_fn,
            eval_input_fn=val_input_fn,
            train_steps=train_steps,
            eval_steps=eval_steps,
            min_eval_frequency=min_eval_frequency)

    return seq2seq_experiment
