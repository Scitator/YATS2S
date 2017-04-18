from seq2seq.batch_utils import time_major_generator
from seq2seq.train import run_train


def train_seq2seq(sess, model, train_gen, val_gen=None, run_params=None):
    train_params = {
        "run_keys": [
            model.decoder.loss,
            model.decoder.train_op, model.encoder.train_op, model.embeddings.train_op],
        "result_keys": ["unreg_loss"],
        "feed_keys": [model.encoder.inputs, model.encoder.inputs_length,
                      model.decoder.targets, model.decoder.targets_length],
        "n_batch": 100,
    }
    train_gen = time_major_generator(train_gen)

    val_params = None
    if val_gen is not None:
        val_params = {
            "run_keys": [
                model.decoder.loss],
            "result_keys": ["val_unreg_loss"],
            "feed_keys": [model.encoder.inputs, model.encoder.inputs_length,
                          model.decoder.targets, model.decoder.targets_length],
            "n_batch": 100,
        }
        val_gen = time_major_generator(val_gen)

    history = run_train(
        sess,
        train_gen, train_params,
        val_gen, val_params,
        run_params)

    return history
