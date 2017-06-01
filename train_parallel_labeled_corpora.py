import tensorflow as tf
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split

from rstools.utils.batch_utils import iterate_minibatches, files_data_generator, merge_generators
from rstools.utils.os_utils import unpickle_data, masked_files
from rstools.tf.training import run_train
from typical_argparse import parse_args
from seq2seq.rnn_seq2seq import DynamicSeq2Seq
from seq2seq.batch_utils import time_major_batch
from seq2seq.training.utils import get_rnn_cell


def train_seq2seq(sess, model, train_gen, val_gen=None, run_params=None, n_batch=-1):
    train_params = {
        "run_keys": [
            model.decoder.loss,
            model.decoder.train_op, model.encoder.train_op, model.embeddings.train_op],
        "result_keys": ["unreg_loss"],
        "feed_keys": [model.encoder.inputs, model.encoder.inputs_length,
                      model.decoder.targets, model.decoder.targets_length],
        "n_batch": n_batch
    }

    val_params = None
    if val_gen is not None:
        val_params = {
            "run_keys": [model.decoder.loss],
            "result_keys": ["val_unreg_loss"],
            "feed_keys": [model.encoder.inputs, model.encoder.inputs_length,
                          model.decoder.targets, model.decoder.targets_length],
            "n_batch": n_batch
        }

    history = run_train(
        sess,
        train_gen, train_params,
        val_gen, val_params,
        run_params)

    return history


def seq2seq_iter(data, batch_size, double=False):
    indices = np.arange(len(data))
    for batch in iterate_minibatches(indices, batch_size):
        batch = [data[i] for i in batch]
        seq, target = zip(*batch)
        seq, seq_len = time_major_batch(seq)
        target, target_len = time_major_batch(target)
        yield seq, seq_len, target, target_len
        if double:
            yield target, target_len, seq, seq_len


def seq2seq_generator_wrapper(generator, double=False):
    for batch in generator:
        seq, target = batch
        seq, seq_len = time_major_batch(seq)
        target, target_len = time_major_batch(target)
        yield seq, seq_len, target, target_len
        if double:
            yield target, target_len, seq, seq_len


def main():
    args = parse_args()

    with open(args.vocab_path) as fin:
        vocab = fin.readlines()
    with open(args.token2id_path) as fout:
        token2id = json.load(fout)
    with open(args.id2token_path) as fout:
        id2token = json.load(fout)
        id2token = {int(key): value for key, value in id2token.items()}

    unk_id = 2
    unk = " "
    encode = lambda line: list(map(lambda t: token2id.get(t, unk_id), line))
    decode = lambda line: "".join(list(map(lambda i: id2token.get(i, unk), line)))

    vocab_size = len(vocab) + 3
    emb_size = args.embedding_size
    batch_size = args.batch_size
    n_batch = args.n_batch

    encoder_cell_params = {"num_units": args.num_units}
    decoder_cell_params = {"num_units": args.num_units + args.num_units * int(args.bidirectional)}

    encoder_args = {
        "cell": get_rnn_cell(
            args.cell, encoder_cell_params,
            num_layers=args.num_layers,
            residual_connections=args.residual_connections,
            residual_dense=args.residual_dense),
        "bidirectional": args.bidirectional,
    }

    decoder_args = {
        "cell": get_rnn_cell(
            args.cell, decoder_cell_params,
            num_layers=args.num_layers,
            residual_connections=args.residual_connections,
            residual_dense=args.residual_dense),
        "attention": args.attention,
    }

    optimization_args = {
        "decay_steps": args.lr_decay_steps,
        "lr_decay": args.lr_decay_koef
    }

    if "*" in args.from_corpora_path and "*" in args.to_corpora_path:
        corpora_from_files = np.array(masked_files(args.from_corpora_path))
        corpora_to_files = np.array(masked_files(args.to_corpora_path))
        assert len(corpora_from_files) == len(corpora_to_files)

        indices = np.arange(len(corpora_from_files))
        train_ids, val_ids = train_test_split(indices, test_size=0.2, random_state=42)

        train_from_files = np.array([corpora_from_files[i] for i in train_ids])
        train_to_files = np.array([corpora_to_files[i] for i in train_ids])

        val_from_files = np.array([corpora_from_files[i] for i in val_ids])
        val_to_files = np.array([corpora_to_files[i] for i in val_ids])

        train_corpora_from_it = files_data_generator(
            mask=train_from_files,
            open_fn=unpickle_data,
            batch_size=batch_size,
            files_shuffle=True,
            data_shuffle=True)
        train_corpora_to_it = files_data_generator(
            mask=train_to_files,
            open_fn=unpickle_data,
            batch_size=batch_size,
            files_shuffle=True,
            data_shuffle=True)

        val_corpora_from_it = files_data_generator(
            mask=val_from_files,
            open_fn=unpickle_data,
            batch_size=batch_size,
            files_shuffle=True,
            data_shuffle=True)
        val_corpora_to_it = files_data_generator(
            mask=val_to_files,
            open_fn=unpickle_data,
            batch_size=batch_size,
            files_shuffle=True,
            data_shuffle=True)

        train_generator = merge_generators([train_corpora_from_it, train_corpora_to_it])
        val_generator = merge_generators([val_corpora_from_it, val_corpora_to_it])

        train_iter = seq2seq_generator_wrapper(train_generator, double=args.double_iter)
        val_iter = seq2seq_generator_wrapper(val_generator, double=args.double_iter)

        if args.lr_decay_on == "epoch":
            optimization_args["decay_steps"] *= n_batch
    else:
        with open(args.from_corpora_path, "rb") as fout:
            pph1_enc = pickle.load(fout)
        with open(args.to_corpora_path, "rb") as fout:
            pph2_enc = pickle.load(fout)

        indices = np.arange(len(pph1_enc))
        train_ids, val_ids = train_test_split(indices, test_size=0.2, random_state=42)

        train_input = [pph1_enc[i] for i in train_ids]
        train_target = [pph2_enc[i] for i in train_ids]
        train_data = list(zip(train_input, train_target))

        val_input = [pph1_enc[i] for i in val_ids]
        val_target = [pph2_enc[i] for i in val_ids]
        val_data = list(zip(val_input, val_target))

        train_iter = seq2seq_iter(train_data, batch_size, double=args.double_iter)
        val_iter = seq2seq_iter(val_data, batch_size, double=args.double_iter)

        if args.lr_decay_on == "epoch":
            optimization_args["decay_steps"] *= len(train_data) / batch_size

    model = DynamicSeq2Seq(
        vocab_size, emb_size,
        encoder_args, decoder_args,
        optimization_args,
        optimization_args,
        optimization_args)

    gpu_option = args.gpu_option
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)

    run_params = {
        "n_epochs": args.n_epochs,
        "log_dir": args.log_dir
    }

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        history = train_seq2seq(
            sess, model,
            train_iter,
            val_iter,
            run_params,
            n_batch)


if __name__ == "__main__":
    main()
