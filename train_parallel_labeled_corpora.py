import tensorflow as tf
from tqdm import tqdm

from rstools.utils.batch_utils import files_data_generator, merge_generators
from rstools.tf.training import run_train
from typical_argparse import parse_args
from seq2seq.rnn_seq2seq import DynamicSeq2Symbol
from seq2seq.batch_utils import time_major_batch
from seq2seq.training.utils import get_rnn_cell


def train_seq2seq(
        sess, model,
        train_gen, val_gen=None,
        run_params=None,
        n_batch=-1, n_batch_val=-1):
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
            "n_batch": n_batch_val
        }

    history = run_train(
        sess,
        train_gen, train_params,
        val_gen, val_params,
        run_params)

    return history


def vocab_encoder_wrapper(vocab, unk_id=2):
    def line_ecoder_fn(line):
        line = line.split(" ")
        return list(map(lambda t: vocab.get(t, unk_id), line))

    return line_ecoder_fn


def open_file_wrapper(proc_fn):
    def open_file_fn(filepath):
        with open(filepath) as fin:
            for line in fin:
                line = line.replace("\n", "")
                yield proc_fn(line)

    return open_file_fn


def labeled_data_generator(
        data_dir, text_vocab, label_vocab, join_id=3,
        prefix="train", batch_size=32):
    source_file = "{}/{}_sources.txt".format(data_dir, prefix)
    target_file = "{}/{}_targets.txt".format(data_dir, prefix)
    label_file = "{}/{}_labels.txt".format(data_dir, prefix)

    files_its = [
        files_data_generator(
            [source_file],
            open_file_wrapper(vocab_encoder_wrapper(text_vocab)),
            batch_size),
        files_data_generator(
            [target_file],
            open_file_wrapper(vocab_encoder_wrapper(text_vocab)),
            batch_size),
        files_data_generator(
            [label_file],
            open_file_wrapper(vocab_encoder_wrapper(label_vocab)),
            batch_size)
    ]

    text_batch = []
    label_batch = []

    for batch_row in tqdm(merge_generators(files_its)):
        text = batch_row[0] + [join_id] + batch_row[1]
        label = batch_row[2]
        text_batch.append(text)
        label_batch.append(label)
        if len(text_batch) >= batch_size:
            text, text_len = time_major_batch(text_batch)
            label, label_len = time_major_batch(label_batch)

            yield text, text_len, label, label_len
            text_batch = []
            label_batch = []


def load_vocab(filepath, ids_bias=0):
    tokens = []
    with open(filepath) as fin:
        for line in fin:
            line = line.replace("\n", "")
            token, freq = line.split()
            tokens.append(token)

    token2id = {t: i + ids_bias for i, t in enumerate(tokens)}
    id2token = {i + ids_bias: t for i, t in enumerate(tokens)}
    return token2id, id2token


def main():
    args = parse_args()
    text_ids_bias = 4
    text_vocab, _ = load_vocab("{}/vocab.txt".format(args.data_dir), ids_bias=text_ids_bias)
    labels_ids_bias = 2
    label_vocab = {"0": 2, "1": 3}

    train_data_gen = labeled_data_generator(
        args.data_dir, text_vocab, label_vocab,
        batch_size=args.batch_size)

    val_data_gen = labeled_data_generator(
        args.data_dir, text_vocab, label_vocab,
        batch_size=args.batch_size, prefix="test")

    vocab_size = len(text_vocab) + text_ids_bias
    emb_size = args.embedding_size
    n_batch = args.n_batch
    n_batch_val = args.n_batch_test

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
        "lr_decay": args.lr_decay_factor
    }

    model = DynamicSeq2Symbol(
        vocab_size, emb_size, len(label_vocab) + labels_ids_bias,
        encoder_args, decoder_args,
        optimization_args,
        optimization_args,
        optimization_args)

    gpu_option = args.gpu_option
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)

    run_params = {
        "n_epochs": args.n_epochs,
        "log_dir": args.log_dir,
        "checkpoint_every": args.checkpoint_every
    }

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        history = train_seq2seq(
            sess, model,
            train_data_gen,
            val_data_gen,
            run_params,
            n_batch=n_batch,
            n_batch_val=n_batch_val)


if __name__ == "__main__":
    main()
