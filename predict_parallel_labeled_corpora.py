import tensorflow as tf
import numpy as np
from tqdm import tqdm
import sys
from rstools.utils.batch_utils import files_data_iterator, merge_generators
from rstools.utils.os_utils import pickle_data
from typical_argparse import parse_args
from seq2seq.rnn_seq2seq import DynamicSeq2Symbol
from seq2seq.batch_utils import time_major_batch
from seq2seq.training.utils import get_rnn_cell


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
        data_dir, text_vocab, join_id=3,
        prefix="train", batch_size=32):
    source_file = "{}/{}_sources.txt".format(data_dir, prefix)
    target_file = "{}/{}_targets.txt".format(data_dir, prefix)

    files_its = [
        files_data_iterator(
            [source_file],
            open_file_wrapper(vocab_encoder_wrapper(text_vocab)),
            batch_size),
        files_data_iterator(
            [target_file],
            open_file_wrapper(vocab_encoder_wrapper(text_vocab)),
            batch_size),
    ]

    text_batch = []

    for batch_row in tqdm(merge_generators(files_its)):
        text = batch_row[0] + [join_id] + batch_row[1]
        text_batch.append(text)
        if len(text_batch) >= batch_size:
            text, text_len = time_major_batch(text_batch)

            yield text, text_len
            text_batch = []
    if len(text_batch) >= 0:
        text, text_len = time_major_batch(text_batch)
        yield text, text_len


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
    ids_bias = 4
    text_vocab, _ = load_vocab("{}/vocab.txt".format(args.data_dir), ids_bias=ids_bias)
    labels_ids_bias = 2
    label_vocab = {"0": 2, "1": 3}

    pred_data_gen = labeled_data_generator(
        args.data_dir, text_vocab,
        batch_size=args.batch_size, prefix="pred")

    vocab_size = len(text_vocab) + ids_bias
    emb_size = args.embedding_size

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

    model = DynamicSeq2Symbol(
        vocab_size, emb_size, len(label_vocab) + labels_ids_bias,
        encoder_args, decoder_args)

    gpu_option = args.gpu_option
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)
    predictions = []
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.Saver(
            var_list=tf.trainable_variables(),
            keep_checkpoint_every_n_hours=1)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.checkpoint)

        for text, text_len in pred_data_gen:
            batch_prediction = sess.run(
                model.decoder.inference_prediction_probabilities,
                feed_dict={
                    model.encoder.inputs: text,
                    model.encoder.inputs_length: text_len
                })
            predictions.append(batch_prediction)

    pickle_data(predictions, "predictions.pkl")

    predictions = list(map(lambda x: x[0, :, 2:], predictions))
    predictions = np.vstack(predictions)
    np.save("predictions.npy", predictions)

if __name__ == "__main__":
    main()
