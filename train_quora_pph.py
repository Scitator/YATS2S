import tensorflow as tf
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
import argparse
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from rstools.utils.batch_utils import iterate_minibatches
from rstools.tf.training import run_train
from seq2seq.rnn_seq2seq import DynamicSeq2Seq
from seq2seq.batch_utils import time_major_batch


def train_seq2seq(sess, model, train_gen, val_gen=None, run_params=None):
    train_params = {
        "run_keys": [
            model.decoder.loss,
            model.decoder.train_op, model.encoder.train_op, model.embeddings.train_op],
        "result_keys": ["unreg_loss"],
        "feed_keys": [model.encoder.inputs, model.encoder.inputs_length,
                      model.decoder.targets, model.decoder.targets_length],
    }

    val_params = None
    if val_gen is not None:
        val_params = {
            "run_keys": [model.decoder.loss],
            "result_keys": ["val_unreg_loss"],
            "feed_keys": [model.encoder.inputs, model.encoder.inputs_length,
                          model.decoder.targets, model.decoder.targets_length],
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--gpu_option',
        type=float,
        default=0.45)
    parser.add_argument(
        '--encoder_size',
        type=int,
        default=512)
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=64)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64)
    parser.add_argument(
        '--decoder_size',
        type=int,
        default=512)
    parser.add_argument(
        '--bidirectional',
        action='store_true',
        default=False)
    parser.add_argument(
        '--attention',
        action='store_true',
        default=False)
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=1000)
    parser.add_argument(
        '--log_dir',
        type=str,
        default="./logs")

    parser.add_argument(
        '--lr_decay_on',
        type=str,
        default='epoch',
        choices=['epoch', 'batch'])
    parser.add_argument(
        '--lr_decay_steps',
        type=int,
        default=1)
    parser.add_argument(
        '--lr_decay_koef',
        type=float,
        default=0.99)

    parser.add_argument(
        '--double_iter',
        action='store_true',
        default=False)

    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    with open("data/pph1_enc.pkl", "rb") as fout:
        pph1_enc = pickle.load(fout)
    with open("data/pph2_enc.pkl", "rb") as fout:
        pph2_enc = pickle.load(fout)
    with open("data/pph_vocab.txt") as fin:
        vocab = fin.readlines()
    with open("data/token2id.json") as fout:
        token2id = json.load(fout)
    with open("data/id2token.json") as fout:
        id2token = json.load(fout)
        id2token = {int(key): value for key, value in id2token.items()}

    unk_id = 2
    unk = " "
    encode = lambda line: list(map(lambda t: token2id.get(t, unk_id), line))
    decode = lambda line: "".join(list(map(lambda i: id2token.get(i, unk), line)))

    indices = np.arange(len(pph1_enc))
    train_ids, val_ids = train_test_split(indices, test_size=0.2, random_state=42)

    train_input = [pph1_enc[i] for i in train_ids]
    train_target = [pph2_enc[i] for i in train_ids]
    train_data = list(zip(train_input, train_target))

    val_input = [pph1_enc[i] for i in val_ids]
    val_target = [pph2_enc[i] for i in val_ids]
    val_data = list(zip(val_input, val_target))

    vocab_size = len(vocab) + 3
    emb_size = args.embedding_size
    batch_size = args.batch_size

    encoder_args = {
        "cell": rnn.LSTMCell(args.encoder_size),
        "bidirectional": args.bidirectional,
    }

    decoder_args = {
        "cell": rnn.LSTMCell(args.decoder_size),
        "attention": args.attention,
    }

    optimization_args = {
        "decay_steps": args.lr_decay_steps,
        "lr_decay": args.lr_decay_koef
    }

    if args.lr_decay_on == "epoch":
        optimization_args["decay_steps"] *= len(train_data) / batch_size

    model = DynamicSeq2Seq(
        vocab_size, emb_size,
        encoder_args, decoder_args,
        optimization_args,
        optimization_args,
        optimization_args)

    train_iter = seq2seq_iter(train_data, batch_size, double=args.double_iter)
    val_iter = seq2seq_iter(val_data, batch_size, double=args.double_iter)

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
            run_params)


if __name__ == '__main__':
    main()
