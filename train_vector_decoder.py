import tensorflow as tf
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
import argparse
from tensorflow.contrib import rnn
from rstools.utils.batch_utils import iterate_minibatches
from rstools.tf.training import run_train
from seq2seq.rnn_vec2seq import DynamicVec2Seq
from seq2seq.batch_utils import time_major_batch


def train_vec2seq(sess, model, train_gen, val_gen=None, run_params=None, n_batch=-1):
    train_params = {
        "run_keys": [
            model.decoder.loss,
            model.decoder.train_op, model.encoder_train_op, model.embeddings.train_op],
        "result_keys": ["unreg_loss"],
        "feed_keys": [model.inputs_vec,
                      model.decoder.targets, model.decoder.targets_length],
        "n_batch": n_batch
    }

    val_params = None
    if val_gen is not None:
        val_params = {
            "run_keys": [model.decoder.loss],
            "result_keys": ["val_unreg_loss"],
            "feed_keys": [model.inputs_vec,
                          model.decoder.targets, model.decoder.targets_length],
            "n_batch": n_batch
        }

    history = run_train(
        sess,
        train_gen, train_params,
        val_gen, val_params,
        run_params)

    return history


def vec2seq_iter(data, batch_size):
    indices = np.arange(len(data))
    for batch in iterate_minibatches(indices, batch_size):
        batch = [data[i] for i in batch]
        vec, targets = zip(*batch)
        targets = [(i, x) for i, y in enumerate(targets) for x in y]
        target_indices = np.arange(len(data))
        for target_indices_batch in iterate_minibatches(target_indices, batch_size):
            target = [targets[i] for i in target_indices_batch]
            vec_batch = [vec[i] for i, _ in target]
            target_batch = [x for _, x in target]
            target_batch, target_batch_len = time_major_batch(target_batch)
            yield vec_batch, target_batch, target_batch_len


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from_vec_path",
        type=str,
        default="data/yds/images_enc.npy")
    parser.add_argument(
        "--to_corpora_path",
        type=str,
        default="data/yds/labels_enc.pkl")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="data/yds/vocab.txt")
    parser.add_argument(
        "--token2id_path",
        type=str,
        default="data/yds/token2id.json")
    parser.add_argument(
        "--id2token_path",
        type=str,
        default="data/yds/id2token.json")

    parser.add_argument(
        "--gpu_option",
        type=float,
        default=0.45)
    parser.add_argument(
        "--encoder_size",
        type=int,
        default=512)
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=64)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64)
    parser.add_argument(
        "--decoder_size",
        type=int,
        default=512)
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1000)
    parser.add_argument(
        "--n_batch",
        type=int,
        default=-1)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs")

    parser.add_argument(
        "--lr_decay_on",
        type=str,
        default="epoch",
        choices=["epoch", "batch"])
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=1)
    parser.add_argument(
        "--lr_decay_koef",
        type=float,
        default=0.99)

    args, _ = parser.parse_known_args()
    return args


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

    images_enc = np.load(args.from_vec_path, mmap_mode="r")
    with open(args.to_corpora_path, "rb") as fout:
        labels_enc = pickle.load(fout)

    encoder_args = {
        "input_shape": (images_enc.shape[1], ),
        "state_shape": args.encoder_size
    }

    decoder_args = {
        "cell": rnn.LSTMCell(args.decoder_size),
        "attention": False,
    }

    optimization_args = {
        "decay_steps": args.lr_decay_steps,
        "lr_decay": args.lr_decay_koef
    }

    indices = np.arange(len(images_enc))
    train_ids, val_ids = train_test_split(indices, test_size=0.2, random_state=42)

    train_input = [images_enc[i] for i in train_ids]
    train_target = [labels_enc[i] for i in train_ids]
    train_data = list(zip(train_input, train_target))

    val_input = [images_enc[i] for i in val_ids]
    val_target = [labels_enc[i] for i in val_ids]
    val_data = list(zip(val_input, val_target))

    train_iter = vec2seq_iter(train_data, batch_size)
    val_iter = vec2seq_iter(val_data, batch_size)

    if args.lr_decay_on == "epoch":
        optimization_args["decay_steps"] *= len(train_data) / batch_size

    model = DynamicVec2Seq(
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
        history = train_vec2seq(
            sess, model,
            train_iter,
            val_iter,
            run_params,
            n_batch)


if __name__ == "__main__":
    main()
