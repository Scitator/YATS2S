import numpy as np
import tensorflow as tf
import argparse
import json

from seq2seq.rnn_seq2seq import create_seq2seq_experiment_fn
from seq2seq.input.generator_io import generator_input_fn


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


def file_data_generator_py(filepath, line_encode_fn):
    def generator():
        with open(filepath) as fin:
            for line in fin:
                inputs, targets = line.replace("\n", "").split("\t")
                inputs = np.array(line_encode_fn(inputs), dtype=np.int32)
                inputs_length = np.array(len(inputs), dtype=np.int32)
                targets = np.array(line_encode_fn(targets), dtype=np.int32)
                targets_length = np.array(len(targets), dtype=np.int32)

                yield {
                    "inputs": inputs,
                    "inputs_length": inputs_length,
                    "targets": targets,
                    "targets_length": targets_length
                }

    return generator


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_corpora_path",
        type=str)
    parser.add_argument(
        "--test_corpora_path",
        type=str)

    # Vocab & embeddings params
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="data/quora/pph_vocab.txt")
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=64)

    # rnn cell params
    parser.add_argument(
        "--cell_num",
        type=int,
        default=2)
    parser.add_argument(
        "--cell",
        type=str,
        default="LSTMCell")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2)
    parser.add_argument(
        "--num_units",
        type=int,
        default=16)

    # special cell params
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=False)
    parser.add_argument(
        "--attention",
        type=str,
        default=None,
        choices=["bahdanau", "luong"])
    parser.add_argument(
        "--residual_connections",
        action="store_true",
        default=False)
    parser.add_argument(
        "--residual_dense",
        action="store_true",
        default=False)

    # training params
    parser.add_argument(
        "--training_mode",
        type=str,
        default="greedy",
        choices=["greedy", "scheduled_sampling_embedding", "scheduled_sampling_output"])
    parser.add_argument(
        "--scheduled_sampling_probability",
        type=float,
        default=0.0)

    # inference params [WIP]
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="greedy",
        choices=["greedy", "beam"])
    parser.add_argument(
        "--beam_width",
        type=int,
        default=3)

    # tf data params
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None)
    parser.add_argument(
        "--queue_capacity",
        type=int,
        default=128)
    parser.add_argument(
        "--num_threads",
        type=int,
        default=2)

    # tf training params
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs")
    parser.add_argument(
        "--train_steps",
        type=int,
        default=int(1e4))
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=int(1e3))
    parser.add_argument(
        "--min_eval_frequency",
        type=int,
        default=int(1e3))

    # optimization
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4)
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=100000)
    parser.add_argument(
        "--lr_decay_koef",
        type=float,
        default=0.99)
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=10.0)

    # other run params
    parser.add_argument(
        "--gpu_option",
        type=float,
        default=0.5)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # special symbols: PAD, EOS, UNK
    unk_id = 2
    vocab_ids_bias = unk_id + 1
    vocab, _ = load_vocab(args.vocab_path, ids_bias=vocab_ids_bias)
    encode = lambda line: list(map(lambda x: vocab.get(x, unk_id), line.split(" ")))
    vocab_size = len(vocab) + vocab_ids_bias

    train_input_fn = generator_input_fn(
        x=file_data_generator_py(args.train_corpora_path, line_encode_fn=encode),
        target_key=["targets", "targets_length"],
        batch_size=args.batch_size, shuffle=False, num_epochs=args.num_epochs,
        queue_capacity=args.queue_capacity, num_threads=args.num_threads,
        pad_data=True)

    val_input_fn = generator_input_fn(
        x=file_data_generator_py(args.test_corpora_path, line_encode_fn=encode),
        target_key=["targets", "targets_length"],
        batch_size=args.batch_size, shuffle=False, num_epochs=args.num_epochs,
        queue_capacity=args.queue_capacity, num_threads=1,
        pad_data=True)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_option)
    run_config = tf.contrib.learn.RunConfig(
        session_config=tf.ConfigProto(gpu_options=gpu_options),
        model_dir=args.log_dir)

    hparams = tf.contrib.training.HParams(
        cell_num=args.cell_num,
        vocab_size=vocab_size, embedding_size=args.embedding_size,
        cell=args.cell, num_layers=args.num_layers, num_units=args.num_units,
        bidirectional=args.bidirectional, attention=args.attention,
        residual_connections=args.residual_connections,
        residual_dense=args.residual_dense,
        training_mode=args.training_mode,
        learning_rate=args.learning_rate,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_koef=args.lr_decay_koef,
        gradient_clip=args.gradient_clip,
        scheduled_sampling_probability=args.scheduled_sampling_probability,
        inference_mode=args.inference_mode,
        beam_width=args.beam_width)

    experiment_fn = create_seq2seq_experiment_fn(
        train_input_fn, val_input_fn,
        args.train_steps, args.eval_steps, args.min_eval_frequency)

    with open("{}/hparams.json".format(args.log_dir), "w") as fout:
        json.dump(hparams.values(), fout)

    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=hparams)


if __name__ == "__main__":
    main()
