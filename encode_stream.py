import numpy as np
import sys
import argparse
import json
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.contrib import rnn
from seq2seq.rnn_seq2seq import DynamicSeq2Seq
from seq2seq.batch_utils import time_major_batch


def parse_args():
    desc = "DynamicRNN stream encoder"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for stream encoding. (default: %(default)s)")

    parser.add_argument(
        "--use_norm", action="store_true",
        help="Boolean flag indicating if embeddings should be normalized")

    parser.add_argument(
        "--token2id_path",
        type=str)

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to Skip-Thoughts model tensorflow checkpoint")

    # RNN Encoder
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
        "--decoder_size",
        type=int,
        default=512)
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        default=False)
    parser.add_argument(
        "--attention",
        action="store_true",
        default=False)
    parser.add_argument(
        "--lstm_connection_type",
        type=int,
        choices=[0, 1, 2],
        default=1)

    args = parser.parse_args()

    return args


def create_model(vocab_size, embedding_size, encoder_args, decoder_args):
    model = DynamicSeq2Seq(
        vocab_size, embedding_size, encoder_args, decoder_args)
    return model


def encode_chunk(sess, model, chunk, use_norm=False, lstm_connection=1):
    query_batch, query_batch_len = time_major_batch(chunk)
    predicted = sess.run(
        model.encoder.state,
        feed_dict={
            model.encoder.inputs: query_batch,
            model.encoder.inputs_length: query_batch_len})
    
    if isinstance(predicted, tuple):
        if lstm_connection == 2:
            predicted = np.concatenate((predicted[0], predicted[1]), axis=1)
        else:
            predicted = predicted[lstm_connection]

    if use_norm:
        normalize(predicted, axis=1, copy=False)

    return predicted


def rnn_encoder_encode_stream(sess, stream, model, batch_size, use_norm=False, lstm_connection=1):
    chunk = []
    for line in stream:
        chunk.append(line)
        if len(chunk) >= batch_size:
            yield encode_chunk(sess, model, chunk, use_norm)
            chunk = []
    if len(chunk) > 0:
        yield encode_chunk(sess, model, chunk, use_norm)


def encoder_pipeline(
        sess, data_stream, token2id, embedding_size,
        encoder_size, bidirectional, decoder_size, attention,
        checkpoint_path,
        batch_size=32, use_norm=False, lstm_connection=1):

    encoder_args = {
        "cell": rnn.LSTMCell(encoder_size),
        "bidirectional": bidirectional,
    }

    # @TODO: rewrite save-load for no-decoder usage
    decoder_args = {
        "cell": rnn.LSTMCell(decoder_size),
        "attention": attention,
    }
    spec_symbols_bias = 3
    model = create_model(len(token2id) + spec_symbols_bias, embedding_size, encoder_args, decoder_args)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)

    for embedding_matr in rnn_encoder_encode_stream(sess, data_stream, model, batch_size, use_norm):
        yield embedding_matr


def main_encoder_pipeline(args):
    with open(args.token2id_path) as fout:
        token2id = json.load(fout)

    gpu_option = args.gpu_option
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)
    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    for embedding_matr in encoder_pipeline(
            sess, sys.stdin, token2id, args.embedding_size,
            args.encoder_size, args.bidirectional, args.decoder_size, args.attention,
            args.checkpoint_path, args.batch_size, args.use_norm, args.lstm_connection):
        for embedding_vec in embedding_matr:
            embedding_str = " ".join(list(map(str, embedding_vec)))
            print(embedding_str)


def main():
    args = parse_args()

    main_encoder_pipeline(args)


if __name__ == "__main__":
    main()
