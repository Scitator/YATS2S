import tensorflow as tf
from tqdm import tqdm

from rstools.utils.batch_utils import files_data_generator, merge_generators
from rstools.tf.training import run_train
from typical_argparse import parse_args
from seq2seq.rnn_seq2seq import DynamicSeq2Seq
from seq2seq.batch_utils import time_major_batch
from seq2seq.training.utils import get_rnn_cell

def train_seq2seq(
        sess, model, train_gen, val_gen=None, run_params=None,
        n_batch=-1, scheduled_sampling_probability=None):
    train_params = {
        "run_keys": [
            model.decoder.loss,
            model.decoder.train_op, model.encoder.train_op, model.embeddings.train_op],
        "result_keys": ["unreg_loss"],
        "feed_keys": [model.encoder.inputs, model.encoder.inputs_length,
                      model.decoder.targets, model.decoder.targets_length],
        "n_batch": n_batch
    }

    if scheduled_sampling_probability is not None:
        train_params["feed_keys"] += [model.decoder.scheduled_sampling_probability]

    val_params = None
    if val_gen is not None:
        val_params = {
            "run_keys": [model.decoder.loss],
            "result_keys": ["val_unreg_loss"],
            "feed_keys": [model.encoder.inputs, model.encoder.inputs_length,
                          model.decoder.targets, model.decoder.targets_length],
            "n_batch": n_batch
        }
        if scheduled_sampling_probability is not None:
            val_params["feed_keys"] += [model.decoder.scheduled_sampling_probability]

    history = run_train(
        sess,
        train_gen, train_params,
        val_gen, val_params,
        run_params)

    return history


def yield_seq_target(seq, target, double=False, scheduled_sampling_probability=None):
    seq, seq_len = time_major_batch(seq)
    target, target_len = time_major_batch(target)

    if scheduled_sampling_probability is not None:
        yield seq, seq_len, target, target_len, scheduled_sampling_probability
    else:
        yield seq, seq_len, target, target_len

    if double:
        if scheduled_sampling_probability is not None:
            yield target, target_len, seq, seq_len, scheduled_sampling_probability
        else:
            yield target, target_len, seq, seq_len


def seq2seq_iter(data, batch_size, double=False, scheduled_sampling_probability=None):
    indices = np.arange(len(data))
    for batch in iterate_minibatches(indices, batch_size, shuffle=True):
        batch = [data[i] for i in batch]
        seq, target = zip(*batch)
        for yield_data in yield_seq_target(
                seq, target,
                double=double, scheduled_sampling_probability=scheduled_sampling_probability):
            yield yield_data


def seq2seq_generator_wrapper(generator, double=False, scheduled_sampling_probability=None):
    for batch in generator:
        seq, target = batch
        for yield_data in yield_seq_target(
                seq, target,
                double=double, scheduled_sampling_probability=scheduled_sampling_probability):
            yield yield_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--from_corpora_path",
        type=str,
        default="data/quora/pph1_enc.pkl")
    parser.add_argument(
        "--to_corpora_path",
        type=str,
        default="data/quora/pph2_enc.pkl")
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="data/quora/pph_vocab.txt")
    parser.add_argument(
        "--token2id_path",
        type=str,
        default="data/quora/token2id.json")
    parser.add_argument(
        "--id2token_path",
        type=str,
        default="data/quora/id2token.json")

    parser.add_argument(
        "--double_iter",
        action="store_true",
        default=False)

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
        "--bidirectional",
        action="store_true",
        default=False)
    parser.add_argument(
        "--attention",
        action="store_true",
        default=False)
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100)
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
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=10)

    parser.add_argument(
        "--training_mode",
        type=str,
        default="greedy",
        choices=["greedy", "scheduled_sampling_embedding", "scheduled_sampling_output"])
    parser.add_argument(
        "--scheduled_sampling_probability",
        type=float,
        default=None)

    parser.add_argument(
        "--decoding_mode",
        type=str,
        default="greedy",
        choices=["greedy", "beam"])
    parser.add_argument(
        "--beam_width",
        type=int,
        default=5)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    vocab = {}

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

    model = DynamicSeq2Seq(
        vocab_size, emb_size,
        encoder_args, decoder_args,
        optimization_args,
        optimization_args,
        optimization_args)

    run_params = {
        "n_epochs": args.n_epochs,
        "log_dir": args.log_dir,
        "checkpoint_every": args.checkpoint_every,
        "model_global_step": model.decoder.global_step
    }

    gpu_option = args.gpu_option
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_option)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        history = train_seq2seq(
            sess, model,
            train_iter,
            val_iter,
            run_params,
            n_batch,
            args.scheduled_sampling_probability)


if __name__ == "__main__":
    main()
