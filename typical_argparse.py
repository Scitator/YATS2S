import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64)
    parser.add_argument(
        "--gpu_option",
        type=float,
        default=0.45)

    parser.add_argument(
        "--embedding_size",
        type=int,
        default=64)

    parser.add_argument(
        "--cell",
        type=str)

    parser.add_argument(
        "--num_units",
        type=int,
        default=512)
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2)

    parser.add_argument(
        "--residual_connections",
        action="store_true",
        default=False)
    parser.add_argument(
        "--residual_dense",
        action="store_true",
        default=False)

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
        default=1000)
    parser.add_argument(
        "--n_batch",
        type=int,
        default=-1)
    parser.add_argument(
        "--n_batch_test",
        type=int,
        default=-1)
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs")
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=-1)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="cross_entropy")

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
        "--lr_decay_factor",
        type=float,
        default=0.99)

    args = parser.parse_args()
    return args
