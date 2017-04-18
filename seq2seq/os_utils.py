import os
import pickle


def create_if_need(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_history(history, out_dir):
    with open(os.path.join(out_dir, "history.pkl"), "wb") as fout:
        pickle.dump(history, fout)
