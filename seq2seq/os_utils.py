import os
import pickle


def create_if_need(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_history(history, out_dir):
    with open(os.path.join(out_dir, "history.pkl"), "wb") as fout:
        pickle.dump(history, fout)


def save_model(sess, saver, save_dir, model_global_step=None):
    create_if_need(save_dir)
    save_path = os.path.join(save_dir, "model.cpkl")
    saver.save(sess, save_path, global_step=model_global_step)
