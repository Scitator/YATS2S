import os


def create_if_need(path):
    if not os.path.exists(path):
        os.makedirs(path)
