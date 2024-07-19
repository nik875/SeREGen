import os
import shutil
import pickle
import torch


def _create_save_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def _save_object(obj, base_path, filename):
    with open(os.path.join(base_path, filename), "wb") as f:
        pickle.dump(obj, f)


def _load_object(base_path, filename):
    with open(os.path.join(base_path, filename), "rb") as f:
        return pickle.load(f)


def _save_torch(model, base_path, filename):
    torch.save(model, os.path.join(base_path, filename))


def _load_torch(base_path, filename):
    return torch.load(os.path.join(base_path, filename))
