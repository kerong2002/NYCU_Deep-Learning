import numpy as np
import pickle
"""
    utils module
    supported functions:
        - one_hot_encoding
        - one_hot_to_label
        - save_model
        - load_model
"""

def one_hot_encoding(y, n_classes):
    return np.eye(n_classes)[y].reshape(-1, n_classes)

def one_hot_to_label(y):
    return np.argmax(y, axis = 1)

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
        
def load_model(model, path):
    with open(path, 'rb') as f:
        model.net = pickle.load(f)
        
def use_tensorboard(logdir):
    import glob
    import tensorboardX
    if not glob.glob(f"logs/{logdir}") or not glob.glob(f"logs/{logdir}/*"):
        writer = tensorboardX.SummaryWriter(log_dir=f"logs/{logdir}/1")
    else:
        maximum = max([int(x.split("/")[-1]) for x in glob.glob(f"logs/{logdir}/*")])
        writer = tensorboardX.SummaryWriter(log_dir=f"logs/{logdir}/{maximum + 1}")
    return writer