import numpy as np

import Hank
import Hank.nn as nn
from Hank.loss import BCELoss
from Hank.utils import load_model
from data_generator import generate_linear, generate_XOR_easy
from argparse import ArgumentParser


class Model(nn.module):
    def __init__(self):
        super().__init__()
        self.net = []
        
    def forward(self, x):
        for layer in self.net:
            x = layer.forward(x)
        return x

    def backward(self):
        pass
            
def main(args):
    if args.problem == "linear":
        X, Y = generate_linear(n=100)
        in_dim, out_dim = 2, 1
    elif args.problem == "XOR":
        X, Y = generate_XOR_easy()
        in_dim, out_dim = 2, 1
    else:
        raise ValueError(f"Problem {args.problem} is not supported")
    
    model = Model()
    criterion = BCELoss()
    load_model(model, "model.pkl")
    losses = []
    accuracy = 0
    for i, (x, y) in enumerate(zip(X, Y)):
        x, y = x.reshape(in_dim, -1), y.reshape(out_dim, -1)
        y_pred = model(x)
        losses.append(criterion(y, y_pred)[0])
        accuracy += (np.round(y_pred) == y)
        print(f"Iteration{i:>3}|    Ground truth: {y[0][0]:.2f}    Predicted: {y_pred[0][0]}")
    print(f"loss={np.mean(losses):.5f}, accuracy={accuracy[0][0] / len(X) * 100:.2f}%")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--problem", type=str, default="linear", choices=["linear", "XOR"])
    args = parser.parse_args()
    main(args)