import numpy as np

import Hank
import Hank.nn as nn

from Hank.optim import SGD, SGDM, Adam, RMSprop, StepRL
from Hank.loss import CrossEntropyLoss
from Hank.utils import save_model, use_tensorboard

from plot_result import show_loss, show_result
from data_generator import generate_linear, generate_XOR_easy

from argparse import ArgumentParser

class Model(nn.module):
    def __init__(self, input_size, output_size, activation, hidden_size):
        super().__init__()
        if activation in nn.activation_functions_list:
            self.act = eval(f"nn.{activation}")
        else:
            raise ValueError(f"Activation function {activation} is not supported")
        self.net = [
            nn.Linear(input_size, hidden_size),
            self.act(),
            nn.Linear(hidden_size, hidden_size),
            self.act(),
            nn.Linear(hidden_size, output_size),
        ]
        self.delta = np.empty(len(self.net) + 1, dtype = object)
        
    def forward(self, x):
        for layer in self.net:
            x = layer.forward(x)
        return x

    def backward(self, delta, y_pred):
        self.delta[-1] = delta
        for i in range(len(self.net) - 1, -1, -1):
            if isinstance(self.net[i], nn.layer):
                self.delta[i] = self.net[i].backward(self.delta[i + 1])
            elif isinstance(self.net[i], nn.activation):
                if i + 1 >= len(self.net):
                    self.delta[i] = self.delta[i + 1] * self.net[i].backward(y_pred)
                else:
                    self.delta[i] = self.delta[i + 1] * self.net[i].backward(self.net[i + 1].a)
            else:
                assert False, "Unknown layer type"
            
def main(args):
    if args.problem == "linear":
        X, Y = generate_linear(n=100)
        in_dim, out_dim = 2, 2
    elif args.problem == "XOR":
        X, Y = generate_XOR_easy()
        in_dim, out_dim = 2, 2
    else:
        raise ValueError(f"Problem {args.problem} is not supported")

    if args.tensorboard:
        writer = use_tensorboard(args.log_dir)
        
    model = Model(in_dim, out_dim, args.activation, args.hidden)
    
    if args.optimizer == "SGD":
        optimizer = SGD(model, lr = args.lr)
    elif args.optimizer == "SGDM":
        optimizer = SGDM(model, lr = args.lr)
    elif args.optimizer == "RMSProp":
        optimizer = RMSprop(model, lr = args.lr)
    elif args.optimizer == "Adam":
        optimizer = Adam(model, lr = args.lr)
    
    if args.scheduler:
        scheduler = StepRL(optimizer, step_size = args.epoch // 10, gamma = 0.9)
    
    losses_per_epoch = []
    criterion = CrossEntropyLoss()
    for e in range(args.epoch):
        losses = []
        # training
        for (x, y) in zip(X, Y):
            x, y = x.reshape(in_dim, -1), y.reshape(1, 1)
            y_pred = model(x)
            loss, delta = criterion(y_pred, y)
            model.backward(delta, y_pred)
            optimizer.step()
            losses.append(loss)
        losses_per_epoch.append(np.mean(losses))
        
        # validation
        x_all, y_all = X.reshape(-1, in_dim, 1), Y.reshape(-1, 1)
        y_pred = model(x_all)
        y_pred = np.argmax(y_pred, axis = 1)
        acc = np.mean(y_pred == y_all)
        
        if (e + 1) % 500 == 0:
            print(f"Epoch {e + 1:>4}/{args.epoch} Loss: {np.mean(losses):.8f}, Acc: {acc:.2f}, LR: {optimizer.lr}")
        
        if args.tensorboard:
            writer.add_scalar("Loss", np.mean(losses), e)
            writer.add_scalar("Accuracy", acc, e)
            
        if args.scheduler:
            scheduler.step()
    
        
    if args.save_model:
        save_model(model.net, "model.pkl")
        
    show_loss(losses_per_epoch)
    y_pred = model.forward(X.reshape(-1, in_dim, 1))
    y_pred = np.argmax(y_pred, axis = 1)
    show_result(X, Y, y_pred)
    
    if args.tensorboard:
        writer.close()
        
if __name__ == "__main__":
    Parser = ArgumentParser()
    Parser.add_argument("--problem", type = str, default = "linear", choices = ["linear", "XOR"])
    Parser.add_argument("--activation", type = str, default = "sigmoid", choices = nn.activation_functions_list)
    Parser.add_argument("--optimizer", type = str, default = "SGD", choices = ["SGD", "SGDM", "RMSProp", "Adam"])
    Parser.add_argument("--epoch", type = int, default = 5000)
    Parser.add_argument("--lr", type = float, default = 5e-3)
    Parser.add_argument("--hidden", type = int, default = 32)
    Parser.add_argument("--scheduler", action="store_true")
    Parser.add_argument("--save_model", action = "store_true")
    Parser.add_argument("--tensorboard", action = "store_true")
    Parser.add_argument("--log_dir", type = str, default = "tmp")
    args = Parser.parse_args()
    main(args)