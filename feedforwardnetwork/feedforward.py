import numpy as np
from feedforwardnetwork.layer import FullyConnectedLayer
from loss.squared_loss import SquaredLoss
from tqdm import tqdm
import random

class FeedForward:
    def __init__(self, layer_len, eps=.0001, seed=None, loss=SquaredLoss, **kwargs):
        self.seed = np.random.randint(int(1e4)) if seed is None else int(seed)
        np.random.seed(self.seed)
        self.layer_len = layer_len
        self.eps = eps
        self.n_layers = len(self.layer_len) - 1
        self.layers = [FullyConnectedLayer(layer_len[ii], layer_len[ii + 1], **kwargs) for ii in range(self.n_layers)]
        self.loss = loss(self.layer_len[-1])
        self.loss_hist = None

    def __call__(self, x):
        # forward passing here
        x_in = x
        for ll in range(self.n_layers):
            x_in = self.layers[ll](x_in)
        return x_in

    def backprop(self):
        # using gradient backpropagation
        delta = self.layers[-1].update_grad(self.loss.dloss_dy)
        for ll in range(self.n_layers - 1)[::-1]:
            delta = self.layers[ll].update_grad(delta)

    def gradient_descent_par(self):
        # update the params using grad descent
        for ll in self.layers:
            for nn in ll.neurons:
                nn.gradient_descent(self.eps)

    def train(self, x_train, y_train, n_iter_max=10000, loss_tol=.1):
        self.loss_hist = np.zeros(n_iter_max)
        x_train, y_train = np.array(x_train), np.array(y_train)
        assert x_train.shape[0] == y_train.shape[0], 'x_train, y_train should have the same number of samples'
        assert x_train.shape[1] == self.layer_len[0], 'shape of x_train is not compatible with the first layer'

        pbar = tqdm(range(n_iter_max))

        for it in pbar:
            loss = 0  # reset loss to 0
            for ii in range(x_train.shape[0]):
                x_out = self(x_train[ii, :])  # accumulate loss
                loss += self.loss(x_out, y_train[ii, :])
                self.backprop()
                [layer.zero_grad(which=["xin"]) for layer in self.layers]  # reset gradient with respect to xin to zero
            self.loss_hist[it] = loss

            if (it % 10) == 0:
                pbar.set_postfix_str(f'loss : {loss:.3f}')
            if loss < loss_tol:
                pbar.set_postfix_str(f'loss: {loss:.3f}. Convergence has been attained!')
                self.loss_hist = self.loss_hist[:it]
                break
            self.gradient_descent_par()
            [layer.zero_grad(which=["weights", "bias"]) for layer in self.layers]  # reset gradients with respect to par to zero
