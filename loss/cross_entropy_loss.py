import numpy as np
from loss.loss_template import Loss

class CrossEntropyLoss(Loss):
    def get_loss(self):
        """
        Compute the cross-entropy loss.
        """
        # Stabilize the softmax calculation by subtracting the max value
        y_shifted = self.y - np.max(self.y)
        exp_y_shifted = np.exp(y_shifted)
        softmax = exp_y_shifted / np.sum(exp_y_shifted)

        # Cross-entropy loss
        self.loss = -np.log(softmax[self.y_train[0]])

    def get_dloss_dy(self):
        """
        Compute the gradient of the cross-entropy loss with respect to y.
        """
        # Stabilize the softmax calculation by subtracting the max value
        y_shifted = self.y - np.max(self.y)
        exp_y_shifted = np.exp(y_shifted)
        softmax = exp_y_shifted / np.sum(exp_y_shifted)

        # Gradient of the loss with respect to y
        self.dloss_dy = softmax.copy()
        self.dloss_dy[self.y_train] -= 1
