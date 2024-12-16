import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)   # 14, 128
        self.embed = nn.Embedding(n_steps, num_out) # 1001, 128
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)   # 8 -> (8, 128)
        # out = gamma.view(-1, self.num_out) * out

        out = gamma.view(t.size()[0], -1, self.num_out) * out
        return out


class ConditionalGuidedModel(nn.Module):
    def __init__(self, config, MTS_args):
        super(ConditionalGuidedModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1    # 1001
        self.cat_x = config.model.cat_x
        self.cat_y_pred = config.model.cat_y_pred
        # data_dim = 14   
        data_dim = MTS_args.enc_in * 2  # change this to adapt to more dataset

        self.lin1 = ConditionalLinear(data_dim, 128, n_steps)   # data_dim: 14
        self.lin2 = ConditionalLinear(128, 128, n_steps)    # n_steps: 1001, ConditionalLinear: linear 128->128, embed: 1001 -> 128
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = nn.Linear(128, MTS_args.enc_in)

    def forward(self, x, y_t, y_0_hat, t):  # enc_out (32, 96, 32), y_t (32, 240, 7), y_0_hat (32, 240, 7), t (32,)
        if self.cat_x:  # true
            if self.cat_y_pred: # true
                eps_pred = torch.cat((y_t, y_0_hat), dim=-1)    # (32, 240, 14)
            else:
                eps_pred = torch.cat((y_t, x), dim=2)
        else:
            if self.cat_y_pred:
                eps_pred = torch.cat((y_t, y_0_hat), dim=2)
            else:
                eps_pred = y_t
        if y_t.device.type == 'mps':    # cuda
            eps_pred = self.lin1(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            eps_pred = self.lin2(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

            eps_pred = self.lin3(eps_pred, t)
            eps_pred = F.softplus(eps_pred.cpu()).to(y_t.device)

        else:
            eps_pred = F.softplus(self.lin1(eps_pred, t))   # （32， 240， 128）
            eps_pred = F.softplus(self.lin2(eps_pred, t))   # （32， 240， 128）
            eps_pred = F.softplus(self.lin3(eps_pred, t))   # （32， 240， 128）
        eps_pred = self.lin4(eps_pred)  # （32， 240， 7）
        return eps_pred # （32， 240， 7）


# deterministic feed forward neural network
class DeterministicFeedForwardNeuralNetwork(nn.Module):

    def __init__(self, dim_in, dim_out, hid_layers,
                 use_batchnorm=False, negative_slope=0.01, dropout_rate=0):
        super(DeterministicFeedForwardNeuralNetwork, self).__init__()
        self.dim_in = dim_in  # dimension of nn input
        self.dim_out = dim_out  # dimension of nn output
        self.hid_layers = hid_layers  # nn hidden layer architecture
        self.nn_layers = [self.dim_in] + self.hid_layers  # nn hidden layer architecture, except output layer
        self.use_batchnorm = use_batchnorm  # whether apply batch norm
        self.negative_slope = negative_slope  # negative slope for LeakyReLU
        self.dropout_rate = dropout_rate
        layers = self.create_nn_layers()
        self.network = nn.Sequential(*layers)

    def create_nn_layers(self):
        layers = []
        for idx in range(len(self.nn_layers) - 1):
            layers.append(nn.Linear(self.nn_layers[idx], self.nn_layers[idx + 1]))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.nn_layers[idx + 1]))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
        layers.append(nn.Linear(self.nn_layers[-1], self.dim_out))
        return layers

    def forward(self, x):
        return self.network(x)


# early stopping scheme for hyperparameter tuning
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): Number of steps to wait after average improvement is below certain threshold.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement;
                           shall be a small positive value.
                           Default: 0
            best_score: value of the best metric on the validation set.
            best_epoch: epoch with the best metric on the validation set.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.early_stop = False

    def __call__(self, val_cost, epoch, verbose=False):

        score = val_cost

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch + 1
        elif score > self.best_score - self.delta:
            self.counter += 1
            if verbose:
                print("EarlyStopping counter: {} out of {}...".format(
                    self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch + 1
            self.counter = 0
