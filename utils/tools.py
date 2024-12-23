import numpy as np
import torch
import matplotlib.pyplot as plt
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path, cond_pred_model=None, path2_load=None):
        is_best = False
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, cond_pred_model, path2_load)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, cond_pred_model, path2_load)
            self.counter = 0
            is_best = True
        return is_best

    def save_checkpoint(self, val_loss, model, path, cond_pred_model, path2_load):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        if cond_pred_model is not None and path2_load is not None:
            torch.save(cond_pred_model.state_dict(), path2_load)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visual_1(true, preds=None, name=None,  dim=6):
    """
    Results visualization
    """
    col = 3
    row = math.ceil(dim / col)
    fig, axs = plt.subplots(row, col, figsize=(16, 12))
    for d in range(dim):
        axs[d // col, d % col].plot(preds[:, d], label='Prediction', linewidth=2)
        axs[d // col, d % col].plot(true[:, d], label='GroundTruth', linewidth=2)
        axs[d // col, d % col].legend()
        axs[d // col, d % col].set_title(f'Dimension {d+1}')
    
    if name is not None:
        print(f'save img to {name}')
        plt.savefig(name, bbox_inches='tight')
    plt.show()

def plot(target, forecast, prediction_length=192, dim=7, prediction_intervals=(50.0, 90.0), color='g', fname=None):
    # forecast: (100, 192, 8), target: (192, 8)
    pred_mean = np.mean(forecast, axis=0)   # (192, 8)
    pred_5 = np.percentile(forecast, 5, axis=0)
    pred_25 = np.percentile(forecast, 25, axis=0)
    pred_50 = np.percentile(forecast, 50, axis=0)
    pred_75 = np.percentile(forecast, 75, axis=0)
    pred_95 = np.percentile(forecast, 95, axis=0)
    col = 3
    row = math.ceil(dim / col)

    fig, axs = plt.subplots(row, col, figsize=(16, 12))

    # draw img for each dimension
    for d in range(dim):
        # 90% quantile interval
        axs[d // col, d % col].fill_between(np.arange(prediction_length), 
                                        pred_5[:, d], 
                                        pred_95[:, d], 
                                        color='lightgreen', alpha=0.5, label='90% Interval')
        
        # 50% quantile interval
        axs[d // col, d % col].fill_between(np.arange(prediction_length), 
                                        pred_25[:, d], 
                                        pred_75[:, d], 
                                        color='darkgreen', alpha=0.5, label='50% Interval')
        
        # mean
        axs[d // col, d % col].plot(pred_mean[:, d], color='black', label='Pred Mean', linewidth=2)
        
        # ground truth
        axs[d // col, d % col].plot(target[:, d], color='red', label='True', linewidth=2)
        axs[d // col, d % col].legend()
        axs[d // col, d % col].set_title(f'Dimension {d+1}')
    plt.tight_layout()
    if fname is not None:
        print(f'save img to {fname}')
        plt.savefig(fname)
    plt.show()

def loss_weight_linear_schedule(epoch, start_epoch=0, end_epoch=199, k_z_initial=1, k_z_final=0.01):
    if epoch <= start_epoch:
        return k_z_initial
    elif epoch > end_epoch:
        return k_z_final
    else:
        return k_z_initial - (k_z_initial - k_z_final) * (epoch - start_epoch) / (end_epoch - start_epoch) 

def loss_weight_exp_schedule(epoch, start_epoch=0, end_epoch=199, k_z_initial=1, k_z_final=0.01):
    if epoch <= start_epoch:
        return k_z_initial
    elif epoch > end_epoch:
        return k_z_final
    else:
        alpha = np.log(k_z_initial / k_z_final) / (end_epoch - start_epoch)
        l = - np.log(k_z_initial) / alpha
        decay = np.exp(-alpha * (epoch - start_epoch + l))
        return decay
