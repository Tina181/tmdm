from data_provider.data_factory import data_provider

from utils.tools import EarlyStopping, plot, loss_weight_linear_schedule, loss_weight_exp_schedule
from utils.metrics import metric

from model9_NS_transformer.ns_models import ns_Transformer
from model9_NS_transformer.exp.exp_basic import Exp_Basic
from model9_NS_transformer.diffusion_models import diffuMTS
# TODO change diffsuion_utils to diffusion_utils_1 -> not record every step prediction -> only y_0
from model9_NS_transformer.diffusion_models.diffusion_utils_1 import *

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

from multiprocessing import Pool
import CRPS.CRPS as pscore 

import warnings
import datetime

warnings.filterwarnings('ignore')

def ccc(id, pred, true):
    if id% 799 == 0:
        print(id, datetime.datetime.now())
    # print(id, datetime.datetime.now())
    res_box = np.zeros(len(true))
    for i in range(len(true)):
        res = pscore(pred[i], true[i]).compute()
        res_box[i] = res[0]
    return res_box


def log_normal(x, mu, var):
    """Logarithm of normal distribution with mean=mu and variance=var
       log(x|μ, σ^2) = loss = -0.5 * Σ log(2π) + log(σ^2) + ((x - μ)/σ)^2

    Args:
       x: (array) corresponding array containing the input
       mu: (array) corresponding array containing the mean
       var: (array) corresponding array containing the variance

    Returns:
       output: (array/float) depending on average parameters the result will be the mean
                            of all the sample losses or an array with the losses per sample
    """
    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    # return -0.5 * torch.sum(
    #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = diffuMTS.Model(self.args, self.device).float()
        # model = Transformer.Model(self.args).float()

        #NOTE if not training, load pretrained TMDM model
        # if not self.args.is_training:
        #     if self.args.pretrained_model_path is not None:
        #         model.load_state_dict(torch.load(self.args.pretrained_model_path))

        cond_pred_model = ns_Transformer.Model(self.args).float()
        cond_pred_model_train = ns_Transformer.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            cond_pred_model = nn.DataParallel(cond_pred_model, device_ids=self.args.device_ids)
            cond_pred_model_train = nn.DataParallel(cond_pred_model_train, device_ids=self.args.device_ids)
        return model, cond_pred_model, cond_pred_model_train

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self, mode='Model'):
        if mode == 'Model':
            # model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            model_optim = optim.Adam([{'params': self.model.parameters()}, {'params': self.cond_pred_model.parameters()}],
                                     lr=self.args.learning_rate)
        elif mode == 'Cond':
            model_optim = optim.Adam(self.cond_pred_model_train.parameters(), lr=self.args.learning_rate_Cond)
        else:
            model_optim = None
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        self.cond_pred_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)   # (32, 96, 7)
                batch_y = batch_y.float().to(self.device)   # (32, 240, 7)

                batch_x_mark = batch_x_mark.float().to(self.device)     # (32, 96, 4)
                batch_y_mark = batch_y_mark.float().to(self.device) # (32, 240, 4)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float() # (32, 192, 7)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)  # (32, 240, 7)
                # encoder - decoder
                if self.args.use_amp:   # false
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:  # false
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        n = batch_x.size(0) # B=32
                        num_timesteps = self.model.num_timesteps if not self.args.use_multi_gpu else self.model.module.num_timesteps
                        t = torch.randint(
                            low=0, high=num_timesteps, size=(n // 2 + 1,)
                        ).to(self.device)   # (17,)
                        t = torch.cat([t, num_timesteps - 1 - t], dim=0)[:n] # (32,)
                        # calculate diffusion mean (transformer)
                        _, y_0_hat_batch, KL_loss, z_sample = self.cond_pred_model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark)  # (32, 240, 7)
                        # calculate transformer loss
                        loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))

                        loss_vae_all = loss_vae + self.args.k_z * KL_loss
                        # y_0_hat_batch = z_sample

                        y_T_mean = y_0_hat_batch    # (32, 240, 7)
                        e = torch.randn_like(batch_y).to(self.device)   # (32, 240, 7)
                        
                        alphas_bar_sqrt = self.model.alphas_bar_sqrt if not self.args.use_multi_gpu else self.model.module.alphas_bar_sqrt
                        one_minus_alphas_bar_sqrt = self.model.one_minus_alphas_bar_sqrt if not self.args.use_multi_gpu else self.model.module.one_minus_alphas_bar_sqrt
                         # calculate diffusion forward process: step t
                        y_t_batch = q_sample(batch_y, y_T_mean, alphas_bar_sqrt,
                                             one_minus_alphas_bar_sqrt, t, noise=e)  # (32, 240, 7)
                        output = self.model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)    # epsilon theta (32, 240, 7)

                        loss = (e[:, -self.args.pred_len:, :] - output[:, -self.args.pred_len:, :]).square().mean() + self.args.k_cond * loss_vae_all
                loss = loss.detach().cpu()

                total_loss.append(loss)
        # after all valid batches
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting) # './checkpoints/ETTh2_96_192_ETTh2_96_192_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0'
        path2 = os.path.join(path, 'best_cond_model_dir/')  # './checkpoints/ETTh2_96_192_ETTh2_96_192_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/best_cond_model_dir/'
        path2_load = path2 + '/' + 'checkpoint.pth'  # './checkpoints/ETTh2_96_192_ETTh2_96_192_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth'

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(path2):
            os.makedirs(path2)

        time_now = time.time()

        train_steps = len(train_loader) # 372
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)


        model_optim = self._select_optimizer()


        criterion = self._select_criterion()    # MSELoss

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):

            # Training the diffusion part
            epoch_time = time.time()
            if self.args.k_cond_schedule is None:
                k_cond = self.args.k_cond
            elif self.args.k_cond_schedule == 'linear':
                k_cond = loss_weight_linear_schedule(epoch, start_epoch=0, end_epoch=self.args.train_epochs, k_cond_initial=1.3, k_cond_final=0.7)
            elif self.args.kcond_schedule == 'exp':
                k_cond = loss_weight_exp_schedule(epoch, start_epoch=0, end_epoch=self.args.train_epochs, k_z_initial=1.3, k_z_final=0.7)

            iter_count = 0
            train_loss = []
            self.model.train()
            self.cond_pred_model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)   # (32, 96, 7) , cuda: 6

                batch_y = batch_y.float().to(self.device)   # (32, 240, 7) -> 240 = 192+48
                batch_x_mark = batch_x_mark.float().to(self.device) # (32, 96, 4)
                batch_y_mark = batch_y_mark.float().to(self.device) # (32, 240, 4)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float() # (32, 192, 7)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)  # (32, 240, 7)

                # encoder - decoder
                if self.args.use_amp:   # false
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:  # false
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        n = batch_x.size(0) # 32
                        num_timesteps = self.model.num_timesteps if not self.args.use_multi_gpu else self.model.module.num_timesteps
                        t = torch.randint(
                            low=0, high=num_timesteps, size=(n // 2 + 1,)
                        ).to(self.device)   # (17,)
                        t = torch.cat([t, num_timesteps - 1 - t], dim=0)[:n] # (32,)
                        # y_0_hat_batch (32, 240, 7), z_sample: (32, 96, 512)
                        _, y_0_hat_batch, KL_loss, z_sample = self.cond_pred_model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark)
                        # batch_y (32, 240, 7), y_0_hat_batch (32, 240, 7)
                        loss_vae = log_normal(batch_y, y_0_hat_batch, torch.from_numpy(np.array(1)))

                        loss_vae_all = loss_vae + self.args.k_z * KL_loss
                        # y_0_hat_batch = z_sample

                        y_T_mean = y_0_hat_batch    # mean of diffusion limit distribution (32, 240, 7)
                        e = torch.randn_like(batch_y).to(self.device)   # (32, 240, 7)

                        # CARD diffusion, forward process
                        alphas_bar_sqrt = self.model.alphas_bar_sqrt if not self.args.use_multi_gpu else self.model.module.alphas_bar_sqrt
                        one_minus_alphas_bar_sqrt = self.model.one_minus_alphas_bar_sqrt if not self.args.use_multi_gpu else self.model.module.one_minus_alphas_bar_sqrt
                        
                        y_t_batch = q_sample(batch_y, y_T_mean, alphas_bar_sqrt,
                                             one_minus_alphas_bar_sqrt, t, noise=e)  # (32, 240, 7)

                        output = self.model(batch_x, batch_x_mark, batch_y, y_t_batch, y_0_hat_batch, t)    # (32, 240, 7)

                        # loss = (e[:, -self.args.pred_len:, :] - output[:, -self.args.pred_len:, :]).square().mean()
                        loss = (e - output).square().mean() + k_cond * loss_vae_all # 1 diffusion loss , 2 transformer loss
                        loss = loss.mean()
                        train_loss.append(loss.item())

                        if (i + 1) % 100 == 0:  # print every 100 batches
                            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                            speed = (time.time() - time_now) / iter_count
                            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                            iter_count = 0
                            time_now = time.time()

                        if self.args.use_amp:   # false
                            scaler.scale(loss).backward()
                            scaler.step(model_optim)
                            scaler.update()
                        else:
                            loss.backward() # backward loss of every batch
                            model_optim.step()

                        a = 0
            # one epoch finished            
            print("Epoch: {} cost time: {:.2f}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss) # compute mean of all batch losses in one epoch
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)


            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}  Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path, self.cond_pred_model, path2_load)

            if (math.isnan(train_loss)):
                break

            if early_stopping.early_stop:
                print("Early stopping")
                break


        # all epochs finished
        best_model_path = path + '/' + 'checkpoint.pth'
        best_cond_pred_model = path2_load
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))   # self.device: 'cuda:6'
        self.cond_pred_model.load_state_dict(torch.load(best_cond_pred_model, map_location=self.device))
        return self.model, self.cond_pred_model


    def test(self, setting, test=0):
        #####################################################################################################
        ########################## local functions within the class function scope ##########################

        def get_attribute(attr_name, model=self.model, use_multi_gpu=self.args.use_multi_gpu,):

            if use_multi_gpu:
                return getattr(model.module, attr_name)
            else:
                return getattr(model, attr_name)
       
        def store_gen_y_at_step_t(config, config_diff, idx, y_tile_seq):
            """
            Store generated y from a mini-batch to the array of corresponding time step.
            """
            current_t = get_attribute('diffusion_steps') - idx  # 0
            gen_y = y_tile_seq[idx].reshape(config.test_batch_size,
                                            int(config_diff.testing.n_z_samples / config_diff.testing.n_z_samples_depart),
                                            (config.label_len + config.pred_len),
                                            config.c_out).cpu().numpy() # (1, 100, 240, 862)
            # directly modify the dict value by concat np.array instead of append np.array gen_y to list
            # reduces a huge amount of memory consumption
            if len(gen_y_by_batch_list[current_t]) == 0:
                gen_y_by_batch_list[current_t] = gen_y
            else:
                gen_y_by_batch_list[current_t] = np.concatenate([gen_y_by_batch_list[current_t], gen_y], axis=0)
            return gen_y

        def compute_true_coverage_by_gen_QI(config, dataset_object, all_true_y, all_generated_y):
            n_bins = config.testing.n_bins  # 10
            quantile_list = np.arange(n_bins + 1) * (100 / n_bins)
            # compute generated y quantiles
            y_pred_quantiles = np.percentile(all_generated_y.squeeze(), q=quantile_list, axis=1)
            y_true = all_true_y.T
            quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0)
            # y_true_quantile_bin_count = np.bincount(y_true_quantile_membership)
            y_true_quantile_bin_count = np.array(
                [(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)])

            # combine true y falls outside of 0-100 gen y quantile to the first and last interval
            y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
            y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
            y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]
            # compute true y coverage ratio for each gen y quantile interval
            # y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object.test_n_samples
            y_true_ratio_by_bin = y_true_quantile_bin_count_ / dataset_object
            assert np.abs(
                np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
            qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()
            return y_true_ratio_by_bin, qice_coverage_ratio, y_true
        
        def compute_QICE(config, all_true_y, all_generated_y, dataset_object=0, y_true_quantile_bin_count_sum=0):
            # all_true_y: (688128, 100) , all_generated_y: (688128, 1)
            dataset_object +=all_true_y.shape[0]
            n_bins = config.testing.n_bins
            quantile_list = np.arange(n_bins + 1) * (100 / n_bins)  # array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])
            # compute generated y quantiles
            y_pred_quantiles = np.percentile(all_generated_y.squeeze(), q=quantile_list, axis=1)    # (11, 688128)
            y_true = all_true_y.T   # (1, 688128)
            quantile_membership_array = ((y_true - y_pred_quantiles) > 0).astype(int)   # (11, 688128)
            y_true_quantile_membership = quantile_membership_array.sum(axis=0)  # (688128,)  [7, 2, 6, ..., 8, 9, 4], min:0, max:11
            # y_true_quantile_bin_count = np.bincount(y_true_quantile_membership)  -> array([ 65422, 101242,  61711,  53008,  63884,  87083,  65635,  58612, 51925,  37671,  29083,  12852])
            y_true_quantile_bin_count = np.array( # [0, 1, 2, ..., 11]
                [(y_true_quantile_membership == v).sum() for v in np.arange(n_bins + 2)])   # (12,) count total number for every quantile interval 

            # combine true y falls outside of 0-100 gen y quantile to the first and last interval
            y_true_quantile_bin_count[1] += y_true_quantile_bin_count[0]
            y_true_quantile_bin_count[-2] += y_true_quantile_bin_count[-1]
            y_true_quantile_bin_count_ = y_true_quantile_bin_count[1:-1]    # (12,) -> (10,)
            y_true_quantile_bin_count_sum += y_true_quantile_bin_count_
            # compute true y coverage ratio for each gen y quantile interval
            # array([0.24219913, 0.08967954, 0.07703218, 0.09283738, 0.12655058, 0.09538196, 0.08517601, 0.07545834, 0.05474418, 0.0609407 ])
            y_true_ratio_by_bin = y_true_quantile_bin_count_sum / dataset_object   
            assert np.abs(
                np.sum(y_true_ratio_by_bin) - 1) < 1e-10, "Sum of quantile coverage ratios shall be 1!"
            qice_coverage_ratio = np.absolute(np.ones(n_bins) / n_bins - y_true_ratio_by_bin).mean()    # 0.03374994187127976
            return y_true_ratio_by_bin, qice_coverage_ratio, y_true, dataset_object, y_true_quantile_bin_count_sum

        def compute_PICP(config, y_true, all_gen_y, return_CI=False):
            """
            Another coverage metric.
            all_true_y: (64*8*192*7, 1), all_generated_y: (64*8*192*7, 100)
            """
            low, high = config.testing.PICP_range
            CI_y_pred = np.percentile(all_gen_y.squeeze(), q=[low, high], axis=1)   # (64*8*192*7, 100)-> (2, 64*8*192*7)
            # compute percentage of true y in the range of credible interval
            y_in_range = (y_true >= CI_y_pred[0]) & (y_true <= CI_y_pred[1])    # （64*8*192*7，）
            coverage = y_in_range.mean()    # sum/ 64*8*192*7
            if return_CI:
                return coverage, CI_y_pred, low, high
            else:
                return coverage, low, high

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            if self.args.pretrained_model_path is not None:
                print('load pretrained model and cond model')
                if not self.args.use_multi_gpu:
                    # use singel gpu
                    self.model.load_state_dict(torch.load(self.args.pretrained_model_path, map_location=self.device))
                    self.cond_pred_model.load_state_dict(torch.load(self.args.pretrained_cond_model_path,
                                                                    map_location=self.device))
                else:
                    # use multi gpu
                    self.model.module.load_state_dict(torch.load(self.args.pretrained_model_path, map_location=self.device))
                    self.cond_pred_model.module.load_state_dict(torch.load(self.args.pretrained_cond_model_path,
                                                                            map_location=self.device))
            else:
                print('load best model')
                self.model.load_state_dict(
                    torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location=self.device))
                self.cond_pred_model.load_state_dict(torch.load(os.path.join(os.path.join(self.args.checkpoints, setting),
                                                                            'best_cond_model_dir') + '/' + 'checkpoint.pth',
                                                                map_location=self.device))  # need to set device: "cuda:0"

        preds_save_final = []
        trues_save_final = []
        history_trues_save_final = []
        coverage_list, dataset_object, y_true_quantile_bin_count_sum = [], 0, 0
        CRPS_list, CRPS_sum_list = [], []
        mae_list, mse_list, rmse_list, mape_list, mspe_list = [], [], [], [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        minibatch_sample_start = time.time()

        self.model.eval()
        self.cond_pred_model.eval()
        num_timesteps = get_attribute('num_timesteps')
        diffusion_steps = get_attribute('diffusion_steps')
        model_args = get_attribute('args')
        diffusion_config = get_attribute('diffusion_config')
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                gen_y_by_batch_list = [[] for _ in range(diffusion_steps + 1)]
                y_se_by_batch_list = [[] for _ in range(diffusion_steps + 1)]

                batch_x = batch_x.float().to(self.device)   # (8, 96, 7)
                batch_y = batch_y.float().to(self.device)   # (8, 240, 7)


                batch_x_mark = batch_x_mark.float().to(self.device) # (8, 96, 4)
                batch_y_mark = batch_y_mark.float().to(self.device) # (8, 240, 4)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float() # (8, 192, 7) self.label_len: 48
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)  # (8, 240, 7)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        _, y_0_hat_batch, _, z_sample = self.cond_pred_model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark)  # y_0_hat_batch: (8, 240, 7), z_sample (9, 96, 512)
                        # sample num: repeat_n = 100
                        repeat_n = int(
                            diffusion_config.testing.n_z_samples / diffusion_config.testing.n_z_samples_depart)   # 100/1 = 100
                        y_0_hat_tile = y_0_hat_batch.repeat(repeat_n, 1, 1, 1)  # (100, 8, 240, 7)
                        y_0_hat_tile = y_0_hat_tile.transpose(0, 1).flatten(0, 1).to(self.device)   # (800, 240, 7)
                        y_T_mean_tile = y_0_hat_tile    # (800, 240, 7)
                        x_tile = batch_x.repeat(repeat_n, 1, 1, 1)  # (100, 8, 240, 7)
                        x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)   # (800, 96, 7)

                        x_mark_tile = batch_x_mark.repeat(repeat_n, 1, 1, 1)    # (100, 8, 240, 7)
                        x_mark_tile = x_mark_tile.transpose(0, 1).flatten(0, 1).to(self.device) # (800, 96, 7)

                        gen_y_box = []
                        for _ in range(diffusion_config.testing.n_z_samples_depart): # 1
                            for _ in range(diffusion_config.testing.n_z_samples_depart): # 1 sample 100 times
                                alphas = get_attribute('alphas')
                                one_minus_alphas_bar_sqrt = get_attribute('one_minus_alphas_bar_sqrt')
                                # t = torch.tensor([2]).to(self.device)
                                # eps_theta = self.model(x_tile, x_mark_tile, 0, y_0_hat_tile, y_0_hat_tile, t).to(self.device).detach()
                                y_tile_seq = p_sample_loop(self.model, x_tile, x_mark_tile, y_0_hat_tile, y_T_mean_tile,
                                                           diffusion_steps,
                                                           alphas, one_minus_alphas_bar_sqrt, 
                                                            self.args.use_multi_gpu,
                                                           ) # list, len: 1001

                            # TODO change gen_y, only store the last step y_0
                            # gen_y = store_gen_y_at_step_t(config=model_args,
                            #                               config_diff=diffusion_config,
                            #                               idx=diffusion_steps, y_tile_seq=y_tile_seq)  # (8, 100, 240, 7)
                            gen_y = y_tile_seq[-1].reshape(model_args.test_batch_size,
                                            int(diffusion_config.testing.n_z_samples / diffusion_config.testing.n_z_samples_depart),
                                            (model_args.label_len + model_args.pred_len),
                                            model_args.c_out).cpu().numpy()
                            gen_y_box.append(gen_y)
                        outputs = np.concatenate(gen_y_box, axis=1) # (8, 100, 240, 7)

                        f_dim = -1 if self.args.features == 'MS' else 0     # 0
                        outputs = outputs[:, :, -self.args.pred_len:, f_dim:]   # (8, 100, 192, 7)
                        history_true = batch_x.detach().cpu().numpy()
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)      # (8, 192, 7)
                        batch_y = batch_y.detach().cpu().numpy()    # (8, 192, 7)

                        pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze() # (8, 100, 192, 7)
                        true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()   # (8, 192, 7)

                        # below we compute metrics, without saving total preds
                        preds = np.array(pred)     # (8, 100, 192, 7)
                        trues = np.array(true)  # (8, 192, 7)
                        history_trues = np.array(history_true)  # (8, 96, 7)

                        if i<5: # save the first 5 batch preds, trues, history_trues
                            preds_save_final.append(preds)
                            trues_save_final.append(trues)
                            history_trues_save_final.append(history_trues)
                        
                        preds = np.expand_dims(preds, axis=0)   # (1, 8, 100, 192, 7)
                        trues = np.expand_dims(trues, axis=0)   # (1, 8, 192, 7)
                        preds_save = preds
                        trues_save = trues

                        # calculate preds mean
                        preds_ns = np.mean(preds, axis=2)   # (1, 8, 100, 192, 7) -> # (1, 8, 192, 7)
                        preds_ns = preds_ns.reshape(-1, preds_ns.shape[-2], preds_ns.shape[-1]) # (1, 8, 192, 7) -> (1*8, 192, 7)
                        trues_ns = trues.reshape(-1, trues.shape[-2], trues.shape[-1])   # (1*8, 192, 7)
                        mae, mse, _ , mape, mspe = metric(preds_ns, trues_ns)
                        mae_list.append(mae)
                        mse_list.append(mse)
                        # rmse_list.append(rmse)
                        mape_list.append(mape)
                        mspe_list.append(mspe)

                        # reshape
                        preds = preds.reshape(-1, preds.shape[-3], preds.shape[-2] * preds.shape[-1])   # (1, 8, 100, 192, 7) -> (1*8, 100, 192*7)
                        preds = preds.transpose(0, 2, 1)    # (1*8, 192*7, 100)
                        preds = preds.reshape(-1, preds.shape[-1])      # (1*8*192*7, 100)

                        trues = trues.reshape(-1, 1, trues.shape[-2] * trues.shape[-1])   
                        trues = trues.transpose(0, 2, 1)    
                        trues = trues.reshape(-1, trues.shape[-1])      # (1*8*192*7, 1)

                        # calculate picp
                        coverage, _, _ = compute_PICP(config=diffusion_config, y_true=trues.T, all_gen_y=preds)
                        coverage_list.append(coverage)

                        # calculate qice
                        y_true_ratio_by_bin, qice_coverage_ratio, y_true, dataset_object, y_true_quantile_bin_count_sum = compute_QICE(config=diffusion_config,
                        dataset_object=dataset_object, all_true_y=trues, all_generated_y=preds, y_true_quantile_bin_count_sum=y_true_quantile_bin_count_sum)

                        # calculate CRPS
                        pool = Pool(processes=8)
                        all_res = []
                        pred = preds_save.reshape(-1, preds_save.shape[-3], preds_save.shape[-2], preds_save.shape[-1])     # (8, 100, 192, 7)
                        true = trues_save.reshape(-1, trues_save.shape[-2], trues_save.shape[-1])   # (8, 192, 7)
                        for i in range(preds_save.shape[-1]): # Dimension 7
                            p_in = pred[:, :, :, i]     # (8, 100, 192)
                            p_in = p_in.transpose(0, 2, 1)  # (8, 192, 100)
                            p_in = p_in.reshape(-1, p_in.shape[-1]) # (8*192, 100)
                            t_in = true[:, :, i]    # (8, 192)
                            t_in = t_in.reshape(-1) # (8 * 192,)
                            all_res.append(pool.apply_async(ccc, args=(i, p_in, t_in))) # p_in (512*192, 100), t_in (512 * 192,)
                        p_in = np.sum(pred, axis=-1)    # (8, 100, 192, 7) -> (8, 100, 192)
                        p_in = p_in.transpose(0, 2, 1)  # (8, 192, 100)
                        p_in = p_in.reshape(-1, p_in.shape[-1]) # (8 * 192, 100)
                        t_in = np.sum(true, axis=-1)    # (8 , 192)
                        t_in = t_in.reshape(-1) # (8 * 192,)
                        CRPS_sum_f = pool.apply_async(ccc, args=(8, p_in, t_in))  # NOTE: calculate CRPS_sum  
                        pool.close()
                        pool.join()
                        all_res_get = []
                        for i in range(len(all_res)):   # Dimension 7
                            all_res_get.append(all_res[i].get())
                        all_res_get = np.array(all_res_get) # (7, 8*192)

                        CRPS_0 = np.mean(all_res_get, axis=0).mean(axis=0)  # (1*8*192,)
                        CRPS_list.append(CRPS_0)
                        CRPS_sum_0 = CRPS_sum_f.get()   # (test_group_size * 8 * 192)
                        CRPS_sum_list.append(CRPS_sum_0)

                # after 5 batch
                if i % 5 == 0 and i != 0:
                    print('Testing: %d/%d cost time: %.2f min' % (
                        i, len(test_loader), (time.time() - minibatch_sample_start) / 60))
                    minibatch_sample_start = time.time()

        # after all test batches
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'pred_5.npy', preds_save_final)
        np.save(folder_path + 'true_5.npy', trues_save_final)
        np.save(folder_path + 'history_true_5.npy', history_trues_save_final)

        target = np.concatenate((history_trues_save_final[0][0], trues_save_final[0][0]), axis=0)
        history_plot = history_trues_save_final[0][0]
        history_plot_repeat = np.repeat(history_plot.reshape(1, history_plot.shape[-2], history_plot.shape[-1]), 100, axis=0)
        forecast = np.concatenate((history_plot_repeat, preds_save_final[0][0]), axis=1)
        plot(target=target, forecast=forecast, prediction_length=self.args.pred_len + self.args.seq_len, 
             dim=6, fname=folder_path + 'img.png')

        # compute all ns metrics
        mse, mae, rmse, mape, mspe = np.mean(mse_list), np.mean(mae_list), np.sqrt(np.mean(mse_list)), np.mean(mape_list), np.mean(mspe_list)
        print('NT metric: mse:{:.4f}, mae:{:.4f} , rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(mse, mae, rmse, mape, mspe))
        
        # compute CARD metrics
        PICP = np.mean(coverage_list)
        print('calculate PICP each batch {:.4f}%'.format(PICP * 100))

        QICE = qice_coverage_ratio
        print('calculate QICE each batch {:.4f}%'.format(QICE * 100))

        # compute CRPS, CRPS_sum
        CRPS = np.mean([np.mean(CRPS_0) for CRPS_0 in CRPS_list])
        CRPS_sum = np.mean(CRPS_sum_list)
        print('CRPS', CRPS.mean(), '\n', 'CRPS_sum', CRPS_sum.mean())

        # write metrics in txt file
        f = open(os.path.join(folder_path, 'metrics.txt'), 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{} \n'.format(mse, mae, rmse, mape, mspe))
        f.write('PICP:{}, QICE:{} \n'.format(PICP * 100, QICE * 100))
        f.write('CRPS:{}, CRPS_sum:{} \n'.format(CRPS.mean(), CRPS_sum.mean()))
        f.write('\n')
        f.write('\n')
        f.close()

        # write metrics in txt file
        f = open('result.txt', 'a')
        f.write(f'datetime: {datetime.datetime.now()} \n')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{} \n'.format(mse, mae, rmse, mape, mspe))
        f.write('PICP:{}, QICE:{} \n'.format(PICP * 100, QICE * 100))
        f.write('CRPS:{}, CRPS_sum:{} \n'.format(CRPS.mean(), CRPS_sum.mean()))
        f.write('\n')
        f.write('\n')
        f.close()


        np.save(folder_path + 'metrics_new.npy',
                np.array([mse, mae, rmse, mape, mspe, qice_coverage_ratio * 100, coverage * 100, CRPS_0, CRPS_sum]))

        # np.save("./results/{}.npy".format(self.args.model_id), np.array(mse))
        # np.save("./results/{}_Ntimes.npy_new".format(self.args.model_id),
        #         np.array([mse, mae, rmse, mape, mspe, qice_coverage_ratio * 100, coverage * 100, CRPS_0, CRPS_sum]))

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load: # load best model 
            
            if self.args.pretrained_model_path is not None:
                self.model.load_state_dict(torch.load(self.args.pretrained_model_path, map_location=self.device))
            else:
                path = os.path.join(self.args.checkpoints, setting)
                best_model_path = path + '/' + 'checkpoint.pth'
                self.model.load_state_dict(torch.load(best_model_path), map_location=self.device)

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
