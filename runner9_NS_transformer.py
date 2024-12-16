import argparse
import torch
from model9_NS_transformer.exp.exp_main import Exp_Main
from model9_NS_transformer.exp.exp_res_diffusion import Exp_Main_ResDiffusion
import random
import numpy as np
import setproctitle
import datetime
import os

if __name__ == '__main__':
    setproctitle.setproctitle('WXY_thread')

    parser = argparse.ArgumentParser(description='Non-stationary Transformers for Time Series Forecasting')

    # basic config
    parser.add_argument('--is_training', type=bool, default=True, help='status')
    parser.add_argument('--model_id', type=str, default='ETTh2_96_192', help='model id')    # NOTE
    parser.add_argument('--model', type=str, default='ETTh2_96_192',
                        help='model name, options: [ns_Transformer, Transformer]')  # NOTE

    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file') # TODO
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file') # TODO
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints, to be save after training')

    # forecasting task  # TODO
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    parser.add_argument('--k_z', type=float, default=1e-2, help='KL weight 1e-9')
    parser.add_argument('--k_cond', type=float, default=1, help='Condition weight')
    parser.add_argument('--d_z', type=int, default=8, help='KL weight')


    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')  # 32
    parser.add_argument('--test_batch_size', type=int, default=8, help='batch size of train input data')  # 32
    parser.add_argument('--patience', type=int, default=15, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate for diffusion model')
    parser.add_argument('--learning_rate_Cond', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu, set for one device')
    parser.add_argument('--use_multi_gpu', type=bool, default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[64, 64],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # CART related args # TODO
    parser.add_argument('--diffusion_config_dir', type=str, default='./model9_NS_transformer/configs/toy_8gauss.yml',
                        help='')

    # parser.add_argument('--cond_pred_model_dir', type=str,
    #                     default='./checkpoints/cond_pred_model_pertrain_NS_Transformer/checkpoint.pth', help='')
    parser.add_argument('--cond_pred_model_pertrain_dir', type=str,
                        default='./checkpoints/cond_pred_model_pertrain_NS_Transformer/checkpoint.pth', help='')

    parser.add_argument('--CART_input_x_embed_dim', type=int, default=32, help='feature dim for x in diffusion model')
    parser.add_argument('--mse_timestep', type=int, default=0, help='')

    parser.add_argument('--MLP_diffusion_net', type=bool, default=False, help='use MLP or Unet')

    # Some args for Ax (all about diffusion part)
    parser.add_argument('--timesteps', type=int, default=1000, help='args in diffuMTS')
    parser.add_argument('--diffusion_steps', type=int, default=1000, help='true diffusion steps in evaluation, keep it = timesteps normally')

    # exp args
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='use pretrained TMDM model pth to do evaluation')
    parser.add_argument('--pretrained_cond_model_path', type=str, default=None, help='use pretrained cond_pred_model pth to do evaluation')
    parser.add_argument('--not_training', action='store_true', help='not training')
    parser.add_argument('--use_res_diffusion', action='store_true', help='use res_diffusion')


    args = parser.parse_args()
    args.is_training = args.is_training and not args.not_training
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.seed == -1:
        fix_seed = np.random.randint(2147483647)
    else:
        fix_seed = args.seed

    print('Using seed:', fix_seed)
    print('current_time', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # use gpu
    # NOTE: use_multi_gpu, run the following code in terminal
    # python runner9_NS_transformer.py --use_multi_gpu True --devices 4,5,6,7
    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')    # [4, 5, 6, 7]
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]  # [4, 5, 6, 7]
            # args.gpu = args.device_ids[0]   # 0 when use multi_gpu, the system will use the first gpu to compute
            # if we choose [4, 5, 6, 7], set os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
            # then it will map 4,5,6,7 to 0, 1, 2, 3
            # args.gpu = min(len(args.device_ids) - 1, args.gpu )    # always use the last gpu
            args.gpu =  args.device_ids[0] 
        else:
            torch.cuda.set_device(args.gpu)

    print('Args in experiment:')
    print(args)

    # set experiment
    if not args.use_res_diffusion:
        print('use_tmdm')
        Exp = Exp_Main
    else:
        print('use res_diffusion')
        Exp = Exp_Main_ResDiffusion

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    if args.is_training:
        for ii in range(args.itr):  # number of experiment iterations
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, 
                ii,
                )

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            # exp.test_cond(setting)
            exp.test(setting)   # test=0 -> use the model trained in training, not directly load the model

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                      args.model,
                                                                                                      args.data,
                                                                                                      args.features,
                                                                                                      args.seq_len,
                                                                                                      args.label_len,
                                                                                                      args.pred_len,
                                                                                                      args.d_model,
                                                                                                      args.n_heads,
                                                                                                      args.e_layers,
                                                                                                      args.d_layers,
                                                                                                      args.d_ff,
                                                                                                      args.factor,
                                                                                                      args.embed,
                                                                                                      args.distil,
                                                                                                      args.des, 
                                                                                                      ii,
                                                                                                      )

        exp = Exp(args)  # set experiments

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test_cond(setting, test=1)
        exp.test(setting, test=1)

        if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)
        torch.cuda.empty_cache()