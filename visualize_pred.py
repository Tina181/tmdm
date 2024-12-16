import numpy as np
import matplotlib.pyplot as plt
import os
import math

def plot(target, forecast, prediction_length=192, dim=16, prediction_intervals=(50.0, 90.0), color='g', fname=None):
    # forecast: (100, 192, 8), target: (192, 8)
    pred_mean = np.mean(forecast, axis=0)   # (192, 8)
    pred_5 = np.percentile(forecast, 5, axis=0)
    pred_25 = np.percentile(forecast, 25, axis=0)
    pred_50 = np.percentile(forecast, 50, axis=0)
    pred_75 = np.percentile(forecast, 75, axis=0)
    pred_95 = np.percentile(forecast, 95, axis=0)
    row = math.ceil(dim / 4)
    col = 4

    # 创建子图
    fig, axs = plt.subplots(row, col, figsize=(16, 12))

    # 绘制每个维度的图
    for d in range(dim):
        # 90% 分位数的色块（更深的颜色）
        axs[d // col, d % col].fill_between(np.arange(prediction_length), 
                                        pred_5[:, d], 
                                        pred_95[:, d], 
                                        color='lightgreen', alpha=0.5, label='90% Interval')
        
        # 50% 分位数的色块（更浅的颜色）
        axs[d // col, d % col].fill_between(np.arange(prediction_length), 
                                        pred_25[:, d], 
                                        pred_75[:, d], 
                                        color='darkgreen', alpha=0.5, label='50% Interval')
        
        # 绘制预测均值
        axs[d // col, d % col].plot(pred_mean[:, d], color='black', label='Pred Mean')
        
        # 绘制真实值
        axs[d // col, d % col].plot(target[:, d], color='red', label='True')
        
        axs[d // col, d % col].legend()
        axs[d // col, d % col].set_title(f'Dimension {d+1}')
    plt.tight_layout()
    if fname is not None:
        print(f'save img to {fname}')
        plt.savefig(fname)
    plt.show()

# TODO change results_path to draw img on different datasets
# results_path = 'results/Exchage_96_192_Exchage_96_192_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/'
# pred = np.load(os.path.join(results_path, 'pred.npy'))  # 形状为 (165, 8, 100, 192, 8)
# true = np.load(os.path.join(results_path, 'true.npy'))  # 形状为 (165, 8, 192, 8)
# dim = 8

# results_path = 'results/ETTh2_96_192_ETTh2_96_192_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0'
# pred = np.load(os.path.join(results_path, 'pred.npy'))
# true = np.load(os.path.join(results_path, 'true.npy'))
# dim = 7

# results_path = 'results/RES_ILI_36_36_large_model_RES_ILI_36_36_large_model_custom_ftM_sl36_ll16_pl36_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0'
# pred = np.load(os.path.join(results_path, 'pred.npy'))
# true = np.load(os.path.join(results_path, 'true.npy'))
# dim = 7  # draw first 8 dims of the pred, notice some dataset dim < 8
# prediction_length = true.shape[2]

results_path = 'results/ETTm2_96_192_ETTm2_96_192_ETTm2_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0'
pred = np.load(os.path.join(results_path, 'pred.npy'))
true = np.load(os.path.join(results_path, 'true.npy'))
dim = 7  # draw first 8 dims of the pred, notice some dataset dim < 8
prediction_length = true.shape[2]

# 假设我们只绘制第一个样本的数据
sample_index = 0

# 绘制图像
plot(target=true[0][sample_index], forecast=pred[0][sample_index], dim=dim, prediction_length=prediction_length, fname=results_path + '/img.png')
