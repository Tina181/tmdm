import numpy as np

# file_path = 'results/ECL_96_96_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/metrics.npy'
file_path = 'results_save1/ETTh2_96_192_ETTh2_96_192_custom_ftM_sl96_ll48_pl192_dm512_nh8_el2_dl1_df2048_fc3_ebtimeF_dtTrue_Exp_0/metrics.npy'

data = np.load(file_path)
# print(f'mae, mse, rmse, mape, mspe: {data}')
metrics_names = ['mae', 'mse', 'rmse', 'mape', 'mspe', 'QICE', 'PICP', 'CRPS', 'CRPS_sum']

# 打印每个指标及其对应的值
for name, value in zip(metrics_names, data):
    print(f'{name}: {value:.6f}')  # 保留6位小数