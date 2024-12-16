import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())  # 检查CUDA设备数量
print(torch.cuda.is_available())  # 检查CUDA是否可用

