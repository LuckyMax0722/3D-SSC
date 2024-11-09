import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(device)
# 获取当前使用的显卡编号
current_device = torch.cuda.current_device()

# 获取当前显卡的名称
device_name = torch.cuda.get_device_name(current_device)

print(f"当前使用的显卡编号: {current_device}")
print(f"当前使用的显卡名称: {device_name}")
