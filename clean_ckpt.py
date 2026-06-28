"""
将 ssl_model.ckpt 转换为零依赖的纯权重文件。

解决两个服务器问题：
  1. WindowsPath → 用自定义 Unpickler 转为 PurePath
  2. msml 模块依赖 → 只保存 state_dict，丢弃所有元数据

用法：
  cd D:\DreaMS
  python clean_ckpt.py

输出：
  dreams/models/pretrained/ssl_model_state_only.pt  （纯权重，约 0.46 GB）
"""

import pickle
import torch
from pathlib import PurePath

# ==============================================================================
# 自定义 Unpickler：将 WindowsPath 转为跨平台的 PurePath
# ==============================================================================
class CrossPlatformUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'pathlib' and name == 'WindowsPath':
            return PurePath
        if module == 'pathlib' and name == 'PosixPath':
            return PurePath
        return super().find_class(module, name)


# ==============================================================================
# 主流程
# ==============================================================================
ckpt_in = 'dreams/models/pretrained/ssl_model.ckpt'
ckpt_out = 'dreams/models/pretrained/ssl_model_state_only.pt'

print(f'Loading: {ckpt_in}')
with open(ckpt_in, 'rb') as f:
    unp = CrossPlatformUnpickler(f)
    data = unp.load()

# 提取 state_dict
if isinstance(data, dict) and 'state_dict' in data:
    state_dict = data['state_dict']
    print(f'  Extracted state_dict: {len(state_dict)} parameters')
else:
    state_dict = data
    print(f'  Data is raw state_dict (keys: {len(state_dict) if isinstance(state_dict, dict) else "N/A"})')

# 保存为纯权重文件
torch.save(state_dict, ckpt_out)

import os
size_gb = os.path.getsize(ckpt_out) / 1e9
print(f'Saved: {ckpt_out} ({size_gb:.2f} GB)')
print('Done — zero dependency, server-ready.')
