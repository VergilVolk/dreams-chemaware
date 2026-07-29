"""
将 Keras HDF5 MS2DeepScore 模型转换为 PyTorch 格式

模型架构 (从 HDF5 权重反推):
  Input: 9948-dim (binned spectrum from MS2DeepScore SpectrumBinner)
  Base network (shared, both inputs go through same layers):
    Dense1: 9948 → 500 + ReLU
    BatchNorm1: 500
    Dense2: 500 → 500 + ReLU
    BatchNorm2: 500
    Embedding: 500 → 200
  Output: cosine_similarity(emb_a, emb_b) ∈ [-1, 1]

用法:
  python tasks/convert_ms2ds_model.py
  输出: data/models/MS2DeepScore_allGNPSpositive_10k_500_500_200.pt
"""
import h5py
import numpy as np
import torch
import torch.nn as nn


class MS2DSSiamese(nn.Module):
    """Siamese network matching the Keras MS2DeepScore architecture"""

    def __init__(self, input_dim=9948, hidden=500, embedding=200):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden, momentum=0.99, eps=0.001)
        self.dense2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden, momentum=0.99, eps=0.001)
        self.embedding = nn.Linear(hidden, embedding)

    def forward_one(self, x):
        """Forward pass for a single spectrum"""
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.embedding(x)
        return x

    def forward(self, x_a, x_b):
        """Forward pass for a pair — returns cosine similarity"""
        emb_a = self.forward_one(x_a)
        emb_b = self.forward_one(x_b)
        # Cosine similarity (Keras cosine_similarity layer)
        cos = nn.functional.cosine_similarity(emb_a, emb_b, dim=1)
        return cos


def convert_weights(hdf5_path, pt_path):
    """Read Keras weights from HDF5, build PyTorch model, transfer weights"""
    f = h5py.File(hdf5_path, 'r')
    mw = f['model_weights']['base']

    # Read Keras weights
    w = {}

    # Dense1: Keras kernel is (in, out), PyTorch Linear weight is (out, in)
    w['dense1.weight'] = torch.from_numpy(np.array(mw['dense1']['kernel:0']).T.copy())
    w['dense1.bias'] = torch.from_numpy(np.array(mw['dense1']['bias:0']).copy())

    # BatchNorm1
    w['bn1.weight'] = torch.from_numpy(np.array(mw['normalization1']['gamma:0']).copy())
    w['bn1.bias'] = torch.from_numpy(np.array(mw['normalization1']['beta:0']).copy())
    w['bn1.running_mean'] = torch.from_numpy(np.array(mw['normalization1']['moving_mean:0']).copy())
    w['bn1.running_var'] = torch.from_numpy(np.array(mw['normalization1']['moving_variance:0']).copy())

    # Dense2
    w['dense2.weight'] = torch.from_numpy(np.array(mw['dense2']['kernel:0']).T.copy())
    w['dense2.bias'] = torch.from_numpy(np.array(mw['dense2']['bias:0']).copy())

    # BatchNorm2
    w['bn2.weight'] = torch.from_numpy(np.array(mw['normalization2']['gamma:0']).copy())
    w['bn2.bias'] = torch.from_numpy(np.array(mw['normalization2']['beta:0']).copy())
    w['bn2.running_mean'] = torch.from_numpy(np.array(mw['normalization2']['moving_mean:0']).copy())
    w['bn2.running_var'] = torch.from_numpy(np.array(mw['normalization2']['moving_variance:0']).copy())

    # Embedding
    w['embedding.weight'] = torch.from_numpy(np.array(mw['embedding']['kernel:0']).T.copy())
    w['embedding.bias'] = torch.from_numpy(np.array(mw['embedding']['bias:0']).copy())

    f.close()

    # Build PyTorch model
    input_dim = w['dense1.weight'].shape[1]  # 9948
    hidden = w['dense1.weight'].shape[0]      # 500
    emb_dim = w['embedding.bias'].shape[0]    # 200
    model = MS2DSSiamese(input_dim, hidden, emb_dim)

    # Load state dict
    model.load_state_dict(w)
    model.eval()

    # Verify
    print(f'Model architecture:')
    print(f'  Input dim:  {input_dim}')
    print(f'  Hidden dim: {hidden}')
    print(f'  Embedding:  {emb_dim}')
    for name, param in model.named_parameters():
        print(f'  {name}: {param.shape}')

    # Save
    torch.save({'model_state_dict': model.state_dict(),
                'input_dim': input_dim, 'hidden': hidden, 'embedding': emb_dim}, pt_path)
    print(f'\nSaved: {pt_path}')

    # Quick sanity check
    model.eval()
    with torch.no_grad():
        x = torch.randn(4, input_dim)
        sim = model(x, x)  # Same input → should be ~1.0
        print(f'Sanity check: cosine(x, x) = {sim.tolist()}')
        x2 = torch.randn(4, input_dim)
        sim2 = model(x, x2)
        print(f'Sanity check: cosine(x, rand) = {sim2.tolist()}')

    return model


if __name__ == '__main__':
    import os
    hdf5_path = 'data/models/MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5'
    pt_path = 'data/models/MS2DeepScore_allGNPSpositive_10k_500_500_200.pt'
    if not os.path.exists(hdf5_path):
        print(f'ERROR: {hdf5_path} not found')
        exit(1)
    convert_weights(hdf5_path, pt_path)
