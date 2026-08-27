"""
SMILES 到结构图转换模块
"""

import io
import base64
from rdkit import Chem
from rdkit.Chem import Draw, AllChem


def smiles_to_image(smiles: str, size: int = 300) -> bytes:
    """
    将 SMILES 转换为分子结构图（PNG）

    参数:
        smiles: SMILES 字符串
        size: 图片大小（像素）

    返回:
        PNG 图片的二进制数据
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 生成 2D 坐标
        AllChem.Compute2DCoords(mol)

        # 绘制分子
        img = Draw.MolToImage(mol, size=(size, size))

        # 转换为字节
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

    except Exception as e:
        print(f"Error converting SMILES to image: {e}")
        return None


def smiles_to_base64(smiles: str, size: int = 300) -> str:
    """
    将 SMILES 转换为 base64 编码的图片（可直接用于 HTML img 标签）

    参数:
        smiles: SMILES 字符串
        size: 图片大小

    返回:
        base64 编码的图片字符串
    """
    img_bytes = smiles_to_image(smiles, size)
    if img_bytes:
        return base64.b64encode(img_bytes).decode('utf-8')
    return None


def get_molecule_info(smiles: str) -> dict:
    """
    获取分子的化学信息

    参数:
        smiles: SMILES 字符串

    返回:
        分子信息字典
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        from rdkit.Chem import Descriptors, Crippen

        return {
            'molecular_weight': f"{Descriptors.MolWt(mol):.2f}",
            'logp': f"{Crippen.MolLogP(mol):.2f}",
            'num_h_donors': Descriptors.NumHDonors(mol),
            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_heavy_atoms': Descriptors.HeavyAtomCount(mol),
        }

    except Exception as e:
        print(f"Error getting molecule info: {e}")
        return None
