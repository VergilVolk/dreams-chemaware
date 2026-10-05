"""Noise-owned native DreaMS runtime contracts."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "tasks/train_noise_reference_native.py"
text = path.read_text(encoding="utf-8")
tree = ast.parse(text)

assert "Continue official DreaMS on the frozen Noise reference triplets" in text
assert "from dreams.models.heads.heads import ContrastiveHead" in text
assert "from dreams.utils.data import ContrastiveSpectraDataset" in text
assert "model = ContrastiveHead(" in text
assert "ContrastiveSpectraDataset(" in text
assert "torch.optim.Adam(" not in text
assert "F.relu" not in text and "clamp_min" not in text
assert '"--checkpoint-save-weights-only", action="store_true"' in text
assert '"--no-save-last-checkpoint", action="store_true"' in text
assert '"--keep-final-partial-batch", action="store_true"' in text
assert "save_last=not args.no_save_last_checkpoint" in text
assert "save_weights_only=args.checkpoint_save_weights_only" in text
assert '"status": "NOISE_REFERENCE_NATIVE_TRAINING_COMPLETE"' in text

progress_class = next(
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "SlurmLineProgress"
)
batch_end = next(
    node for node in progress_class.body
    if isinstance(node, ast.FunctionDef) and node.name == "on_train_batch_end"
)
assert len(batch_end.body) == 1 and isinstance(batch_end.body[0], ast.If)
conditional = batch_end.body[0]
assert any(isinstance(node, ast.Assign) for node in conditional.body)
assert any(isinstance(node, ast.Expr) for node in conditional.body)

print("[test_noise_reference_native_runtime] PASS")
