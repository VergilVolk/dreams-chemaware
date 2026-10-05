"""Static fail-closed contracts for the online native training entry."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "tasks/train_chemaware_online_packet_native.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    optimizer_constructors = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "torch"
        and node.func.value.attr == "optim"
    ]
    assert not optimizer_constructors, "online entry constructs a second torch optimizer"
    assert "optimizer = trainer.optimizers[0]" in source
    assert "optimizer_reloaded_from_checkpoint\": False" in source
    assert "formula_roles_2_3_4_untouched\": True" in source
    assert "model.step(batch, global_step)" in source
    assert "F.normalize(model(batch)" in source
    assert 'loaders["focused"] = DataLoader' in source
    assert 'loaders["replay"] = DataLoader' in source
    assert 'choices=("query_packet", "phasea_role_budget")' in source
    assert "allocation = allocate_role_steps(role_fractions, interval_length)" in source
    assert "enable_checkpointing=False" in source
    sbatch = (ROOT / "tasks/run_chemaware_online_packet_native.sbatch").read_text(
        encoding="utf-8",
    )
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "srun --export=ALL --preserve-env python -u tasks/train_chemaware_online_packet_native.py" in sbatch
    assert "--paired-reference phaseA_step-002000 --formula-role 2" in sbatch
    assert "tasks/freeze_chemaware_phasea_2pp_artifact.py" in sbatch
    assert "--online-schedule phasea_role_budget" in sbatch
    assert "#SBATCH --job-name=chem_rolebudget" in sbatch
    assert "role 3 remains unopened" in sbatch
    print("PASS: ChemAware continuous-Adam runtime contracts")


if __name__ == "__main__":
    main()
