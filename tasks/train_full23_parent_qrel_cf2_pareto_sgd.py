#!/usr/bin/env python
"""Repair CF2 training so the executed step preserves the MGDA direction.

The candidate evidence, Full23 v1 parent, five global formula outer folds,
random seeds, losses, learning rate and fixed 20-epoch budget are identical to
the completed CF2 experiment.  The sole scientific change is optimization:
the closed-form two-task MGDA direction is applied directly with plain SGD,
shared L2 is included inside both task objectives, and same-batch Armijo
backtracking rejects any proposal that raises either source objective.

This is a consumed-development repair motivated by the observed mismatch
between MGDA and AdamW.  It is not an untouched external confirmation.
"""
from __future__ import annotations

import train_full23_parent_qrel_cf2_mgda_residual as cf2
import train_full23_parent_query_relative_mgda_residual as parent_qrel


METHOD = "full23_parent_qrel_cf2_pareto_sgd_residual"
SCHEMA = "full23_parent_qrel_cf2_pareto_sgd_residual_v1"
STATUS = "FULL23_PARENT_QREL_CF2_PARETO_SGD_RESIDUAL_DEVELOPMENT_COMPLETE"
CHECKPOINT_NAME = "full23_parent_qrel_cf2_pareto_sgd_residual.pt"


def main() -> None:
    cf2.run(
        cf2.arguments(),
        fit_delta=parent_qrel.fit_delta_pareto_sgd,
        method=METHOD,
        schema=SCHEMA,
        status=STATUS,
        checkpoint_name=CHECKPOINT_NAME,
        delta_semantics=(
            "released_parent + zero_initialized_linear_CF2_delta + "
            "direct_Pareto_SGD_backtracking"
        ),
        task_gradient_combination=(
            "closed_form_two_task_MGDA_direct_plain_SGD_"
            "same_batch_Armijo_backtracking"
        ),
        run_label="Full23 CF2 Pareto-SGD",
    )


if __name__ == "__main__":
    main()
