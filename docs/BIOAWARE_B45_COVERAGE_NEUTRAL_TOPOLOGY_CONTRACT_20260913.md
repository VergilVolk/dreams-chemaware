# BioAware B45 coverage-neutral topology contract

Date: 2026-09-13  
Status: development-only decisive test after the consumed B44 failure.

## Question

Does static KEGG/Rhea topology contain candidate-ranking information after the
catalogue-membership shortcut that caused the B44 failure is made impossible?

## Frozen design

- Universe: the six already opened B42 sources only.  B44 is never read.
- Evaluation: nested leave-source-out on all 860 negative-mode queries.
- Leakage control: each held source is removed wholesale; its truth identities
  and truth formulas are purged from fitting data.
- Training pair: truth and wrong candidate must have exactly the same
  `(catalogue member count, catalogue intersection)` signature.
- Candidate action: a candidate may challenge the DreaMS baseline only if it has
  exactly the same membership signature as that baseline.
- Real features: DreaMS spectral score plus KEGG/Rhea consensus log-degree mean
  and minimum.
- Controls: spectral-only plus three deterministic degree permutations within
  each query and membership-signature stratum.  The degree-pair multiset is
  preserved and only its candidate assignment is broken.
- Gate selection: inner leave-source-out only; lambda=2 harm penalty; every inner
  source must have nonnegative risk net.

## Pass gate

All conditions are required:

1. at least 400 queries have a coverage-neutral alternative;
2. at least 100 baseline errors are in principle recoverable;
3. overall Recall@1 gain is at least 3 pp;
4. formula- and identity-cluster CI lower bounds are positive;
5. corrected is greater than twice introduced;
6. every held source is nonnegative;
7. real degree assignment beats every degree permutation with a positive
   formula-cluster CI lower bound.

Failure terminates candidate-static topology.  Passing permits construction of
a new, separately sealed external test, but does not recover or tune the
consumed B44 result and does not establish sample-context BioAware.
