import json
r = json.load(open(r'data/validation/GLM_p0_spectral_coordinate/report.json/report.json'))
print('status:', r['status'])
print('viability:', r.get('coordinate_viability_pass'))
print('advancement:', r.get('project_mapper_advancement_pass'))
print()
for panel in ('identity_disjoint','formula_disjoint'):
    methods = r['panels'][panel]['methods']
    print(f'== {panel} ==')
    for m in ('official_dreams','noise_v1','weighted_spectral_entropy',
              'ms2deepscore_2x_public','spec2vec_gnps_public'):
        d = methods.get(m, {}).get('all', {})
        if d:
            print(f"  {m:30s} R@1={d['recall@1']:.4f} R@5={d['recall@5']:.4f} MRR={d['mrr']:.4f} false_merge={d['same_formula_false_merge_rate']:.4f}")
    print('  -- near --')
    for m in ('official_dreams','noise_v1','weighted_spectral_entropy'):
        d = methods.get(m, {}).get('near_structure', {})
        if d:
            print(f"  {m:30s} R@1={d['recall@1']:.4f} false_merge={d['same_formula_false_merge_rate']:.4f}")
    print('  -- cross_instr --')
    for m in ('official_dreams','noise_v1','weighted_spectral_entropy'):
        d = methods.get(m, {}).get('cross_instrument_available', {})
        if d:
            print(f"  {m:30s} R@1={d['recall@1']:.4f} false_merge={d['same_formula_false_merge_rate']:.4f}")
    print()
