#!/bin/bash
#SBATCH --gpus=1
#SBATCH -o validate_%j.out
#SBATCH -e validate_%j.err
#SBATCH -t 01:00:00

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH=$PWD:$PYTHONPATH

echo "===== Step 1: Check data columns ====="
python -u -c "
import dreams.utils.data as du
msdata = du.MSData.load('data/MassSpecGym_MurckoHist_split.hdf5')
print('Columns:', msdata.columns())
print('Len:', len(msdata))
if len(msdata) > 0:
    for col in msdata.columns():
        try:
            val = msdata.get_values(col, 0)
            if isinstance(val, bytes): val = val.decode('utf-8')
            print(f'  {col}: {str(val)[:120]}')
        except Exception as e:
            print(f'  {col}: ERROR - {e}')
"

echo ""
echo "===== Step 2: Run validation ====="
python -u validate_rule_overlap.py --n_pairs 3000

echo "===== Job finished at $(date) ====="
