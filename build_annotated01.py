"""
build_annotated01.py — SQLite磁盘去重版（内存安全）

每张谱图: 读取→校验→SQLite查重→写入MGF
去重: SQLite UNIQUE(ik, peak_fp)，磁盘索引，内存占用恒定

用法: python build_annotated01.py
"""
import os, hashlib, json, sqlite3
from tqdm import tqdm
from rdkit import Chem

DB = 'data/annotated01_dedup.db'
MGF = 'data/annotated01.mgf'

# Init SQLite
conn = sqlite3.connect(DB)
conn.execute('CREATE TABLE IF NOT EXISTS dedup (ik TEXT, fp TEXT, UNIQUE(ik, fp))')
conn.commit()

ALL_FILES = []
for root, dirs, files in os.walk('data'):
    for f in files:
        if f.endswith(('.mgf','.msp')) and f != 'annotated01.mgf':
            ALL_FILES.append(os.path.join(root, f))

stats = {'raw':0, 'bad_smi':0, 'bad_ik':0, 'few_pk':0, 'bad_pm':0, 'dup':0, 'ok':0}
out = open(MGF, 'w', encoding='utf-8')

for fp in tqdm(ALL_FILES, desc='Files'):
    cur={}; peaks=[]; cur_ik=None
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line=line.strip()
            if not line:
                if cur and peaks: stats['raw']+=1
                else: cur={}; peaks=[]; continue

                smi=cur.get('SMILES','').strip(); ik=cur_ik
                if not smi or len(ik or '') not in (14,27): stats['bad_ik']+=1; cur={}; peaks=[]; continue
                if len(peaks)<3: stats['few_pk']+=1; cur={}; peaks=[]; continue

                pm_str=str(cur.get('PEPMASS','0') or '0').split('/')[0]
                try: pm=float(pm_str)
                except: pm=0
                if not (50<=pm<=2000): stats['bad_pm']+=1; cur={}; peaks=[]; continue

                mol=Chem.MolFromSmiles(smi)
                if mol is None: stats['bad_smi']+=1; cur={}; peaks=[]; continue

                # Dedup via SQLite
                mzs=sorted([p[0] for p in peaks])
                fp_key=hashlib.md5(f'{ik}|{",".join(f"{mz:.1f}" for mz in mzs[:20])}'.encode()).hexdigest()
                try:
                    conn.execute('INSERT INTO dedup VALUES (?,?)', (ik, fp_key))
                    conn.commit()
                except sqlite3.IntegrityError:
                    stats['dup']+=1; cur={}; peaks=[]; continue

                # Write
                ion=str(cur.get('IONMODE','')).upper()
                ion='POSITIVE' if 'P' in ion else ('NEGATIVE' if 'N' in ion else 'UNKNOWN')
                nm=cur.get('NAME','?').encode('ascii','replace').decode('ascii')[:80]
                out.write(f'BEGIN IONS\nNAME={nm}\nSMILES={smi}\nINCHIKEY={ik}\n')
                if cur.get('FORMULA',''): out.write(f'FORMULA={cur["FORMULA"]}\n')
                out.write(f'PEPMASS={pm}\nIONMODE={ion}\nMSLEVEL=2\n')
                for mz,i in peaks: out.write(f'{mz:.4f} {i:.4f}\n')
                out.write('END IONS\n\n')
                stats['ok']+=1
                cur={}; peaks=[]; continue

            if '=' in line and line[0].isalpha():
                k,v=line.split('=',1); v=v.strip()
                if k in ('SMILES','smiles'): cur['SMILES']=v
                elif k in ('INCHIKEY','InChIKey','INCHIAUX'): cur_ik=v[:27]
                elif k in ('FORMULA','Formula'): cur['FORMULA']=v
                elif k in ('PEPMASS','PRECURSOR_MZ','PrecursorMZ'): cur['PEPMASS']=v
                elif k in ('IONMODE','Ion_mode','ION_MODE'): cur['IONMODE']=v
                elif k in ('NAME','Name'): cur['NAME']=v
            elif line and (line[0].isdigit() or line[0]=='-'):
                p=line.split()
                if len(p)>=2:
                    try: mz,i=float(p[0]),float(p[1])
                    except: continue
                    if mz>0 and i>0: peaks.append((mz,i))
    # Last in file
    if cur and peaks and cur.get('SMILES','') and cur_ik and len(cur_ik) in (14,27) and len(peaks)>=3:
        smi=cur['SMILES'].strip(); ik=cur_ik
        pm_str=str(cur.get('PEPMASS','0') or '0').split('/')[0]
        try: pm=float(pm_str)
        except: pm=0
        if 50<=pm<=2000 and Chem.MolFromSmiles(smi):
            mzs=sorted([p[0] for p in peaks])
            fp_key=hashlib.md5(f'{ik}|{",".join(f"{mz:.1f}" for mz in mzs[:20])}'.encode()).hexdigest()
            try:
                conn.execute('INSERT INTO dedup VALUES (?,?)', (ik, fp_key))
                conn.commit()
                out.write(f'BEGIN IONS\nSMILES={smi}\nINCHIKEY={ik}\nPEPMASS={pm}\nMSLEVEL=2\n')
                for mz,i in peaks: out.write(f'{mz:.4f} {i:.4f}\n')
                out.write('END IONS\n\n')
                stats['ok']+=1
            except sqlite3.IntegrityError: stats['dup']+=1

out.close(); conn.close()
os.remove(DB)

size=os.path.getsize(MGF)/1e6
print(f'\nRaw={stats["raw"]}  OK={stats["ok"]}  Dup={stats["dup"]}')
print(f'Bad: smi={stats["bad_smi"]} ik={stats["bad_ik"]} pk={stats["few_pk"]} pm={stats["bad_pm"]}')
print(f'annotated01.mgf: {stats["ok"]} spectra, {size:.0f}MB')
json.dump(stats, open('data/annotated01_stats.json','w'))
