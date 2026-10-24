"""Local stdlib verification of a returned completed method run."""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from task5_data import sha256, metrics, spearman


def same(a,b):
    return a is None and b is None or a is not None and b is not None and abs(a-b)<1e-12


def verify(folder):
    cfg=json.loads((folder/'config.json').read_text())
    done=json.loads((folder/'completion.json').read_text())
    assert done['status']=='COMPLETED'
    assert json.loads((folder/'INDEPENDENT_VERIFIED.json').read_text())['status']=='PASS'
    rows=[json.loads(x) for x in (folder/'val_predictions.jsonl').read_text().splitlines()]
    gold=[json.loads(x) for x in (ROOT/'data/en_nn_seed2027/val.jsonl').read_text().splitlines()]
    if cfg['smoke']:gold=gold[:32]
    assert len(rows)==len({r['ContextID'] for r in rows})==len(gold)
    assert [r['ContextID'] for r in rows]==[r['ContextID'] for r in gold]
    assert [r['gold'] for r in rows]==[[float(r['ModAvg']),float(r['HeadAvg'])] for r in gold]
    assert sha256(folder/'val_predictions.jsonl')==done['prediction_sha256']
    for filename,expected in done['checkpoint_sha256'].items():assert sha256(folder/'best'/filename)==expected,filename
    for filename,expected in cfg['source_sha256'].items():assert sha256(ROOT/'src'/filename)==expected,filename
    for name in ('train','val'):assert sha256(ROOT/'data/en_nn_seed2027'/(name+'.jsonl'))==cfg[name+'_sha256']
    scored=json.loads((folder/'metrics_std_round12.json').read_text())
    groups=defaultdict(list)
    for i,row in enumerate(gold):groups[row['Compound']].append(i)
    for form in ('raw','clipped'):
        local=metrics(gold,[r[form] for r in rows])
        for metric in ('mean_spearman','mean_rmse'):assert same(local[metric],scored[form][metric])
        for col,label in enumerate(('ModAvg','HeadAvg')):
            for metric in ('spearman','rmse'):assert same(local[label][metric],scored[form][label][metric])
            gold_sd=[round(statistics.pstdev(float(gold[i][label]) for i in ids),12) for ids in groups.values()]
            pred_sd=[round(statistics.pstdev(rows[i][form][col] for i in ids),12) for ids in groups.values()]
            assert same(spearman(gold_sd,pred_sd),scored[form][label]['context_variance_spearman'])
    export=[x.split('\t') for x in (folder/'en-nn-internal-val-pred.tsv').read_text().splitlines()]
    assert [x[0] for x in export]==[x['ContextID'] for x in gold]
    assert all(abs(float(x[i+1])-r['clipped'][i])<=1e-8 for x,r in zip(export,rows) for i in range(2))
    if 'validation_trace_sha256' in done:assert sha256(folder/'best_validation_inputs.jsonl')==done['validation_trace_sha256']
    assert done['reload_max_abs_diff']<1e-6 and done['repeat_max_abs_diff']<1e-6
    proof={'status':'PASS','rows':len(rows),'source_data_checkpoint_prediction_hashes':True,
           'primary_and_canonical_secondary_recomputed':True,'id_gold_export_verified':True,
           'actual_reload_diff':done['reload_max_abs_diff'],'actual_repeat_diff':done['repeat_max_abs_diff'],
           'reload_scope':done.get('reload_scope','deterministic fixed inputs')}
    (folder/'LOCAL_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'run':folder.name,'local_verification':proof,'metrics':scored['clipped']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path)
    verify(parser.parse_args().run)
