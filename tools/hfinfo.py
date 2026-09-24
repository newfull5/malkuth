#!/usr/bin/env python3
"""Usage: hfinfo.py <hf_dataset_id> [...]  -> prints license, gated, splits, label names from HF API"""
import sys, json, urllib.request
def info(ds):
    try:
        req=urllib.request.Request(f"https://huggingface.co/api/datasets/{ds}", headers={"User-Agent":"curl/8"})
        d=json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:
        print(f"== {ds}: ERROR {e}"); return
    cd=d.get('cardData') or {}
    print(f"== {ds}")
    print("  url: https://huggingface.co/datasets/"+d.get('id',ds))
    print("  license(cardData):", cd.get('license'), "| license tags:", [t for t in d.get('tags',[]) if t.startswith('license:')])
    print("  gated:", d.get('gated'), "| private:", d.get('private'), "| downloads:", d.get('downloads'), "| lastModified:", d.get('lastModified'))
    print("  lang tags:", [t for t in d.get('tags',[]) if t.startswith('language:')][:15])
    di=cd.get('dataset_info')
    if isinstance(di,dict): di=[di]
    for c in (di or [])[:12]:
        if not isinstance(c,dict): continue
        print("  config:", c.get('config_name'), "splits:", [(s.get('name'), s.get('num_examples')) for s in c.get('splits',[])])
        for f in c.get('features',[]):
            if isinstance(f,dict) and 'dtype' in f and isinstance(f['dtype'],dict) and 'class_label' in f['dtype']:
                names=f['dtype']['class_label'].get('names')
                if isinstance(names,dict): names=list(names.values())
                print(f"    label feature '{f.get('name')}': {len(names)} classes -> {names[:80]}")
            elif isinstance(f,dict) and isinstance(f.get('sequence'),dict) and 'class_label' in f['sequence'].get('dtype',{}) if isinstance(f.get('sequence'),dict) else False:
                names=f['sequence']['dtype']['class_label'].get('names')
                if isinstance(names,dict): names=list(names.values())
                print(f"    multilabel feature '{f.get('name')}': {len(names)} classes -> {names[:80]}")
for ds in sys.argv[1:]: info(ds)
