#!/usr/bin/env python3
"""Fetch the public benchmark after a stalled historical filesystem read.

Official published compressed-file MD5s are verified before extraction. Raw
SHA-256s are recorded for later comparison to historical run metadata.
"""
import gzip
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"analysis/completion_20260905/data/FashionMNIST/raw"
FILES={"train-images-idx3-ubyte.gz":"8d4fb7e6c68d591d4c3dfef9ec88bf0d",
       "train-labels-idx1-ubyte.gz":"25c81989df183df01b3e8a0aad5dffbe",
       "t10k-images-idx3-ubyte.gz":"bef4ecab320f06d8554ea6380940ec79",
       "t10k-labels-idx1-ubyte.gz":"bb300cfdad3c16e7a12a480ee83cd310"}


def fetch(item):
    name,expected=item
    url="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"+name
    response=requests.get(url,timeout=60);response.raise_for_status()
    data=response.content;assert hashlib.md5(data).hexdigest()==expected
    raw=gzip.decompress(data)
    (OUT/name).write_bytes(data);(OUT/name[:-3]).write_bytes(raw)
    return {"file":name,"url":url,"official_compressed_md5":expected,"raw_sha256":hashlib.sha256(raw).hexdigest(),"raw_bytes":len(raw)}


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:record=list(pool.map(fetch,FILES.items()))
    (ROOT/"analysis/completion_20260905/fashion_cache_manifest.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))
