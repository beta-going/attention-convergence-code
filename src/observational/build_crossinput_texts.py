#!/usr/bin/env python3
"""Cross-input text pool: 20 ZH + 20 EN, 3 length bands (7/7/6).
Bands target ~128-192 / 256-320 / 384-512 tokens; per-model actual token
counts are recorded in crossinput_run.py. Offline fallback: --zh-pool/--en-pool
plaintext files (one paragraph block per blank-line-separated chunk)."""
import argparse, json, random, re
# import os
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["HF_DATASETS_OFFLINE"] = "1"

BANDS = {"zh": [("short",300,450),("mid",500,700),("long",750,1000)],
         "en": [("short",150,250),("mid",250,400),("long",400,550)]}
PER_BAND = [7,7,6]

def zh_len(t): return len(re.sub(r"\s","",t))
def en_len(t): return len(t.split())

def pool_plaintext(path):
    txt = open(path,encoding="utf-8").read()
    return [p.strip() for p in re.split(r"\n\s*\n",txt) if len(p)>100]

def pool_wikitext_en():                      # streaming, no big download
    from datasets import load_dataset
    from datasets import load_from_disk

    # 直接加载本地保存的 Arrow 格式数据集
    ds = load_dataset("parquet",data_files=os.environ.get("AE_WIKITEXT_GLOB", "data/external/wikitext-103-raw-v1/*.parquet"),split="train")
    print(ds)
    out,buf = [],[]
    for r in ds:
        line = r["text"].strip()
        if not line or line.startswith("="): continue
        buf.append(line)
        if line.endswith((".","!","?")) and sum(map(len,buf))>600:
            out.append(" ".join(buf)); buf=[]
        if len(out)>=4000: break
    return out

def pool_wiki_zh():
    from datasets import load_dataset,load_from_disk
    #ds = load_from_disk("<ZH_WIKI_DIR>")
    ds = load_dataset(
        "parquet",
        data_files=os.environ.get("AE_WIKI_ZH_GLOB", "data/external/wikipedia-20231101-zh/*.parquet"),
        split="train",
    )
    out=[]
    for r in ds:
        for p in r["text"].split("\n"):
            if len(p.strip())>=300: out.append(p.strip())
        if len(out)>=4000: break
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh-pool"); ap.add_argument("--en-pool")
    ap.add_argument("--out",default="crossinput_texts.json")
    ap.add_argument("--seed",type=int,default=42)
    a = ap.parse_args()
    rng, texts = random.Random(a.seed), {}
    for lang in ("zh","en"):
        length = zh_len if lang=="zh" else en_len
        pool = (pool_plaintext(a.zh_pool if lang=="zh" else a.en_pool)
                if (a.zh_pool and lang=="zh") or (a.en_pool and lang=="en")
                else (pool_wiki_zh() if lang=="zh" else pool_wikitext_en()))
        pool = list(dict.fromkeys(pool))               # dedup, keep order
        sel, idx = [], 0
        for (name,lo,hi),n in zip(BANDS[lang],PER_BAND):
            cand = [p for p in pool if lo<=length(p)<hi]
            rng.shuffle(cand)
            sel += [{"band":name,"idx":idx+i,"text":p} for i,p in enumerate(cand[:n])]
            idx += n
        assert all(len(s["text"])>0 for s in sel) and len(sel)==20, lang
        texts[lang] = sel
        print(f"{lang}: {len(sel)} texts")
    import os as _os
    _d = _os.path.dirname(a.out)
    if _d: _os.makedirs(_d, exist_ok=True)
    json.dump(texts,open(a.out,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
    print(f"wrote {a.out}")

if __name__=="__main__": main()
