import tools
import shutil
from pathlib import Path
import json
import cv2
from eval import combined_metric
from tqdm import tqdm

basepwd=Path("/root/autodl-tmp").resolve()

lqpwd=basepwd/"datasets/lq"
metapwd=basepwd/"datasets/metadata"

opwd=basepwd/"datasets/best"
tmppwd=basepwd/"datasets/tmp"
tmpipwd=basepwd/"datasets/tmp/inputs"
tmpopwd=basepwd/"datasets/tmp/outputs"

shutil.rmtree(opwd, ignore_errors=True); opwd.mkdir(parents=True, exist_ok=True)

imgs=sorted((lqpwd).glob("*.png"))
imgs=[p.name[:-4] for p in imgs]

def find_best_ir(imgs):
    shutil.rmtree(tmppwd, ignore_errors=True); tmppwd.mkdir(parents=True, exist_ok=True)
    enum=[]
    results=[]
    metadatas=[]
    mx_iters=0
    for i in imgs:
        results.append([])
        with open(metapwd/f"{i}.json", 'r') as f:
            metadatas.append(json.load(f))
        def search_perm(a,t):
            if not len(a):
                id=len(enum)
                enum.append(t)
                results[-1].append(id)
                shutil.copy(lqpwd/f"{i}.png", tmppwd/f"{id}.png")
            for x in range(len(a)):
                for y in tools.tools[a[x]]:
                    search_perm(a[:x]+a[x+1:], t+[y])
        search_perm(metadatas[-1]["degradations"],[])
        mx_iters=max(mx_iters, len(metadatas[-1]["degradations"]))
    # print(f"Total {len(enum)} pipelines to apply, max steps: {mx_iters}")
    for i in range(mx_iters):
        for j in tools.all_tools:
            deals=[]
            for k in range(len(enum)):
                if len(enum[k])>i and (j is enum[k][i]):
                    deals.append(k)
            if not len(deals): continue
            tmpipwd.mkdir(parents=True, exist_ok=True)
            for k in deals: shutil.copy(tmppwd/f"{k}.png", tmpipwd/f"{k}.png")
            j.apply(tmpipwd, tmpopwd)
            for k in deals: shutil.copy(tmpopwd/f"{k}.png", tmppwd/f"{k}.png")
            shutil.rmtree(tmpipwd, ignore_errors=True); shutil.rmtree(tmpopwd, ignore_errors=True)
    # print("Evaluating results...")
    for i in range(len(imgs)):
        best_id=-1
        best_score=(-float('inf'), {})
        gt=cv2.imread(metadatas[i]['gt_path'])
        for j in results[i]:
            ir=cv2.imread(tmppwd/f"{j}.png")
            score=combined_metric(gt, ir)
            if score[0]>best_score[0]:
                best_score=score
                best_id=j
        metadatas[i]['best_path']=str(opwd/f"{imgs[i]}.png")
        shutil.copy(tmppwd/f"{best_id}.png", metadatas[i]['best_path'])
        metadatas[i]['best_pipeline']=[str(t) for t in enum[best_id]]
        metadatas[i]['best_score']={'combined': best_score[0], **best_score[1]}
        with open(metapwd/f"{imgs[i]}.json", 'w') as f:
            json.dump(metadatas[i], f, indent=4)
    shutil.rmtree(tmppwd, ignore_errors=True)

print("Available IR tools:")
print(tools.tools)
n=len(imgs)
bs=128
bn=(n-1)//bs+1
for bi in tqdm(range(bn)):
    find_best_ir(imgs[bi*bs:(bi+1)*bs])