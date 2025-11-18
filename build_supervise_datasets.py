import tools
import shutil
from pathlib import Path
import json
import cv2
from eval import combined_metric
from tqdm import tqdm
import random

basepwd=Path("/root/autodl-tmp").resolve()

imetapwd=basepwd/"datasets/metadata"
superpwd=basepwd/"datasets/supervise"
opwd=superpwd/"data"
ometapwd=superpwd/"metadata"

tmppwd=basepwd/"datasets/tmp"
tmpipwd=basepwd/"datasets/tmp/inputs"
tmpopwd=basepwd/"datasets/tmp/outputs"

opwd.mkdir(parents=True, exist_ok=True); ometapwd.mkdir(parents=True, exist_ok=True)

imetadata=list(imetapwd.glob("*.json"))

# dataset_size=4
dataset_size=2**13
metadatas=[]

def random_choise_exclude(ex):
	x=None
	while x is None or x==ex:
		x=random.choice(tools.all_tools).__str__()
	return x

mx_iters=0
for i in range(dataset_size):
	with open(imetadata[i%len(imetadata)],"r") as f:
		metadata=json.load(f)
	rd=random.random()
	if rd<0.1: ty='done'
	elif rd<0.8: ty='apply'
	else: ty='cancel'
	metadata['type']=ty
	if ty=='done':
		metadata['now']=metadata['best_path']
		metadata['history']=metadata['best_pipeline']
		metadata['response']="done"
	elif ty=='apply':
		metadata['now']=str(opwd/f"{i}.png")
		shutil.copy(metadata['lq_path'], metadata['now'])
		id=random.randint(0, len(metadata['best_pipeline'])-1)
		metadata['history']=metadata['best_pipeline'][:id]
		metadata['response']=metadata['best_pipeline'][id]
		if random.random()<0.2: metadata['last_cancel']=random_choise_exclude(metadata['response'])
		mx_iters=max(mx_iters, len(metadata["history"]))
	else:
		metadata['now']=str(opwd/f"{i}.png")
		shutil.copy(metadata['lq_path'], metadata['now'])
		id=random.randint(0, len(metadata['best_pipeline'])-1)
		metadata['history']=metadata['best_pipeline'][:id]+[random_choise_exclude(metadata['best_pipeline'][id])]
		metadata['response']="cancel"
		mx_iters=max(mx_iters, len(metadata["history"]))
	metadatas.append(metadata)
	# print(metadata)

for i in range(mx_iters):
	for j in tools.all_tools:
		deals=[]
		for k in range(dataset_size):
			if metadatas[k]['type']!='done' and len(metadatas[k]['history'])>i and str(j)==metadatas[k]['history'][i]:
				deals.append(k)
		if not len(deals): continue
		tmpipwd.mkdir(parents=True, exist_ok=True)
		for k in deals: shutil.copy(metadatas[k]['now'], tmpipwd/f"{k}.png")
		j.apply(tmpipwd, tmpopwd)
		for k in deals: shutil.copy(tmpopwd/f"{k}.png", metadatas[k]['now'])
		shutil.rmtree(tmpipwd, ignore_errors=True); shutil.rmtree(tmpopwd, ignore_errors=True)
shutil.rmtree(tmppwd, ignore_errors=True)
for i in range(dataset_size):
	with open(ometapwd/f"{i}.json", 'w') as f:
		json.dump(metadatas[i], f, indent=4)