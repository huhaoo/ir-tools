from add_degradation import *
from pathlib import Path
import json
import random
import cv2
from tqdm import tqdm

degenerations=["denoising","motion_deblurring","sr","deraining","dehazing"]

imgs=[f"{i:03}" for i in range(1, 101)]
basepwd=Path("/root/autodl-tmp").resolve()
gtpwd=basepwd/"mio100/gt"
depthpwd=basepwd/"mio100/depth"

lqpwd=basepwd/"datasets/lq"
metapwd=basepwd/"datasets/metadata"
lqpwd.mkdir(parents=True, exist_ok=True); metapwd.mkdir(parents=True, exist_ok=True)

dataset_size=10
max_degs=3
for i in tqdm(range(dataset_size)):
	id=random.choice(imgs)
	random.shuffle(degenerations)
	deg=degenerations[:random.randint(1,max_degs)]
	img=cv2.imread(gtpwd/f"{id}.png")
	for d in deg:
		if d=='denoising': img=add_noise(img)
		elif d=='motion_deblurring': img=add_motion_blur(img)
		elif d=='sr': img=lr(img,True)
		elif d=='deraining': img=add_rain(img)
		elif d=='dehazing': img=add_haze(img, depthpwd/id)
	cv2.imwrite(lqpwd/f"{i}.png", img)
	metadata={
		'source_id': id,
		'degradations': deg,
		'depth_pwd': depthpwd/id,
		'gt_path': gtpwd/f"{id}.png",
		'lq_path': lqpwd/f"{i}.png",
	}
	metadata={k: str(v) if isinstance(v, Path) else v for k,v in metadata.items()}
	with open(metapwd/f"{i}.json", 'w') as f:
		json.dump(metadata, f, indent=4)