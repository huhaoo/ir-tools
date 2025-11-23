from pathlib import Path
import random
import string
import shutil
import subprocess

def rand_str(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class tool:
	def __init__(self):
		self.pwd=Path("/root/autodl-tmp/ir-tools/")
		self.valid_exts = ['.png','.jpg','.jpeg','.bmp']
	def __str__(self): raise NotImplementedError
	def __repr__(self): return str(self)
	def apply(self, ipwd, opwd):
		ipwd=Path(ipwd).resolve(); opwd=Path(opwd).resolve()
		# iterate all files in ipwd
		for f in ipwd.iterdir():
			assert f.is_file() and f.suffix.lower() in self.valid_exts, f"input file must be png image: {f}"
			self.apply_single(f, opwd/f.name)
	def apply_single(self, ipath, opath):
		ipath=Path(ipath).resolve(); opath=Path(opath).resolve()
		cpwd=ipath.parent/rand_str(16)
		cipwd=cpwd/'input'; cipwd.mkdir(parents=True, exist_ok=True)
		copwd=cpwd/'output'; copwd.mkdir(parents=True, exist_ok=True)
		shutil.copy(ipath, cipwd/ipath.name)
		self.apply(cipwd, copwd)
		shutil.copy(copwd/ipath.name, opath)
		shutil.rmtree(cpwd)

def replace_yml(ipath, opath, replacements):
	ipath=Path(ipath).resolve(); opath=Path(opath).resolve()
	with open(ipath, 'r') as f:
		content = f.read()
	for k, v in replacements.items():
		content = content.replace(k, v)
	with open(opath, 'w') as f:
		f.write(content)

class xrestormer(tool):
	def __init__(self, task):
		super().__init__()
		self.codepwd=self.pwd/'ext'/'xrestormer'
		self.modelpwd=self.pwd/'pretrain/xrestormer'
		self.configpath=self.pwd/'config/xrestormer.yml'
		self.task=task
		self.model_name={
			'denoising':'denoise_300k.pth',
			'motion_deblurring':'deblur_300k.pth',
			'sr':'sr_300k.pth',
			'deraining':'derain_155k.pth',
			'dehazing':'dehaze_300k.pth',
		}
	def str_(self):
		return f"xrestormer({self.task})"
	def __str__(self):
		return f"<xrestormer_{self.task}>".upper()
		return f"xrestormer({self.task})"
	def apply(self, ipwd, opwd):
		ipwd=Path(ipwd).resolve(); opwd=Path(opwd).resolve()
		shutil.rmtree(opwd, ignore_errors=True); opwd.mkdir(parents=True, exist_ok=True)
		replace_yml(self.configpath, opwd/'xrestormer.yml', {
			"$in": str(ipwd),
			"$out": str(opwd),
			"$model": str(self.modelpwd/self.model_name[self.task])
		})
		cmd=f"conda run -n xrestormer python {self.codepwd/'xrestormer/test.py'} -opt {opwd/'xrestormer.yml'}"
		subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		(opwd/'xrestormer.yml').unlink()

tools={
	'denoising': [xrestormer('denoising')],
	'motion_deblurring': [xrestormer('motion_deblurring')],
	'sr': [xrestormer('sr')],
	'deraining': [xrestormer('deraining')],
	'dehazing': [xrestormer('dehazing')],
}
all_tools=[t for lst in tools.values() for t in lst]

def apply(model, ipwd, opwd):
	if type(model)=='str':
		for t in all_tools:
			if str(t)==model:
				t.apply(ipwd, opwd)
				return
		raise ValueError(f"unknown model: {model}")
	elif type(model) in [list, tuple]:
		for t in model: apply(t, ipwd, opwd)
	else:
		assert isinstance(model, tool), "model must be str or tool instance"
		model.apply(ipwd, opwd)

def apply_single(model, ipath, opath):
	if type(model)=='str':
		for t in all_tools:
			if str(t)==model:
				t.apply_single(ipath, opath)
				return
		raise ValueError(f"unknown model: {model}")
	elif type(model) in [list, tuple]:
		for t in model: apply_single(t, ipath, opath)
	else:
		assert isinstance(model, tool), "model must be str or tool instance"
		model.apply_single(ipath, opath)

if __name__ == "__main__":
	xrestormer('sr').apply_single('dataset/example.png', 'dataset/output.png')