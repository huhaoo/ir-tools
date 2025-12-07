from pathlib import Path
import random
import string
import shutil
import subprocess

def rand_str(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class tool:
	def __init__(self):
		self.pwd=Path("/root/shared-nvme/irtools/")
		self.valid_exts = ['.png','.jpg','.jpeg','.bmp']
		self.codepwd=self.pwd
	def __str__(self): raise NotImplementedError
	def __repr__(self): return str(self)
	def cmd_vars(self, cuda_id="0"): return f"CUDA_VISIBLE_DEVICES={cuda_id} PYTHONPATH={self.codepwd}:$PYTHONPATH"
	def run_cmd(self, cmd):
		try:
			result=subprocess.run(cmd, shell=True, check=True, cwd=self.codepwd, text=True, capture_output=True)
		except subprocess.CalledProcessError as e:
			print(f"Command failed with exit code {e.returncode}")
			print(f"stdout: {e.stdout}")
			print(f"stderr: {e.stderr}")
			raise e
	def apply(self, ipwd, opwd, cuda_id="0"):
		ipwd=Path(ipwd).resolve(); opwd=Path(opwd).resolve()
		# iterate all files in ipwd
		for f in ipwd.iterdir():
			assert f.is_file() and f.suffix.lower() in self.valid_exts, f"input file must be png image: {f}"
			self.apply_single(f, opwd/f.name, cuda_id)
	def apply_single(self, ipath, opath, cuda_id="0"):
		ipath=Path(ipath).resolve(); opath=Path(opath).resolve()
		cpwd=ipath.parent/rand_str(16)
		cipwd=cpwd/'input'; cipwd.mkdir(parents=True, exist_ok=True)
		copwd=cpwd/'output'; copwd.mkdir(parents=True, exist_ok=True)
		shutil.copy(ipath, cipwd/ipath.name)
		self.apply(cipwd, copwd, cuda_id)
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

class mprnet(tool):
	def __init__(self, task):
		super().__init__()
		self.codepwd=self.pwd/'ext'/'mprnet'
		self.pyp="/root/.conda/envs/mprnet/bin/python"
		self.task={
			'denoising':'Denoising',
			'motion_deblurring':'Deblurring',
			'deraining':'Deraining',
		}[task]
	def __str__(self): return f"<mprnet_{self.task}>".upper()
	def apply(self, ipwd, opwd, cuda_id="0"):
		ipwd=Path(ipwd).resolve(); opwd=Path(opwd).resolve()
		shutil.rmtree(opwd, ignore_errors=True); opwd.mkdir(parents=True, exist_ok=True)
		cmd=f"{self.cmd_vars(cuda_id)} {self.pyp} {self.codepwd/'demo.py'} --input_dir {ipwd} --result_dir {opwd} --task {self.task}"
		self.run_cmd(cmd)

class xrestormer(tool):
	def __init__(self, task):
		super().__init__()
		self.codepwd=self.pwd/'ext'/'xrestormer'
		self.modelpwd=self.pwd/'pretrain/xrestormer'
		self.configpath=self.pwd/'config/xrestormer.yml'
		self.pyp="/root/.conda/envs/xrestormer/bin/python"
		self.task=task
		self.model_name={
			'denoising':'denoise_300k.pth',
			'motion_deblurring':'deblur_300k.pth',
			'sr':'sr_300k.pth',
			'deraining':'derain_155k.pth',
			'dehazing':'dehaze_300k.pth',
		}
	def str_(self): return f"xrestormer({self.task})"
	def __str__(self): return f"<xrestormer_{self.task}>".upper()
	def apply(self, ipwd, opwd, cuda_id="0"):
		ipwd=Path(ipwd).resolve(); opwd=Path(opwd).resolve()
		shutil.rmtree(opwd, ignore_errors=True); opwd.mkdir(parents=True, exist_ok=True)
		replace_yml(self.configpath, opwd/'xrestormer.yml', {
			"$in": str(ipwd),
			"$out": str(opwd),
			"$model": str(self.modelpwd/self.model_name[self.task])
		})
		cmd=f"{self.cmd_vars(cuda_id)} {self.pyp} {self.codepwd/'xrestormer/test.py'} -opt {opwd/'xrestormer.yml'}"
		self.run_cmd(cmd)
		(opwd/'xrestormer.yml').unlink()

tools={
	'denoising': [xrestormer('denoising'), mprnet('denoising')],
	'motion_deblurring': [xrestormer('motion_deblurring'), mprnet('motion_deblurring')],
	'sr': [xrestormer('sr')],
	'deraining': [xrestormer('deraining'), mprnet('deraining')],
	'dehazing': [xrestormer('dehazing')],
}
all_tools=[t for lst in tools.values() for t in lst]

def apply(model, ipwd, opwd, cuda_id="0"):
	if type(model)=='str':
		for t in all_tools:
			if str(t)==model:
				t.apply(ipwd, opwd, cuda_id)
				return
		raise ValueError(f"unknown model: {model}")
	elif type(model) in [list, tuple]:
		for t in model: apply(t, ipwd, opwd, cuda_id)
	else:
		assert isinstance(model, tool), "model must be str or tool instance"
		model.apply(ipwd, opwd, cuda_id)

def apply_single(model, ipath, opath, cuda_id="0"):
	if type(model)==str:
		for t in all_tools:
			if str(t)==model:
				t.apply_single(ipath, opath, cuda_id)
				return
		raise ValueError(f"unknown model: {model}")
	elif type(model) in [list, tuple]:
		for t in model: apply_single(t, ipath, opath, cuda_id)
	else:
		assert isinstance(model, tool), "model must be str or tool instance"
		model.apply_single(ipath, opath, cuda_id)

if __name__ == "__main__":
	xrestormer('denoising').apply('dataset','tmp','0')
	mprnet('denoising').apply('dataset','tmp','1')
	shutil.rmtree('tmp')