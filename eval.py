import cv2
import numpy as np
import add_degradation

def psnr(gt, ir):
	"""Computes the PSNR (Peak Signal-to-Noise Ratio) between two images."""
	mse = np.mean((gt - ir) ** 2)
	if mse == 0:
		return float('inf')
	PIXEL_MAX = 255.0
	return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def ssim(gt, ir):
	"""Computes the SSIM (Structural Similarity Index) between two images."""
	C1 = (0.01 * 255) ** 2
	C2 = (0.03 * 255) ** 2

	gt = gt.astype(np.float64)
	ir = ir.astype(np.float64)

	# Compute means
	mu1 = np.mean(gt)
	mu2 = np.mean(ir)

	# Compute variances
	var1 = np.var(gt)
	var2 = np.var(ir)

	# Compute covariance
	cov = np.mean((gt - mu1) * (ir - mu2))

	# Compute SSIM
	ssim = (2 * mu1 * mu2 + C1) * (2 * cov + C2) / ((mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2))
	return ssim

def combined_metric(gt, ir):
	if gt.shape!=ir.shape:
		ir=cv2.resize(ir, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
	assert gt.shape==ir.shape, "Images must have the same dimensions"

	psnr_weight = 1./32
	ssim_weight = 1.
	psnr_value = psnr(gt, ir)
	ssim_value = ssim(gt, ir)
	return psnr_weight * psnr_value + ssim_weight * ssim_value , {
		'psnr': psnr_value,
		'ssim': ssim_value
	}