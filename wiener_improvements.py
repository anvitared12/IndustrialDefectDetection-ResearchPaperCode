import glob, os
import numpy as np
import cv2
from PIL import Image
from skimage.draw import disk
from skimage.restoration import wiener


def make_disk_psf(radius: int, alias_blur: float = 0.5) -> np.ndarray:
    r = int(radius)
    size = 2 * (r + 3) + 1
    c = size // 2
    psf = np.zeros((size, size), np.float32)
    rr, cc = disk((c, c), r + 0.5, shape=psf.shape)
    psf[rr, cc] = 1.0
    psf /= psf.sum()
    k = 3 if r <= 8 else 5
    psf = cv2.GaussianBlur(psf, (k, k), alias_blur)
    return psf / psf.sum()


def wiener_deblur(image: Image.Image, radius: int = 3, balance: float = 0.1,
                  blend: float = 1.0, alias_blur: float = 0.5) -> Image.Image:

    rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32) / 255.0

    psf = make_disk_psf(radius, alias_blur)
    pad = psf.shape[0]
    Lp = np.pad(L, pad, mode="reflect")
    rest = wiener(Lp, psf, balance, clip=False)[pad:-pad, pad:-pad]
    rest = np.clip(rest, 0.0, 1.0)

    out = blend * rest + (1.0 - blend) * L
    lab[..., 0] = np.uint8(np.clip(out * 255.0 + 0.5, 0, 255))
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))



def sharpness(image: Image.Image, size: int = 256) -> float:
    g = np.array(image.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float64)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def gradient_sparsity(image: Image.Image, size: int = 256) -> float:
    g = np.array(image.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float64) / 255.0
    v = np.r_[np.diff(g, axis=1).ravel(), np.diff(g, axis=0).ravel()]
    return float(np.abs(v).sum() / np.sqrt((v ** 2).sum() + 1e-12))


def reference_sharpness(good_paths, n=40, size=256) -> float:
    vals = [sharpness(Image.open(p), size) for p in good_paths[:n]]
    return float(np.median(vals))


class DomainDeblur:

    def __init__(self, ref_sharp, blur_ratio=0.5,
                 radii=(2, 3, 4, 6, 8, 10), balances=(0.03, 0.1, 0.3, 1.0), blend=0.7):
        self.ref, self.blur_ratio = ref_sharp, blur_ratio
        self.radii, self.balances, self.blend = radii, balances, blend
        self.active, self.radius, self.balance = False, None, None

    def fit(self, unlabeled_images, max_images=20):
        imgs = unlabeled_images[:max_images]
        raw = np.median([sharpness(i) for i in imgs]) / self.ref
        self.active = raw < self.blur_ratio
        self.raw_ratio = raw
        if not self.active:
            return self
        best = (np.inf, None, None)
        for r in self.radii:
            for b in self.balances:
                score = np.median([gradient_sparsity(wiener_deblur(i, r, b, self.blend)) for i in imgs])
                if score < best[0]:
                    best = (score, r, b)


        base = np.median([gradient_sparsity(i) for i in imgs])
        if best[0] >= base:
            self.active = False
            return self
        _, self.radius, self.balance = best
        return self

    def __call__(self, image):
        if not self.active:
            return image
        return wiener_deblur(image, self.radius, self.balance, self.blend)


def threshold_at_fpr(normal_scores, fpr):
    return float(np.quantile(np.asarray(normal_scores), 1.0 - fpr))


def recall_at_threshold(scores, labels, thr):
    pos = np.asarray(scores)[np.asarray(labels) == 1]
    return float((pos >= thr).mean())


def recall_with_ci(test_scores, test_labels, calib_normal_scores, fpr=0.075, n_boot=1000, seed=0):

    rng = np.random.default_rng(seed)
    test_scores, test_labels = np.asarray(test_scores), np.asarray(test_labels)
    calib = np.asarray(calib_normal_scores)
    pos = test_scores[test_labels == 1]

    point = float((pos >= threshold_at_fpr(calib, fpr)).mean())
    boots = []
    for _ in range(n_boot):
        c = rng.choice(calib, len(calib), replace=True)
        p = rng.choice(pos, len(pos), replace=True)
        boots.append((p >= threshold_at_fpr(c, fpr)).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, (float(lo), float(hi))


def recall_curve(test_scores, test_labels, calib_normal_scores, fprs=(0.01, 0.025, 0.05, 0.075, 0.1, 0.2)):
    return {f: recall_at_threshold(test_scores, test_labels, threshold_at_fpr(calib_normal_scores, f))
            for f in fprs}