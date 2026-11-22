"""Chladni sand simulation (deterministic). Domain [0,1]^2, free-edge approx modes."""
import numpy as np
from scipy.ndimage import gaussian_filter

def W(x, y, n, m, s):
    return np.cos(n*np.pi*x)*np.cos(m*np.pi*y) + s*np.cos(m*np.pi*x)*np.cos(n*np.pi*y)

def gradW2(x, y, n, m, s):
    a = np.cos(n*np.pi*x)*np.cos(m*np.pi*y); b = np.cos(m*np.pi*x)*np.cos(n*np.pi*y)
    w = a + s*b
    wx = -n*np.pi*np.sin(n*np.pi*x)*np.cos(m*np.pi*y) - s*m*np.pi*np.sin(m*np.pi*x)*np.cos(n*np.pi*y)
    wy = -m*np.pi*np.cos(n*np.pi*x)*np.sin(m*np.pi*y) - s*n*np.pi*np.cos(m*np.pi*x)*np.sin(n*np.pi*y)
    return 2*w*wx, 2*w*wy, w

class Sand:
    def __init__(self, N=60000, seed=7):
        self.rng = np.random.default_rng(seed)
        self.x = self.rng.random(N); self.y = self.rng.random(N)

    def step(self, mode, drive=1.0, sub=4, sigma=0.012, eta=0.0006):
        n, m, s = mode
        k2 = (n*n + m*m)
        for _ in range(sub):
            gx, gy, w = gradW2(self.x, self.y, n, m, s)
            amp = np.minimum(np.abs(w)/2.0, 1.0)
            j = sigma*drive*(amp**1.2 + 0.03)
            self.x += -eta*drive*gx/k2*6 + j*self.rng.standard_normal(self.x.size)
            self.y += -eta*drive*gy/k2*6 + j*self.rng.standard_normal(self.y.size)
            for a in (self.x, self.y):
                np.abs(a, out=a); a[a > 1] = 2 - a[a > 1]; np.clip(a, 0, 1, out=a)

    def scatter(self, amount):
        """kick everything (mode change)"""
        self.x += amount*self.rng.standard_normal(self.x.size)
        self.y += amount*self.rng.standard_normal(self.y.size)
        for a in (self.x, self.y):
            np.abs(a, out=a); a[a > 1] = 2 - a[a > 1]; np.clip(a, 0, 1, out=a)

PLATE = np.array([0x12, 0x14, 0x19]) / 255
SAND = np.array([0xF3, 0xE2, 0xB6]) / 255
GLOW = np.array([0xE6, 0xC3, 0x6A]) / 255

def plate_base(P):
    yy, xx = np.mgrid[0:P, 0:P] / P
    r = np.hypot(xx - .35, yy - .3)
    shade = 0.85 + 0.35*np.exp(-r*r*3)  # soft specular
    brushed = 1 + 0.025*np.sin(xx*900 + 3*np.sin(yy*40))
    return np.clip(PLATE[None, None, :]*(shade*brushed)[..., None], 0, 1)

def render(sand, P=720, base=None, alpha=1.0):
    if base is None: base = plate_base(P)
    H, _, _ = np.histogram2d(sand.y, sand.x, bins=P, range=[[0, 1], [0, 1]])
    dens = 1 - np.exp(-H*0.55)
    glow = gaussian_filter(H, 4.0); glow = 1 - np.exp(-glow*0.35)
    img = base*(1 - 0.35*glow[..., None]) + GLOW*0.35*glow[..., None]
    img = img*(1 - dens[..., None]) + SAND*dens[..., None]
    img = base*(1-alpha) + img*alpha
    return (np.clip(img, 0, 1)*255).astype(np.uint8)
