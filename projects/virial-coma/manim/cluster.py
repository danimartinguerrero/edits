"""Plummer-sphere N-body cluster in virial equilibrium (G = M = a = 1), leapfrog."""
import numpy as np

def plummer(N, seed=5):
    rng = np.random.default_rng(seed)
    X1 = rng.random(N)
    r = 1.0 / np.sqrt(X1 ** (-2 / 3) - 1)
    r = np.minimum(r, 6.0)
    def iso(n, rad):
        u = rng.uniform(-1, 1, n); ph = rng.uniform(0, 2*np.pi, n); s = np.sqrt(1-u*u)
        return np.c_[rad*s*np.cos(ph), rad*s*np.sin(ph), rad*u]
    pos = iso(N, r)
    q = np.zeros(N)
    for i in range(N):            # von Neumann rejection for q
        while True:
            a, b = rng.random(), rng.random() * 0.1
            if b < a*a*(1-a*a)**3.5:
                q[i] = a; break
    ve = np.sqrt(2) * (1 + r*r) ** -0.25
    vel = iso(N, q * ve)
    pos -= pos.mean(0); vel -= vel.mean(0)
    return pos, vel

def acc(pos, m, eps=0.08):
    d = pos[None, :, :] - pos[:, None, :]
    r2 = (d**2).sum(-1) + eps*eps
    inv = r2 ** -1.5
    np.fill_diagonal(inv, 0)
    return (d * (m[None, :, None] * inv[..., None])).sum(1)

def run(N=400, T=30.0, dt=0.004, save_every=5):
    pos, vel = plummer(N)
    m = np.full(N, 1.0 / N)
    a = acc(pos, m)
    P, V = [pos.copy()], [vel.copy()]
    n = int(T / dt)
    for k in range(n):
        vel += 0.5*dt*a; pos += dt*vel; a = acc(pos, m); vel += 0.5*dt*a
        if (k+1) % save_every == 0:
            P.append(pos.copy()); V.append(vel.copy())
    return np.array(P), np.array(V), dt*save_every, m

if __name__ == "__main__":
    import time; t0 = time.time()
    P, V, h, m = run()
    K = 0.5*(m[None,:]*(V**2).sum(-1)).sum(1)
    print("frames", len(P), "h", h, "time", time.time()-t0)
    print("K start/end", K[0], K[-1], "2K/|W|~", 2*K[0]/(3*np.pi/32))
    np.save("P.npy", P.astype(np.float32)); np.save("V.npy", V.astype(np.float32)); np.save("h.npy", np.array([h]))
