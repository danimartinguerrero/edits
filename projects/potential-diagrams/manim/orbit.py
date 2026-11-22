"""Kepler orbit with a slowly varied energy (units mu = k = L = 1, p = 1).
r(phi) = 1/(1 + eps cos phi), eps = sqrt(1 + 2E), dphi/dt = 1/r^2."""
import numpy as np

KT = 6.0            # sim time per video second
FPS = 30


def eps_of(E):
    return np.sqrt(max(0.0, 1 + 2 * E))


def track(t0, t1, E_sched, sub=40):
    """E_sched(t, phi, state) -> E ; returns arrays t, phi, r, E"""
    ts, ph, rs, Es = [], [], [], []
    phi, E = 0.0, E_sched(t0, 0.0, None)
    dt = 1.0 / FPS / sub
    t = t0
    state = {}
    while t <= t1 + 1e-9:
        E = E_sched(t, phi, state)
        e = eps_of(E)
        den = 1 + e * np.cos(phi)
        r = 1 / den if den > 1e-3 else 1e3
        ts.append(t); ph.append(phi); rs.append(r); Es.append(E)
        for _ in range(sub):
            den = 1 + e * np.cos(phi)
            rr = 1 / den if den > 1e-3 else 1e3
            phi += KT * dt / rr ** 2
        t += 1.0 / FPS
    return np.array(ts), np.array(ph), np.array(rs), np.array(Es)


def sched_A(t, phi, st):
    def ramp(a, b, x0, x1):
        u = np.clip((t - a) / (b - a), 0, 1); u = u * u * (3 - 2 * u)
        return x0 + (x1 - x0) * u
    if t < 3.5:
        return -0.32
    if t < 5.8:
        return ramp(3.5, 4.8, -0.32, -0.5)
    if t < 7.2:
        return ramp(5.8, 6.8, -0.5, -0.25)
    # escape: start ramp at the first perihelion passage after t = 9
    if "tp" not in st:
        k = np.floor((phi + 0.35) / (2 * np.pi))     # crossing just before perihelion
        if "k" in st and k > st["k"]:
            st["tp"] = t
        else:
            st["k"] = k
            return -0.25
    return ramp(st["tp"] - 0.05, st["tp"] + 0.25, -0.25, 0.16)


if __name__ == "__main__":
    tA = track(0.0, 12.0, sched_A)
    tB = track(12.0, 30.0, lambda t, p, s: -0.32)
    np.savez("orbit.npz", tA=np.stack(tA), tB=np.stack(tB))
    i = np.where(tA[3] > 0)[0]
    print("escape starts at video t =", tA[0][i[0]] if len(i) else None, "max r", tA[2].max())
