"""Charged particle in a dipole field (Boris pusher). Units: Earth radius = 1.
B = M (3 (m.rhat) rhat - m) / r^3, m = -z (Earth's dipole points south)."""
import numpy as np

def Bfield(r, M):
    x, y, z = r
    R = np.sqrt(x*x + y*y + z*z)
    m = np.array([0, 0, -1.0])
    rh = r / R
    return M * (3 * np.dot(m, rh) * rh - m) / R**3

def simulate(L=2.6, pitch_deg=38, qm_M=420.0, v=1.0, dt=4e-4, steps=200000, every=20):
    r = np.array([L, 0, 0.0])
    a = np.radians(pitch_deg)
    B0 = Bfield(r, 1.0); bh = B0/np.linalg.norm(B0)
    e1 = np.array([0, 1.0, 0])
    vel = v*(np.cos(a)*bh + np.sin(a)*e1)
    out = []; vpar = []
    for i in range(steps):
        B = Bfield(r, qm_M)
        t = B*dt/2; s = 2*t/(1+np.dot(t,t))
        vp = vel + np.cross(vel, t); vel = vel + np.cross(vp, s)
        r = r + vel*dt
        if i % every == 0:
            out.append(r.copy()); bb = B/np.linalg.norm(B); vpar.append(np.dot(vel, bb))
    return np.array(out), np.array(vpar), dt*every

if __name__ == "__main__":
    import time; t0=time.time()
    P, vp, h = simulate()
    print("pts", len(P), "sim-time", len(P)*h, "secs", time.time()-t0)
    # bounce: sign changes of vpar
    sc = np.where(np.diff(np.sign(vp)) != 0)[0]
    print("mirror crossings at sim t:", (sc*h)[:8])
    phi = np.unwrap(np.arctan2(P[:,1], P[:,0]))
    print("drift angle total (deg):", np.degrees(phi[-1]-phi[0]))
    print("z range", P[:,2].min(), P[:,2].max(), "rmin", np.linalg.norm(P,axis=1).min())
    np.save("traj.npy", P); np.save("vpar.npy", vp)
