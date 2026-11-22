"""Coupled pendulums — physics + math layer (Manim CE 0.22).
Theory and notation follow the user's lab report (Práctica 3): I phi'' = -mgL phi
+ k l^2 (phi_2 - phi_1), omega_0^2 = mgL/I, Omega^2 = k l^2/I, q0/q1 normal
coordinates, omega_pm = (omega_1 pm omega_0)/2, T_b = pi/omega_-.
omega_0 = 3.148 rad/s is the measured value; the coupling is exaggerated
(omega_1 = 1.25 omega_0) so a beat fits in a 30 s clip.
"""
import numpy as np
from manim import *

config.pixel_width = 1080
config.pixel_height = 1920
config.frame_height = 16.0
config.frame_width = 9.0
config.frame_rate = 30
config.background_color = "#000000"

INK, MUTED, PANEL, EDGE = "#F2F2F2", "#9BA2AF", "#0E1014", "#4D5466"
BLUE, LAV, GOLD = "#6FA8FF", "#B7A6F5", "#E6C36A"

W0 = 3.148
W1 = 1.25 * W0
WM, WP = (W1 - W0) / 2, (W1 + W0) / 2
PHI0 = 0.36
RELEASE = 0.45
MATH_START, MATH_END = 12.0, 26.2
STEPS = [12.8, 14.9, 17.0, 19.1, 21.2, 23.3]

TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage{amsmath}\usepackage{amssymb}")


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ── motion: beat, with in-phase / anti-phase demos during steps 3–4 ─────────
def beat(tau):
    return (PHI0 * np.cos(WM * tau) * np.cos(WP * tau),
            PHI0 * np.sin(WM * tau) * np.sin(WP * tau))


def inphase(tau):
    a = 0.8 * PHI0 * np.cos(W0 * tau)
    return a, a


def antiphase(tau):
    a = 0.8 * PHI0 * np.cos(W1 * tau)
    return a, -a


def angles(t):
    if t < RELEASE:
        return PHI0, 0.0
    tau = t - RELEASE
    b = np.array(beat(tau))
    # demo windows (global time): in-phase during step 3, anti-phase during step 4
    w_in = smooth((t - STEPS[2]) / 0.5) * (1 - smooth((t - STEPS[3]) / 0.5))
    w_anti = smooth((t - STEPS[3]) / 0.5) * (1 - smooth((t - STEPS[4]) / 0.5))
    w_b = max(0.0, 1 - w_in - w_anti)
    v = w_b * b + w_in * np.array(inphase(t - STEPS[2])) + w_anti * np.array(antiphase(t - STEPS[3]))
    return float(v[0]), float(v[1])


BIG_S, BIG_C = 1.0, np.array([0.0, 0.2, 0])
SMALL_S, SMALL_C = 0.44, np.array([-2.25, 4.25, 0])


def rig_layout(t):
    if t < MATH_START:
        a = 0.0
    elif t < MATH_END:
        a = smooth((t - MATH_START) / 0.8)
    else:
        a = 1 - smooth((t - MATH_END) / 0.7)
    return BIG_S + (SMALL_S - BIG_S) * a, BIG_C + (SMALL_C - BIG_C) * a


# rig geometry (local units, pivot line at y = +4)
PIV = [np.array([-1.55, 4.0, 0]), np.array([1.55, 4.0, 0])]
LROD, LSPR = 4.3, 2.3


def spring_pts(a, b, coils=11, amp=0.13):
    d = b - a
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[1], u[0], 0])
    pts = [a, a + u * 0.18]
    n = coils * 2
    for i in range(1, n):
        s = 0.18 + (L - 0.36) * i / n
        pts.append(a + u * s + nrm * amp * (1 if i % 2 else -1))
    pts += [b - u * 0.18, b]
    return pts


def build_rig(p1, p2):
    g = VGroup()
    bar = Line([-2.7, 4.0, 0], [2.7, 4.0, 0], stroke_color=EDGE, stroke_width=5)
    hatch = VGroup(*[Line([x, 4.0, 0], [x + 0.22, 4.25, 0], stroke_color=EDGE, stroke_width=2)
                     for x in np.arange(-2.6, 2.6, 0.26)])
    g.add(hatch, bar)
    ends, atts = [], []
    for piv, ph in zip(PIV, (p1, p2)):
        dirv = np.array([np.sin(ph), -np.cos(ph), 0])
        ends.append(piv + dirv * LROD)
        atts.append(piv + dirv * LSPR)
    spring = VMobject(stroke_color=LAV, stroke_width=3).set_points_as_corners(spring_pts(atts[0], atts[1]))
    g.add(spring)
    for piv, end, col in zip(PIV, ends, (GOLD, BLUE)):
        g.add(Line(piv, end, stroke_color=INK, stroke_width=3.2))
        g.add(Dot(piv, radius=0.07, color=INK))
        g.add(Circle(radius=0.62, stroke_width=0, fill_color=col, fill_opacity=0.10).move_to(end))
        g.add(Circle(radius=0.44, stroke_width=0, fill_color=col, fill_opacity=0.16).move_to(end))
        g.add(Circle(radius=0.30, stroke_color=WHITE, stroke_width=1.2, stroke_opacity=0.5,
                     fill_color=col, fill_opacity=1).move_to(end))
    for a in atts:
        g.add(Dot(a, radius=0.05, color=LAV))
    return g


def eq(s, scale=1.0):
    m = MathTex(s, tex_template=TEX, color=INK).scale(scale)
    for tok, col in ((r"\varphi_1", GOLD), (r"\varphi_2", BLUE), (r"q_0", LAV), (r"q_1", LAV),
                     (r"\omega_-", LAV), (r"\omega_+", LAV), (r"\omega_1", BLUE), (r"\Omega", BLUE)):
        try:
            m.set_color_by_tex(tok, col, substring=False)
        except Exception:
            pass
    return m


class Pendulums(Scene):
    def construct(self):
        T = lambda: self.renderer.time

        def rig_f():
            t = T()
            s, c = rig_layout(t)
            g = build_rig(*angles(t))
            g.scale(s, about_point=ORIGIN).shift(c)
            return g

        rig = always_redraw(rig_f)

        # live traces (lab-software look): two lanes, 7 s window
        LANE = [-2.55, -4.35]
        X0, X1, WIN = -3.6, 3.9, 7.0
        AMP = 0.72

        def lanes_f():
            t = T()
            op = 1 - smooth((t - 11.4) / 0.5) if t < 20 else 0
            op *= smooth((t - 0.2) / 0.5)
            g = VGroup()
            if op <= 0.001:
                return g
            ts = np.linspace(max(0, t - WIN), t, 220)
            for k, (y0, col, lab) in enumerate(zip(LANE, (GOLD, BLUE), (r"\varphi_1", r"\varphi_2"))):
                g.add(Line([X0, y0, 0], [X1, y0, 0], stroke_color=EDGE, stroke_width=1.5, stroke_opacity=op))
                xs = X1 - (t - ts) / WIN * (X1 - X0)
                env = []
                vals = []
                for tt in ts:
                    a1, a2 = angles(tt)
                    vals.append((a1, a2)[k])
                    tau = max(0.0, tt - RELEASE)
                    env.append(PHI0 * (abs(np.cos(WM * tau)) if k == 0 else abs(np.sin(WM * tau)))
                               if tt >= RELEASE else (PHI0 if k == 0 else 0))
                vals, env = np.array(vals), np.array(env)
                if len(ts) > 2:
                    for sgn in (1, -1):
                        g.add(VMobject(stroke_color=LAV, stroke_width=1.6, stroke_opacity=0.45 * op)
                              .set_points_as_corners(np.c_[xs, y0 + sgn * env / PHI0 * AMP, 0 * xs]))
                    g.add(VMobject(stroke_color=col, stroke_width=3.4, stroke_opacity=op)
                          .set_points_as_corners(np.c_[xs, y0 + vals / PHI0 * AMP, 0 * xs]))
                    g.add(Dot([xs[-1], y0 + vals[-1] / PHI0 * AMP, 0], radius=0.07, color=col).set_opacity(op))
                g.add(LABS[k].copy().move_to([X0 - 0.45, y0, 0]).set_opacity(op))
            return g

        LABS = [MathTex(r"\varphi_1", color=GOLD).scale(0.9), MathTex(r"\varphi_2", color=BLUE).scale(0.9)]
        lanes = always_redraw(lanes_f)
        ticker = Mobject().add_updater(lambda m, dt: None)  # forces per-frame rendering in wait()
        self.add(ticker, lanes, rig)

        def until(t):
            dt = t - self.renderer.time
            if dt > 1e-3:
                self.wait(dt)

        until(MATH_START)

        TOP, MID = 1.55, -0.95
        steps = [
            [eq(r"I\,\ddot{ {{\varphi_1}} } = -mgL\, {{\varphi_1}} + k\ell^2\,( {{\varphi_2}} - {{\varphi_1}} )", 1.0),
             eq(r"I\,\ddot{ {{\varphi_2}} } = -mgL\, {{\varphi_2}} - k\ell^2\,( {{\varphi_2}} - {{\varphi_1}} )", 1.0)],
            [eq(r"\begin{pmatrix}\ddot\varphi_1\\ \ddot\varphi_2\end{pmatrix} = -\begin{pmatrix}\omega_0^2+\Omega^2 & -\Omega^2\\ -\Omega^2 & \omega_0^2+\Omega^2\end{pmatrix}\begin{pmatrix}\varphi_1\\ \varphi_2\end{pmatrix}", 0.95),
             eq(r"\omega_0^2=\frac{mgL}{I},\qquad {{\Omega}}^2=\frac{k\ell^2}{I}", 1.0)],
            [eq(r"{{q_0}} = {{\varphi_1}} + {{\varphi_2}},\qquad {{q_1}} = {{\varphi_1}} - {{\varphi_2}}", 1.05),
             eq(r"\ddot q_0+\omega_0^2\, {{q_0}} = 0,\qquad \ddot q_1+ {{\omega_1}}^2\, {{q_1}} = 0", 1.05)],
            [eq(r"{{\omega_1}} = \sqrt{\omega_0^2+2 {{\Omega}}^2}\; > \;\omega_0", 1.25)],
            [eq(r"{{\varphi_1}} = \varphi_0\cos {{\omega_-}} t\,\cos {{\omega_+}} t", 1.1),
             eq(r"{{\varphi_2}} = \varphi_0\sin {{\omega_-}} t\,\sin {{\omega_+}} t", 1.1),
             eq(r"\omega_\pm=\frac{ {{\omega_1}} \pm\omega_0}{2}", 1.0)],
        ]
        groups = []
        for parts in steps:
            g = VGroup(*parts).arrange(DOWN, buff=0.4)
            if g.width > 8.3:
                g.scale_to_fit_width(8.3)
            groups.append(g)

        prev = cur = None
        for i, g in enumerate(groups):
            until(STEPS[i])
            g.move_to([0, MID, 0])
            anims = [Write(g, run_time=0.9)]
            if cur is not None:
                anims.append(cur.animate(run_time=0.6).scale(0.66)
                             .move_to([0, TOP + 0.3 * (len(cur) - 1), 0]).set_opacity(0.36))
            if prev is not None:
                anims.append(FadeOut(prev, shift=UP * 0.6, run_time=0.5))
            self.play(*anims)
            prev, cur = cur, g

        until(STEPS[5])
        box_eq = VGroup(
            eq(r"T_b = \frac{\pi}{ {{\omega_-}} } = \frac{2\pi}{ {{\omega_1}} -\omega_0}", 1.3),
            Text("energy swaps sides every  Tₐ/2".replace("ₐ", "b"), font="Inter",
                 color=MUTED).scale(0.38),
        ).arrange(DOWN, buff=0.4)
        panel = RoundedRectangle(corner_radius=0.08, width=box_eq.width + 1.1, height=box_eq.height + 1.0,
                                 fill_color=PANEL, fill_opacity=1, stroke_color=EDGE, stroke_width=2)
        tab_txt = Text("RESULT · BEATS", font="Inter", weight=MEDIUM, color=MUTED).scale(0.34)
        tab = RoundedRectangle(corner_radius=0.05, width=tab_txt.width + 0.4, height=tab_txt.height + 0.26,
                               fill_color="#000000", fill_opacity=1, stroke_color=EDGE, stroke_width=1.6)
        theorem = VGroup(panel, box_eq).move_to([0, MID - 0.2, 0])
        tab.move_to(panel.get_corner(UL) + RIGHT * (tab.width / 2 + 0.3))
        tab_txt.move_to(tab)
        frame_box = SurroundingRectangle(box_eq[0], color=GOLD, buff=0.16, stroke_width=2.4, corner_radius=0.04)
        self.play(FadeOut(prev, run_time=0.4), cur.animate(run_time=0.5).set_opacity(0))
        self.remove(cur)
        self.play(FadeIn(theorem, shift=UP * 0.25), FadeIn(tab), FadeIn(tab_txt), run_time=0.6)
        self.play(Create(frame_box), run_time=0.5)

        until(MATH_END - 0.45)
        self.play(FadeOut(VGroup(theorem, tab, tab_txt, frame_box), shift=DOWN * 0.4), run_time=0.4)
        until(30.0 - 1.0 / 30)
