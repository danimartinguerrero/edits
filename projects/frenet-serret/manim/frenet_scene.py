"""Frenet trihedron on a helix — physics + math layer (Manim CE 0.22).
Notation from the user's Mecánica I notes (Tema 1, §4): tau-hat, n-hat, b-hat,
kappa, calligraphic T for torsion, Darboux vector omega = T tau + kappa b.
3D is projected by hand so the 2D portrait layout stays fully controllable.
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

A, B = 1.5, 0.32                      # helix r = (A cos t, A sin t, B t)
C2 = A * A + B * B
KAPPA, TORS = A / C2, B / C2
MATH_START, MATH_END = 12.0, 26.2
STEPS = [12.8, 14.9, 17.0, 19.1, 21.2, 23.3]
ELEV = 0.30

TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage{amsmath}\usepackage{amssymb}")


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def tp_of(T):            # parameter of the moving point
    return 1.05 * T


def frame_at(t):
    r = np.array([A * np.cos(t), A * np.sin(t), B * t])
    tau = np.array([-A * np.sin(t), A * np.cos(t), B]) / np.sqrt(C2)
    n = np.array([-np.cos(t), -np.sin(t), 0.0])
    b = np.cross(tau, n)
    return r, tau, n, b


def layout(T):
    if T < MATH_START:
        a = 0.0
    elif T < MATH_END:
        a = smooth((T - MATH_START) / 0.8)
    else:
        a = 1 - smooth((T - MATH_END) / 0.7)
    s = 1.3 + (0.6 - 1.3) * a
    c = np.array([0, 0.45, 0]) + (np.array([-2.2, 4.45, 0]) - np.array([0, 0.45, 0])) * a
    return s, c


def projector(T):
    """world -> screen (x, y, depth) ; camera azimuth drifts slowly, follows the point in z"""
    az = 0.55 + 0.11 * T
    zc = B * tp_of(T)
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(ELEV), np.sin(ELEV)
    s, c = layout(T)

    def P(p):
        p = np.atleast_2d(p).astype(float)
        x, y, z = p[:, 0], p[:, 1], p[:, 2] - zc
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        X = xr
        Y = z * ce + yr * se
        depth = -yr * ce + z * se
        f = 16.0 / (16.0 - depth)
        out = np.c_[X * f * s + c[0], Y * f * s + c[1], np.zeros(len(X))]
        return out, depth
    return P


def eq(s, scale=1.0):
    m = MathTex(s, tex_template=TEX, color=INK).scale(scale)
    for tok, col in ((r"\hat{\tau}", GOLD), (r"\hat{n}", BLUE), (r"\hat{b}", LAV), (r"\boldsymbol{\omega}", INK)):
        try:
            m.set_color_by_tex(tok, col, substring=False)
        except Exception:
            pass
    return m


class Frenet(Scene):
    def construct(self):
        T = lambda: self.renderer.time
        LBL = {k: MathTex(s, color=c).scale(0.95) for k, s, c in
               (("t", r"\hat{\tau}", GOLD), ("n", r"\hat{n}", BLUE), ("b", r"\hat{b}", LAV),
                ("w", r"\boldsymbol{\omega}", INK))}

        RHO = MathTex(r"\rho_c = 1/\kappa", color=BLUE)

        def scene_f():
            t = T()
            P = projector(t)
            s, _ = layout(t)
            tp = tp_of(t)
            g = VGroup()
            # helix, split in segments; far side dimmer, travelled part warmer
            ts = np.linspace(tp - 11.5, tp + 11.5, 700)
            pts = np.c_[A * np.cos(ts), A * np.sin(ts), B * ts]
            scr, dep = P(pts)
            seg = 14
            for i in range(0, len(ts) - seg, seg):
                j = i + seg + 1
                d = dep[i:j].mean()
                fade = smooth(1 - abs(ts[i:j].mean() - tp) / 11.5)
                behind = ts[i:j].mean() < tp
                col = interpolate_color(ManimColor(EDGE), ManimColor(GOLD if behind else LAV), 0.35 + 0.35 * behind)
                op = (0.35 + 0.65 * (d > 0)) * fade
                g.add(VMobject(stroke_color=col, stroke_width=(2.2 + 1.6 * (d > 0)) * (0.7 + 0.3 * min(s, 1)),
                               stroke_opacity=op).set_points_smoothly(scr[i:j]))
            # axis hint
            ax, _ = P(np.array([[0, 0, B * tp - 4.2], [0, 0, B * tp + 4.2]]))
            g.add(DashedLine(ax[0], ax[1], dash_length=0.12, stroke_color=EDGE, stroke_width=1.5))

            r, tau, n, b = frame_at(tp)
            L = 1.65
            o, _ = P(r)
            o = o[0]
            # osculating plane + circle (physics section highlight, again in step 1-2)
            osc = smooth((t - 5.6) / 0.6) * (1 - smooth((t - 11.2) / 0.5))
            if osc > 0.01:
                rho = 1 / KAPPA
                quad = [r + 2.3 * tau + 0.6 * n, r - 2.3 * tau + 0.6 * n, r - 2.3 * tau + (rho + 0.8) * n,
                        r + 2.3 * tau + (rho + 0.8) * n]
                q, _ = P(np.array(quad))
                g.add(Polygon(*q, stroke_width=1.2, stroke_color=LAV, stroke_opacity=0.5 * osc,
                              fill_color=LAV, fill_opacity=0.08 * osc))
                cen = r + rho * n
                ang = np.linspace(0, TAU, 120)
                circ = np.array([cen + rho * (np.cos(a) * (-n) + np.sin(a) * tau) for a in ang])
                cs, _ = P(circ)
                g.add(VMobject(stroke_color=BLUE, stroke_width=2.4, stroke_opacity=0.85 * osc)
                      .set_points_smoothly(cs))
                cp, _ = P(cen)
                g.add(Dot(cp[0], radius=0.05, color=BLUE).set_opacity(osc))
                g.add(DashedLine(o, cp[0], dash_length=0.08, stroke_color=BLUE, stroke_width=1.6,
                                 stroke_opacity=0.8 * osc))
                lab = RHO.copy().scale(0.7 * s)
                g.add(lab.move_to((o + cp[0]) / 2 + np.array([0.75, 0.15, 0]) * s).set_opacity(osc))
            # Darboux vector (steps 4-6 and outro)
            dw = smooth((t - STEPS[3]) / 0.6)
            if dw > 0.01:
                w = TORS * tau + KAPPA * b
                tip, _ = P(r + 2.6 * w / np.linalg.norm(w) * 1.0)
                g.add(Arrow(o, tip[0], buff=0, stroke_width=5 * s + 1, color=INK,
                            max_tip_length_to_length_ratio=0.18).set_opacity(dw))
                g.add(LBL["w"].copy().scale(max(s, 0.6)).move_to(tip[0] + (tip[0] - o) * 0.14).set_opacity(dw))
            for v, key, col in ((tau, "t", GOLD), (n, "n", BLUE), (b, "b", LAV)):
                tip, _ = P(r + L * v)
                g.add(Arrow(o, tip[0], buff=0, stroke_width=6 * s + 1.5, color=col,
                            max_tip_length_to_length_ratio=0.2))
                g.add(LBL[key].copy().scale(max(s, 0.6)).move_to(tip[0] + (tip[0] - o) * 0.2))
            g.add(Dot(o, radius=0.13 * s + 0.03, color=WHITE))
            g.add(Circle(radius=0.3 * s + 0.05, stroke_width=0, fill_color=GOLD, fill_opacity=0.18).move_to(o))
            return g

        ticker = Mobject().add_updater(lambda m, dt: None)
        self.add(ticker, always_redraw(scene_f))

        def until(t):
            dt = t - self.renderer.time
            if dt > 1e-3:
                self.wait(dt)

        until(MATH_START)
        TOP, MID = 1.6, -0.95
        steps = [
            [eq(r"s(t)=\int_{t_0}^{t}\lVert\dot{\mathbf r}\rVert\,dt',\qquad {{\hat{\tau}}} = \frac{d\mathbf r}{ds}", 1.1)],
            [eq(r"\kappa=\Big\lVert\frac{d {{\hat{\tau}}} }{ds}\Big\rVert,\quad {{\hat{n}}} =\frac{1}{\kappa}\frac{d {{\hat{\tau}}} }{ds},\quad {{\hat{b}}} = {{\hat{\tau}}} \times {{\hat{n}}}", 1.0)],
            [eq(r"\frac{d}{ds}\begin{pmatrix}\hat{\tau}\\ \hat{n}\\ \hat{b}\end{pmatrix}=\begin{pmatrix}0&\kappa&0\\-\kappa&0&\mathcal{T}\\0&-\mathcal{T}&0\end{pmatrix}\begin{pmatrix}\hat{\tau}\\ \hat{n}\\ \hat{b}\end{pmatrix}", 1.0)],
            [eq(r"\frac{d\hat{u}}{ds}= {{\boldsymbol{\omega}}} \times\hat{u},\qquad {{\boldsymbol{\omega}}} =\mathcal{T}\, {{\hat{\tau}}} +\kappa\, {{\hat{b}}}", 1.1)],
            [eq(r"\mathbf r=(a\cos t,\;a\sin t,\;b\,t),\quad \lVert\dot{\mathbf r}\rVert=\sqrt{a^2+b^2}", 0.95),
             eq(r"\kappa=\frac{a}{a^2+b^2},\qquad \mathcal{T}=\frac{b}{a^2+b^2}", 1.2)],
        ]
        groups = []
        for parts in steps:
            g = VGroup(*parts).arrange(DOWN, buff=0.42)
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
            eq(r"\kappa(s),\ \mathcal{T}(s)\ \Longrightarrow\ \text{the curve}", 1.15),
            MathTex(r"\text{(up to a rigid motion)}", color=MUTED).scale(0.75),
            eq(r"\kappa,\ \mathcal{T}\ \text{constant}\iff\text{helix}", 1.15),
        ).arrange(DOWN, buff=0.32)
        panel = RoundedRectangle(corner_radius=0.08, width=box_eq.width + 1.1, height=box_eq.height + 1.0,
                                 fill_color=PANEL, fill_opacity=1, stroke_color=EDGE, stroke_width=2)
        tab_txt = Text("THEOREM · CURVES", font="Inter", weight=MEDIUM, color=MUTED).scale(0.34)
        tab = RoundedRectangle(corner_radius=0.05, width=tab_txt.width + 0.4, height=tab_txt.height + 0.26,
                               fill_color="#000000", fill_opacity=1, stroke_color=EDGE, stroke_width=1.6)
        theorem = VGroup(panel, box_eq).move_to([0, MID - 0.2, 0])
        tab.move_to(panel.get_corner(UL) + RIGHT * (tab.width / 2 + 0.3))
        tab_txt.move_to(tab)
        frame_box = SurroundingRectangle(box_eq[2], color=GOLD, buff=0.16, stroke_width=2.4, corner_radius=0.04)
        self.play(FadeOut(prev, run_time=0.4), cur.animate(run_time=0.5).set_opacity(0))
        self.remove(cur)
        self.play(FadeIn(theorem, shift=UP * 0.25), FadeIn(tab), FadeIn(tab_txt), run_time=0.6)
        self.play(Create(frame_box), run_time=0.5)

        until(MATH_END - 0.45)
        self.play(FadeOut(VGroup(theorem, tab, tab_txt, frame_box), shift=DOWN * 0.4), run_time=0.4)
        until(30.0 - 1.0 / 30)
