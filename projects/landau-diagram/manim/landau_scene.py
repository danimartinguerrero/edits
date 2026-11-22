"""Landau diagram for elastic collisions — physics + math layer (Manim CE 0.22).
Notation from the user's notes (Tema 3, §3.1): lambda = m1/m2, AO = lambda q,
OB = q, circle of radius q, OC = q1', AC = p1', CB = p2', tan(alpha) =
sin(theta)/(lambda + cos(theta)), p2' = 2q sin(theta/2), beta = (pi - theta)/2."""
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
MATH_START, MATH_END = 12.0, 26.2
STEPS = [12.8, 14.9, 17.0, 19.1, 21.2, 23.3]

TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage{amsmath}\usepackage{amssymb}")


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def win(t, a, b, f=0.4):
    return smooth((t - a) / f) * (1 - smooth((t - b) / f))


# ── billiards (equal masses) ────────────────────────────────────────────────
R_B = 0.34
TGT = np.array([0.9, 0.0])
VB = 4.2
SHOTS = [(0.3, 0.16), (2.3, 0.42), (4.3, 0.6)]      # (start, impact parameter)
SHOT_LEN = 2.0


def shot_state(tau, b):
    """cue & target positions, contact point, outgoing dirs"""
    xc = TGT[0] - np.sqrt((2 * R_B) ** 2 - b * b)
    start = np.array([-3.9, b])
    tc = (xc - start[0]) / VB
    v = np.array([VB, 0.0])
    contact_cue = np.array([xc, b])
    n = (TGT - contact_cue) / (2 * R_B)
    v2 = np.dot(v, n) * n
    v1 = v - v2
    if tau < tc:
        return start + v * tau, TGT.copy(), contact_cue, v1, v2, False
    d = tau - tc
    return contact_cue + v1 * d, TGT + v2 * d, contact_cue, v1, v2, True


def billiards(t, s, c, op, shots=SHOTS, origin_t=0.0):
    g = VGroup()
    if op < 0.01:
        return g

    def to(p):
        return np.array([p[0] * s + c[0], p[1] * s + c[1], 0])
    cur = None
    for k, (t0, b) in enumerate(shots):
        tau = t - origin_t - t0
        if tau < 0:
            continue
        fade = 1.0 if k == len(shots) - 1 or tau < SHOT_LEN else max(0.0, 1 - (tau - SHOT_LEN) / 0.5) * 0.0 + 0.25
        if k < len(shots) - 1 and t - origin_t >= shots[k + 1][0]:
            fade = 0.22
        tt = min(tau, SHOT_LEN + 0.4)
        p1, p2, cc, v1, v2, hit = shot_state(tt, b)
        start = np.array([-3.9, b])
        # trails
        if not hit:
            g.add(Line(to(start), to(p1), stroke_color=GOLD, stroke_width=3, stroke_opacity=0.5 * fade * op))
        else:
            g.add(Line(to(start), to(cc), stroke_color=GOLD, stroke_width=3, stroke_opacity=0.5 * fade * op))
            g.add(Line(to(cc), to(p1), stroke_color=GOLD, stroke_width=3, stroke_opacity=0.8 * fade * op))
            g.add(Line(to(TGT), to(p2), stroke_color=BLUE, stroke_width=3, stroke_opacity=0.8 * fade * op))
            # right-angle marker at the contact point (between outgoing directions)
            if np.linalg.norm(v1) > 1e-3:
                ra = smooth((tt - (SHOT_LEN - 1.2)) / 0.3)
                u1, u2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
                k0 = 0.32
                pts = [cc + u1 * k0, cc + (u1 + u2) * k0, cc + u2 * k0]
                g.add(VMobject(stroke_color=INK, stroke_width=2.6, stroke_opacity=ra * fade * op)
                      .set_points_as_corners([to(p) for p in pts]))
                g.add(DashedLine(to(cc), to(cc + u2 * 1.1), dash_length=0.08, stroke_color=BLUE,
                                 stroke_width=1.6, stroke_opacity=0.6 * ra * fade * op))
        if k == len([sh for sh in shots if t - origin_t >= sh[0]]) - 1:
            cur = (p1, p2)
    if cur is not None:
        for p, col in ((cur[0], GOLD), (cur[1], BLUE)):
            g.add(Circle(radius=R_B * s * 1.5, stroke_width=0, fill_color=col, fill_opacity=0.16 * op).move_to(to(p)))
            g.add(Circle(radius=R_B * s, stroke_color=WHITE, stroke_width=1.4, stroke_opacity=0.5 * op,
                         fill_color=col, fill_opacity=op).move_to(to(p)))
    return g


# ── Landau diagram ──────────────────────────────────────────────────────────
def landau(lam, theta, s, c, op, q=1.75, show_max=0.0, labels=None):
    g = VGroup()
    if op < 0.01:
        return g
    off = np.array([(lam * q - q) / 2, 0])          # centre AB on screen

    def to(p):
        p = np.asarray(p, float) + off
        return np.array([p[0] * s + c[0], p[1] * s + c[1], 0])
    A, O, B = np.array([-lam * q, 0]), np.array([0.0, 0]), np.array([q, 0])
    C = q * np.array([np.cos(theta), np.sin(theta)])
    circ = [to(q * np.array([np.cos(a), np.sin(a)])) for a in np.linspace(0, TAU, 160)]
    g.add(VMobject(stroke_color=EDGE, stroke_width=2.2, stroke_opacity=op).set_points_as_corners(circ))
    g.add(Line(to(A), to(B), stroke_color=MUTED, stroke_width=2.2, stroke_opacity=0.8 * op))
    g.add(Line(to(O), to(C), stroke_color=LAV, stroke_width=3, stroke_opacity=op))
    g.add(Arrow(to(A), to(C), buff=0, color=GOLD, stroke_width=6 * s + 1, max_tip_length_to_length_ratio=0.12).set_opacity(op))
    g.add(Arrow(to(C), to(B), buff=0, color=BLUE, stroke_width=6 * s + 1, max_tip_length_to_length_ratio=0.12).set_opacity(op))
    # right angle at C when A lies on the circle
    on = max(0.0, 1 - abs(lam - 1) / 0.08)
    if on > 0.01:
        u1 = (A - C) / np.linalg.norm(A - C); u2 = (B - C) / np.linalg.norm(B - C)
        k0 = 0.28
        g.add(VMobject(stroke_color=INK, stroke_width=2.6, stroke_opacity=on * op)
              .set_points_as_corners([to(C + u1 * k0), to(C + (u1 + u2) * k0), to(C + u2 * k0)]))
    if show_max > 0.01 and lam > 1:
        th = np.arccos(-1 / lam)
        Cm = q * np.array([np.cos(th), np.sin(th)])
        far = A + (Cm - A) * 1.35
        g.add(DashedLine(to(A), to(far), dash_length=0.1, stroke_color=GOLD, stroke_width=2,
                         stroke_opacity=0.8 * show_max * op))
        g.add(Dot(to(Cm), radius=0.05 * s + 0.02, color=GOLD).set_opacity(show_max * op))
    for P_ in (A, O, B):
        g.add(Dot(to(P_), radius=0.05 * s + 0.02, color=INK).set_opacity(op))
    g.add(Dot(to(C), radius=0.07 * s + 0.02, color=WHITE).set_opacity(op))
    if labels:
        for key, P_, d in (("A", A, DL), ("O", O, DOWN), ("B", B, DR), ("C", C, UP)):
            g.add(labels[key].copy().scale(s).move_to(to(P_) + d * 0.38 * s).set_opacity(op))
    return g


def eq(s, scale=1.0):
    m = MathTex(s, tex_template=TEX, color=INK).scale(scale)
    for tok, col in ((r"\mathbf p_1'", GOLD), (r"\mathbf p_2'", BLUE), (r"\mathbf q_1'", LAV),
                     (r"p_2'", BLUE), (r"\alpha", GOLD), (r"\beta", BLUE), (r"\theta", LAV)):
        try:
            m.set_color_by_tex(tok, col, substring=False)
        except Exception:
            pass
    return m


class Landau(Scene):
    def construct(self):
        T = lambda: self.renderer.time
        LBL = {k: MathTex(k, color=MUTED).scale(0.8) for k in "AOBC"}
        LBL["C"].set_color(INK)

        def lam_theta(t):
            # physics section: lambda = 1 sweep, then grow to 2
            if t < MATH_START:
                lam = 1 + smooth((t - 9.6) / 0.9)
                th = 0.35 + (np.pi - 0.7) * (0.5 - 0.5 * np.cos((t - 6.2) * 1.05))
                return lam, th, smooth((t - 10.6) / 0.5)
            # math section: lambda = 1/2 (steps 1-3), 1 (step 4), 2 (steps 5-6)
            lam = 0.5 + 0.5 * smooth((t - STEPS[3]) / 0.6) + smooth((t - STEPS[4]) / 0.6)
            th = 0.35 + (np.pi - 0.7) * (0.5 - 0.5 * np.cos((t - 12) * 1.3))
            return lam, th, smooth((t - STEPS[4] - 0.5) / 0.5)

        def scene_f():
            t = T()
            g = VGroup()
            if t < 6.5:
                g.add(billiards(t, 1.0, np.array([0, 0.4, 0]), 1 - smooth((t - 6.1) / 0.35)))
            lam, th, mx = lam_theta(t)
            if 6.0 < t < MATH_START:
                g.add(landau(lam, th, 1.0, np.array([0, 0.5, 0]), win(t, 6.1, 11.6, 0.4), show_max=mx, labels=LBL))
            elif MATH_START - 0.2 < t < MATH_END + 0.1:
                g.add(landau(lam, th, 0.55, np.array([-2.2, 4.5, 0]), win(t, MATH_START, MATH_END - 0.1, 0.4),
                             show_max=mx, labels=LBL))
            if t > MATH_END - 0.1:
                shots2 = [(0.0, 0.5), (1.9, 0.25)]
                g.add(billiards(t, 1.0, np.array([0, 0.4, 0]), smooth((t - MATH_END) / 0.4),
                                shots=shots2, origin_t=MATH_END + 0.1))
            return g

        holder = VGroup()

        def swap(m, dt):                    # replace children each frame (no costly become/align)
            for o in m.get_family()[1:]:    # empty stale children: Manim keeps them in its moving list
                o.submobjects = []
                if isinstance(o, VMobject):
                    o.clear_points()
            m.submobjects = [scene_f()]
        holder.add_updater(swap)
        self.add(holder)

        def until(t):
            dt = t - self.renderer.time
            if dt > 1e-3:
                self.wait(dt)

        until(MATH_START)
        TOP, MID = 1.6, -0.95
        steps = [
            [eq(r"\lambda=\frac{m_1}{m_2},\qquad \mathbf q_1=\mu\,\mathbf v_1,\qquad |\mathbf q_1'|=|\mathbf q_1|=q", 1.05)],
            [eq(r"{{\mathbf p_1'}} =\lambda\,\mathbf q_1+ {{\mathbf q_1'}} ,\qquad {{\mathbf p_2'}} =\mathbf q_1- {{\mathbf q_1'}}", 1.1)],
            [eq(r"\tan {{\alpha}} =\frac{\sin {{\theta}} }{\lambda+\cos {{\theta}} },\qquad {{p_2'}} =2q\sin\frac{ {{\theta}} }{2}", 1.05),
             eq(r"{{\beta}} =\frac{\pi- {{\theta}} }{2}", 1.05)],
            [eq(r"m_1=m_2:\quad {{\alpha}} =\frac{ {{\theta}} }{2},\qquad {{\alpha}} + {{\beta}} =\frac{\pi}{2}", 1.15)],
            [eq(r"m_1>m_2:\quad \sin {{\alpha}}_{\text{max}}=\frac{m_2}{m_1}", 1.2),
             MathTex(r"\text{measure }\alpha_{\text{max}}\text{ and you weigh the target}", color=MUTED).scale(0.75)],
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
            eq(r"\frac{T_2'}{T_1}=\frac{4m_1m_2}{(m_1+m_2)^2}\,\sin^2\frac{ {{\theta}} }{2}", 1.2),
            MathTex(r"\text{all of it only if } m_1=m_2,\ \theta=\pi", color=MUTED).scale(0.75),
        ).arrange(DOWN, buff=0.34)
        panel = RoundedRectangle(corner_radius=0.08, width=box_eq.width + 1.1, height=box_eq.height + 1.0,
                                 fill_color=PANEL, fill_opacity=1, stroke_color=EDGE, stroke_width=2)
        tab_txt = Text("PROPOSITION · ENERGY TO THE TARGET", font="Inter", weight=MEDIUM, color=MUTED).scale(0.34)
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
