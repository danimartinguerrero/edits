"""Potential / energy diagrams & Kepler orbits — physics + math layer.
Notation from the user's notes (Tema 2 §1: energy diagram, turning points,
omega^2 = V''(x0)/m; two-body -> mu problem). Units mu = k = L = 1."""
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

D = np.load("orbit.npz")
TA, TB = D["tA"], D["tB"]

TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage{amsmath}\usepackage{amssymb}")


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def win(t, a, b, f=0.4):
    return smooth((t - a) / f) * (1 - smooth((t - b) / f))


def state(t):
    tr = TA if t < 12.0 else TB
    i = int(np.clip(round((t - tr[0][0]) * 30), 0, tr.shape[1] - 1))
    return tr, i


def Uef(r):
    return 0.5 / r ** 2 - 1 / r


R0, R1 = 0.38, 6.6
UMIN, UMAX = -0.62, 0.55


def diagram(t, s, c, op, labels):
    g = VGroup()
    if op < 0.01:
        return g
    W, H = 7.2, 3.9

    def to(r, U):
        x = -W / 2 + (r - R0) / (R1 - R0) * W
        y = -H / 2 + (U - UMIN) / (UMAX - UMIN) * H
        return np.array([x * s + c[0], y * s + c[1], 0])
    tr, i = state(t)
    E, r = tr[3][i], tr[2][i]
    # axes
    g.add(Line(to(R0, UMIN), to(R0, UMAX), stroke_color=EDGE, stroke_width=2, stroke_opacity=op))
    g.add(Line(to(R0, 0), to(R1, 0), stroke_color=EDGE, stroke_width=2, stroke_opacity=op))
    rr = np.linspace(0.42, R1, 260)
    uu = np.clip(Uef(rr), UMIN, UMAX)
    g.add(VMobject(stroke_color=LAV, stroke_width=3.4, stroke_opacity=op)
          .set_points_smoothly([to(a, b) for a, b in zip(rr, uu)]))
    # energy line: solid where allowed (U <= E), faint where forbidden
    allowed = Uef(rr) <= E
    run = []
    for a, ok in zip(rr, allowed):
        if ok:
            run.append(to(a, E))
        elif len(run) > 1:
            g.add(Line(run[0], run[-1], stroke_color=GOLD, stroke_width=3.2, stroke_opacity=op)); run = []
        else:
            run = []
    if len(run) > 1:
        g.add(Line(run[0], run[-1], stroke_color=GOLD, stroke_width=3.2, stroke_opacity=op))
    g.add(DashedLine(to(R0, E), to(R1, E), dash_length=0.1, stroke_color=GOLD, stroke_width=1.4,
                     stroke_opacity=0.35 * op))
    # turning points
    e = np.sqrt(max(0, 1 + 2 * E))
    tps = [1 / (1 + e)] + ([1 / (1 - e)] if e < 0.999 else [])
    for k, rt in enumerate(tps):
        if R0 < rt < R1:
            g.add(Dot(to(rt, E), radius=0.065 * s + 0.01, color=GOLD).set_opacity(op))
            g.add(DashedLine(to(rt, E), to(rt, 0), dash_length=0.07, stroke_color=GOLD, stroke_width=1.2,
                             stroke_opacity=0.5 * op))
            if s > 0.7 and e > 0.08:
                g.add(labels["rmin" if k == 0 else "rmax"].copy().move_to(to(rt, 0) + DOWN * 0.3).set_opacity(op))
    # the particle on the energy line
    if r < R1:
        g.add(Circle(radius=0.2 * s + 0.04, stroke_width=0, fill_color=GOLD, fill_opacity=0.22 * op).move_to(to(r, E)))
        g.add(Dot(to(r, E), radius=0.09 * s + 0.02, color=WHITE).set_opacity(op))
    g.add(labels["U"].copy().scale(max(s, 0.55)).move_to(to(R1 - 0.6, Uef(R1 - 0.6)) + UP * 0.45 * s).set_opacity(op))
    g.add(labels["r"].copy().scale(max(s, 0.55)).move_to(to(R1, 0) + RIGHT * 0.25 * s + DOWN * 0.2 * s).set_opacity(op))
    g.add(labels["E"].copy().scale(max(s, 0.55)).move_to(to(R0, E) + LEFT * 0.3 * s).set_opacity(op))
    return g


def orbit_view(t, s, c, op):
    g = VGroup()
    if op < 0.01:
        return g

    def to(x, y):
        return np.array([x * s + c[0], y * s + c[1], 0])
    tr, i = state(t)
    E = tr[3][i]
    e = np.sqrt(max(0, 1 + 2 * E))
    if e < 0.999:
        ph = np.linspace(0, TAU, 240)
    else:
        lim = np.arccos(-1 / e) - 0.06
        ph = np.linspace(-lim, lim, 240)
    rr = 1 / (1 + e * np.cos(ph))
    pts = [to(r * np.cos(p), r * np.sin(p)) for r, p in zip(rr, ph) if r < 9]
    g.add(VMobject(stroke_color=EDGE, stroke_width=2, stroke_opacity=op).set_points_smoothly(pts))
    # star (focus)
    for k, a in ((3, 0.08), (2, 0.14), (1, 0.3)):
        g.add(Circle(radius=0.12 * k * s, stroke_width=0, fill_color=GOLD, fill_opacity=a * op).move_to(to(0, 0)))
    g.add(Dot(to(0, 0), radius=0.1 * s, color="#FFF4D6").set_opacity(op))
    # trail
    j0 = max(0, i - 40)
    tp = [to(r * np.cos(p), r * np.sin(p)) for r, p in zip(tr[2][j0:i + 1], tr[1][j0:i + 1]) if r < 12]
    if len(tp) > 2:
        g.add(VMobject(stroke_color=BLUE, stroke_width=3, stroke_opacity=0.8 * op).set_points_as_corners(tp))
    r, p = tr[2][i], tr[1][i]
    if r < 12:
        pos = to(r * np.cos(p), r * np.sin(p))
        g.add(Circle(radius=0.22 * s, stroke_width=0, fill_color=BLUE, fill_opacity=0.25 * op).move_to(pos))
        g.add(Dot(pos, radius=0.11 * s, color=BLUE).set_opacity(op))
    return g


def eq(s, scale=1.0):
    m = MathTex(s, tex_template=TEX, color=INK).scale(scale)
    for tok, col in ((r"E", GOLD), (r"U_{\text{ef}}", LAV), (r"\varepsilon", BLUE)):
        try:
            m.set_color_by_tex(tok, col, substring=False)
        except Exception:
            pass
    return m


class Potential(Scene):
    def construct(self):
        T = lambda: self.renderer.time
        LBL = {"U": MathTex(r"U_{\text{ef}}(r)", color=LAV).scale(0.75),
               "r": MathTex(r"r", color=MUTED).scale(0.8),
               "E": MathTex(r"E", color=GOLD).scale(0.8),
               "rmin": MathTex(r"r_{\min}", color=GOLD).scale(0.6),
               "rmax": MathTex(r"r_{\max}", color=GOLD).scale(0.6)}

        def scene_f():
            t = T()
            g = VGroup()
            big = 1 - win(t, MATH_START, MATH_END, 0.7)
            if t < MATH_START + 0.8 or t > MATH_END - 0.1:
                op = win(t, -1, MATH_START - 0.2, 0.3) if t < 20 else smooth((t - MATH_END) / 0.5)
                g.add(orbit_view(t, 1.15, np.array([1.45, 2.6, 0]), op))
                g.add(diagram(t, 1.0, np.array([0, -3.2, 0]), op, LBL))
            if MATH_START - 0.2 < t < MATH_END + 0.2:
                op = win(t, MATH_START + 0.1, MATH_END - 0.15, 0.35)
                g.add(diagram(t, 0.48, np.array([-2.2, 4.6, 0]), op, LBL))
            return g

        holder = VGroup()

        def swap(m, dt):
            for o in m.get_family()[1:]:
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
            [eq(r"{{E}} =\tfrac12 m\dot x^2+V(x)\ \Longrightarrow\ V(x)\le {{E}}", 1.15),
             MathTex(r"\text{turning points: } V(x_i)=E", color=MUTED).scale(0.8)],
            [eq(r"V'(x_0)=0,\qquad \omega^2=\frac{V''(x_0)}{m}", 1.25),
             MathTex(r"\text{stable if } V''(x_0)>0", color=MUTED).scale(0.8)],
            [eq(r"L=\mu r^2\dot\varphi,\qquad {{E}} =\tfrac12\mu\dot r^2+\frac{L^2}{2\mu r^2}+V(r)", 1.05)],
            [eq(r"{{U_{\text{ef}}}} (r)=\frac{L^2}{2\mu r^2}-\frac{k}{r}", 1.3),
             MathTex(r"\text{centrifugal wall} \;+\; \text{gravity well}", color=MUTED).scale(0.8)],
            [eq(r"r(\varphi)=\frac{p}{1+ {{\varepsilon}} \cos\varphi},\qquad {{\varepsilon}} =\sqrt{1+\frac{2 {{E}} L^2}{\mu k^2}}", 1.05)],
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
        rows = VGroup(*[MathTex(a, color=c).scale(0.95) for a, c in (
            (r"E=E_{\min}=-\frac{\mu k^2}{2L^2}\ \ \to\ \ \text{circle}", INK),
            (r"E<0\ \ \to\ \ \text{ellipse (bound)}", INK),
            (r"E=0\ \ \to\ \ \text{parabola}", INK),
            (r"E>0\ \ \to\ \ \text{hyperbola (escape)}", GOLD))]).arrange(DOWN, buff=0.28, aligned_edge=LEFT)
        panel = RoundedRectangle(corner_radius=0.08, width=rows.width + 1.1, height=rows.height + 1.0,
                                 fill_color=PANEL, fill_opacity=1, stroke_color=EDGE, stroke_width=2)
        tab_txt = Text("ENERGY DIAGRAM · KEPLER", font="Inter", weight=MEDIUM, color=MUTED).scale(0.34)
        tab = RoundedRectangle(corner_radius=0.05, width=tab_txt.width + 0.4, height=tab_txt.height + 0.26,
                               fill_color="#000000", fill_opacity=1, stroke_color=EDGE, stroke_width=1.6)
        theorem = VGroup(panel, rows).move_to([0, MID - 0.2, 0])
        tab.move_to(panel.get_corner(UL) + RIGHT * (tab.width / 2 + 0.3))
        tab_txt.move_to(tab)
        frame_box = SurroundingRectangle(rows[1], color=GOLD, buff=0.12, stroke_width=2.4, corner_radius=0.04)
        self.play(FadeOut(prev, run_time=0.4), cur.animate(run_time=0.5).set_opacity(0))
        self.remove(cur)
        self.play(FadeIn(theorem, shift=UP * 0.25), FadeIn(tab), FadeIn(tab_txt), run_time=0.6)
        self.play(Create(frame_box), run_time=0.5)

        until(MATH_END - 0.45)
        self.play(FadeOut(VGroup(theorem, tab, tab_txt, frame_box), shift=DOWN * 0.4), run_time=0.4)
        until(30.0 - 1.0 / 30)
