"""Restitution & collisions — physics + math layer (Manim CE 0.22).
Follows the user's notes (Tema 3): pi_rest = e pi_comp, Huygens–Newton rule,
CM method u'_i = -e u_i, Q = 1/2 mu (1-e^2) v12^2, bouncing ball (Cor. 5.2)
T_inf = (1+e)/(1-e) sqrt(2h/g); collision lanes = Problem 3 (m vs 3m)."""
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

# ── bouncing ball (local coords) ─────────────────────────────────────────────
E_B = 0.8
U = 0.62                        # sqrt(2h/g) in seconds of video
TINF = (1 + E_B) / (1 - E_B) * U
GROUND, HTOP = -1.7, 1.9
X0, X1, TSPAN = -3.7, 3.7, 6.0
G_ACC = 2 * (HTOP - GROUND) / U ** 2


def ball_y(t):
    """height above ground at time t after release, and number of bounces so far"""
    if t <= 0:
        return HTOP - GROUND, 0
    if t < U:
        return (HTOP - GROUND) - 0.5 * G_ACC * t * t, 0
    t -= U
    n = 1
    while n < 80:
        d = 2 * E_B ** n * U
        if t < d:
            v = E_B ** n * G_ACC * U
            return v * t - 0.5 * G_ACC * t * t, n
        t -= d
        n += 1
    return 0.0, n


def bounce_times():
    ts, t = [U], U
    for n in range(1, 40):
        t += 2 * E_B ** n * U
        ts.append(t)
    return np.array(ts)


BT = bounce_times()

# ── collision lanes: m1 -> m2 = 3 m1 at rest (Problem 3) ────────────────────
M1, M2 = 1.0, 3.0
S1, S2 = 0.42, 0.42 * np.sqrt(3)
LANES = [1.0, 0.5, 0.0]
V0 = 2.0
XA, XB = -3.3, 0.35
TC = (XB - S2 / 2 - S1 / 2 - XA) / V0


def lane_state(tau, e):
    if tau < TC:
        return XA + V0 * tau, XB
    v1 = (M1 - e * M2) * V0 / (M1 + M2)
    v2 = (1 + e) * M1 * V0 / (M1 + M2)
    d = tau - TC
    return XA + V0 * TC + v1 * d, XB + v2 * d


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def win(t, a, b, f=0.4):
    return smooth((t - a) / f) * (1 - smooth((t - b) / f))


BIG_B = (1.0, np.array([0, 0.5, 0]))
SMALL = (0.46, np.array([-2.2, 4.5, 0]))


def lerp_layout(a, A, B):
    return A[0] + (B[0] - A[0]) * a, A[1] + (B[1] - A[1]) * a


def eq(s, scale=1.0):
    m = MathTex(s, tex_template=TEX, color=INK).scale(scale)
    for tok, col in ((r"e", GOLD), (r"\pi_{\text{comp}}", BLUE), (r"\pi_{\text{rest}}", LAV),
                     (r"V_{\text{CM}}", LAV), (r"Q", GOLD)):
        try:
            m.set_color_by_tex(tok, col, substring=False)
        except Exception:
            pass
    return m


class Restitution(Scene):
    def construct(self):
        T = lambda: self.renderer.time
        LBL_E = [MathTex(rf"e={e:g}", color=GOLD).scale(0.8) for e in LANES]
        LBL_T = MathTex(r"T_\infty", color=LAV).scale(1.0)
        LBL_M1 = MathTex(r"m", color=INK).scale(0.55)
        LBL_M2 = MathTex(r"3m", color=INK).scale(0.55)
        LBL_CM = MathTex(r"\text{CM}", color=LAV).scale(0.5)
        NUMS = [MathTex(rf"n = {k}", color=MUTED).scale(0.8) for k in range(0, 81)]

        def bounce_group(tb, s, c, op):
            g = VGroup()
            if op < 0.01:
                return g

            def to(x, y):
                return np.array([x * s + c[0], y * s + c[1], 0])
            g.add(Line(to(X0, GROUND), to(X1, GROUND), stroke_color=EDGE, stroke_width=3, stroke_opacity=op))
            xT = X0 + TINF / TSPAN * (X1 - X0)
            tl = smooth((tb - 3.6) / 0.5) if tb < 90 else 1
            if tl > 0:
                g.add(DashedLine(to(xT, GROUND), to(xT, HTOP + 0.2), dash_length=0.12, stroke_color=LAV,
                                 stroke_width=2, stroke_opacity=op * tl))
                g.add(LBL_T.copy().scale(s).move_to(to(xT, HTOP + 0.6)).set_opacity(op * tl))
            # trail
            tt = np.linspace(0, max(1e-3, min(tb, TSPAN)), max(4, int(min(tb, TSPAN) * 140)))
            pts = [to(X0 + t_ / TSPAN * (X1 - X0), GROUND + ball_y(t_)[0]) for t_ in tt]
            if len(pts) > 2 and tb > 0:
                g.add(VMobject(stroke_color=GOLD, stroke_width=2.6, stroke_opacity=0.8 * op).set_points_as_corners(pts))
            for b in BT[BT < tb][:30]:
                g.add(Dot(to(X0 + b / TSPAN * (X1 - X0), GROUND), radius=0.045 * s + 0.01, color=GOLD).set_opacity(op))
            tcur = max(0, min(tb, TSPAN))
            y, n = ball_y(tcur)
            bp = to(X0 + tcur / TSPAN * (X1 - X0), GROUND + y + 0.2)
            g.add(Circle(radius=0.36 * s, stroke_width=0, fill_color=GOLD, fill_opacity=0.18 * op).move_to(bp))
            g.add(Circle(radius=0.2 * s, stroke_color=WHITE, stroke_width=1.5, stroke_opacity=0.6 * op,
                         fill_color=GOLD, fill_opacity=op).move_to(bp))
            g.add(NUMS[min(n, 80)].copy().scale(s).move_to(to(X0 + 0.9, HTOP + 0.6)).set_opacity(op))
            return g

        def lanes_group(tau, s, c, op, lane_gap=1.55):
            g = VGroup()
            if op < 0.01:
                return g

            def to(x, y):
                return np.array([x * s + c[0], y * s + c[1], 0])
            for k, e in enumerate(LANES):
                y0 = (1 - k) * lane_gap
                g.add(Line(to(-3.9, y0 - 0.3), to(3.9, y0 - 0.3), stroke_color=EDGE, stroke_width=2, stroke_opacity=op))
                x1, x2 = lane_state(tau, e)
                x1, x2 = np.clip(x1, -3.9, 3.9), np.clip(x2, -3.9, 3.9)
                b1 = Square(side_length=S1 * s, stroke_width=0, fill_color=GOLD, fill_opacity=op).move_to(to(x1, y0 - 0.3 + S1 / 2))
                b2 = Square(side_length=S2 * s, stroke_width=0, fill_color=BLUE, fill_opacity=op).move_to(to(x2, y0 - 0.3 + S2 / 2))
                g.add(b1, b2)
                if s > 0.7:
                    g.add(LBL_M1.copy().move_to(b1).set_color(BLACK).set_opacity(op))
                    g.add(LBL_M2.copy().move_to(b2).set_color(BLACK).set_opacity(op))
                xc = (M1 * x1 + M2 * x2) / (M1 + M2)
                tri = Triangle(fill_color=LAV, fill_opacity=op, stroke_width=0).scale(0.09 * s).move_to(to(xc, y0 - 0.45))
                g.add(tri)
                g.add(LBL_E[k].copy().scale(s).move_to(to(-3.35, y0 + 0.55)).set_opacity(op))
            return g

        def scene_f():
            t = T()
            g = VGroup()
            # bouncing ball: big 0–6.2, small during steps 5–6, big again in the outro
            if t < 6.4:
                g.add(bounce_group(t - 0.35, *BIG_B, 1 - smooth((t - 6.0) / 0.35)))
            elif STEPS[4] - 0.3 < t < MATH_END + 0.2:
                op = win(t, STEPS[4] - 0.3, MATH_END - 0.1, 0.35)
                g.add(bounce_group(t - STEPS[4], *SMALL, op))
            elif t >= MATH_END:
                g.add(bounce_group(t - MATH_END - 0.25, *BIG_B, smooth((t - MATH_END) / 0.4)))
            # collision lanes: big 6.3–11.6, small (looping per step) during steps 1–4
            if 6.1 < t < 12.0:
                g.add(lanes_group(t - 6.6, 1.0, np.array([0, 0.3, 0]), win(t, 6.2, 11.5)))
            elif STEPS[0] - 0.4 < t < STEPS[4]:
                k = max(0, np.searchsorted(STEPS, t) - 1)
                tau = (t - STEPS[k]) * 1.25 - 0.1
                g.add(lanes_group(tau, 0.5, np.array([-2.2, 4.5, 0]), win(t, STEPS[0] - 0.4, STEPS[4] - 0.3, 0.3)))
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
            [eq(r"{{\pi_{\text{rest}}}} = {{e}} \, {{\pi_{\text{comp}}}} ,\qquad 0\le {{e}} \le 1", 1.25)],
            [eq(r"v_1'-v_2'=- {{e}} \,(v_1-v_2)", 1.3)],
            [eq(r"u_i=v_i- {{V_{\text{CM}}}} \ \ \Longrightarrow\ \ u_i'=- {{e}} \,u_i", 1.2)],
            [eq(r"{{Q}} =\tfrac12\,\mu\,(1- {{e}}^2)\,(v_1-v_2)^2", 1.25),
             MathTex(r"\text{the CM energy can never be dissipated}", color=MUTED).scale(0.75)],
            [eq(r"v_n= {{e}}^n\sqrt{2gh},\qquad h_n= {{e}}^{2n}h", 1.25)],
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
            eq(r"T_\infty=\frac{1+ {{e}} }{1- {{e}} }\sqrt{\frac{2h}{g}}", 1.35),
            MathTex(r"e=0.8:\ \ T_\infty=9\sqrt{2h/g}", color=MUTED).scale(0.8),
            MathTex(r"\text{infinitely many bounces, finite time}", color=MUTED).scale(0.72),
        ).arrange(DOWN, buff=0.3)
        panel = RoundedRectangle(corner_radius=0.08, width=box_eq.width + 1.1, height=box_eq.height + 1.0,
                                 fill_color=PANEL, fill_opacity=1, stroke_color=EDGE, stroke_width=2)
        tab_txt = Text("COROLLARY · BOUNCING BALL", font="Inter", weight=MEDIUM, color=MUTED).scale(0.34)
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
