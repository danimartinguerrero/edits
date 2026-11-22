"""Van Allen belts — trapped proton in Earth's dipole (Manim CE 0.22).
Trajectory is a real Boris-pusher integration (trap.py), gyroradius exaggerated
for visibility. Units: Earth radius = 1."""
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
ELEV = 0.38

TRAJ = np.load("traj.npy")
VPAR = np.load("vpar.npy")
H = float(np.load("h.npy")[0])
MIRR = np.where(np.diff(np.sign(VPAR)) != 0)[0]

TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage{amsmath}\usepackage{amssymb}")


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def sim_index(T):
    return int(min(len(TRAJ) - 1, max(0.0, 2.0 * (T - 0.3)) / H))


def layout(T):
    if T < MATH_START:
        a = 0.0
    elif T < MATH_END:
        a = smooth((T - MATH_START) / 0.8)
    else:
        a = 1 - smooth((T - MATH_END) / 0.7)
    s = 0.95 + (0.44 - 0.95) * a
    c = np.array([0, 0.3, 0]) + (np.array([-2.2, 4.45, 0]) - np.array([0, 0.3, 0])) * a
    return s, c


def projector(T):
    az = -0.4 + 0.05 * T
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(ELEV), np.sin(ELEV)
    s, c = layout(T)

    def P(p):
        p = np.atleast_2d(p).astype(float)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        xr = x * ca - y * sa
        yr = x * sa + y * ca
        X, Y = xr, z * ce + yr * se
        depth = -yr * ce + z * se
        f = 20.0 / (20.0 - depth)
        scr = np.c_[X * f * s + c[0], Y * f * s + c[1], np.zeros(len(X))]
        hidden = (depth < 0) & (np.hypot(X * f, Y * f) < 1.0)
        return scr, depth, hidden
    return P


def field_line(L, az, n=90):
    lm = np.arccos(np.sqrt(1 / L)) * 0.995
    lam = np.linspace(-lm, lm, n)
    r = L * np.cos(lam) ** 2
    return np.c_[r * np.cos(lam) * np.cos(az), r * np.cos(lam) * np.sin(az), r * np.sin(lam)]


def polyline_visible(g, scr, hidden, **kw):
    """add runs of visible points as separate strokes"""
    run = []
    for p, h in zip(scr, hidden):
        if not h:
            run.append(p)
        elif len(run) > 1:
            g.add(VMobject(**kw).set_points_as_corners(run)); run = []
        else:
            run = []
    if len(run) > 1:
        g.add(VMobject(**kw).set_points_as_corners(run))


def eq(s, scale=1.0):
    m = MathTex(s, tex_template=TEX, color=INK).scale(scale)
    for tok, col in ((r"\mu", GOLD), (r"\alpha", LAV), (r"B_m", BLUE), (r"\mathbf B", BLUE),
                     (r"v_\parallel", GOLD), (r"v_\perp", LAV)):
        try:
            m.set_color_by_tex(tok, col, substring=False)
        except Exception:
            pass
    return m


class VanAllen(Scene):
    def construct(self):
        T = lambda: self.renderer.time
        AZS = np.linspace(0, TAU, 12, endpoint=False)
        BELT_AZ = np.linspace(0, TAU, 30, endpoint=False)

        def scene_f():
            t = T()
            P = projector(t)
            s, c = layout(t)
            g = VGroup()
            # radiation belts: soft tori from many thick translucent field-line arcs
            for L, col, op in ((1.75, BLUE, 0.075), (4.2, LAV, 0.055)):
                for az in BELT_AZ:
                    pts = field_line(L, az, 40)
                    pts = pts[np.abs(np.arcsin(pts[:, 2] / np.linalg.norm(pts, axis=1))) < 0.62]
                    scr, d, hid = P(pts)
                    polyline_visible(g, scr, hid, stroke_color=col, stroke_width=34 * s * (L / 3) ** 0.6,
                                     stroke_opacity=op)
            # back field lines
            for L in (1.5, 2.0, 3.0, 4.6):
                for az in AZS:
                    scr, d, hid = P(field_line(L, az))
                    polyline_visible(g, scr, hid, stroke_color=LAV if L != 2.0 else BLUE,
                                     stroke_width=1.3, stroke_opacity=0.22 if L != 2.0 else 0.35)
            # Earth
            ec, _, _ = P(np.zeros(3))
            R = s * 20 / 20
            for k in range(6, 0, -1):
                g.add(Circle(radius=R * (1 + 0.05 * k), stroke_width=0, fill_color=BLUE,
                             fill_opacity=0.035).move_to(ec[0]))
            g.add(Circle(radius=R, stroke_color=BLUE, stroke_width=1.5, stroke_opacity=0.6,
                         fill_color="#0C1830", fill_opacity=1).move_to(ec[0]))
            g.add(Circle(radius=R * 0.72, stroke_width=0, fill_color="#1B3563", fill_opacity=0.55)
                  .move_to(ec[0] + np.array([-0.22, 0.25, 0]) * R))
            g.add(Circle(radius=R * 0.38, stroke_width=0, fill_color="#2C4F8A", fill_opacity=0.35)
                  .move_to(ec[0] + np.array([-0.35, 0.4, 0]) * R))
            th = np.linspace(0, TAU, 90)
            for lat in (-60, -30, 0, 30, 60):
                la = np.radians(lat)
                pts = np.c_[np.cos(la) * np.cos(th), np.cos(la) * np.sin(th), np.full_like(th, np.sin(la))]
                scr, d, _ = P(pts)
                polyline_visible(g, scr, d < 0, stroke_color=BLUE, stroke_width=1, stroke_opacity=0.25)
            for lo in np.linspace(0, np.pi, 6, endpoint=False):
                la = np.linspace(-np.pi / 2, np.pi / 2, 60)
                for sgn in (1, -1):
                    pts = np.c_[np.cos(la) * np.cos(lo) * sgn, np.cos(la) * np.sin(lo) * sgn, np.sin(la)]
                    scr, d, _ = P(pts)
                    polyline_visible(g, scr, d < 0, stroke_color=BLUE, stroke_width=1, stroke_opacity=0.2)
            # dipole axis
            ax, _, axh = P(np.array([[0, 0, -1.6], [0, 0, -1.0], [0, 0, 1.0], [0, 0, 1.6]]))
            g.add(Line(ax[0], ax[1], stroke_color=MUTED, stroke_width=1.5, stroke_opacity=0.5))
            g.add(Line(ax[2], ax[3], stroke_color=MUTED, stroke_width=1.5, stroke_opacity=0.5))
            # particle: faint full history (the drift shell) + bright recent trail
            i = sim_index(t)
            if t > 0.3 and i > 2:
                hist = TRAJ[:i:3]
                scr, d, hid = P(hist)
                polyline_visible(g, scr, hid, stroke_color=GOLD, stroke_width=1.1, stroke_opacity=0.22)
                j0 = max(0, i - int(5.0 / H))
                rec = TRAJ[j0:i + 1]
                scr, d, hid = P(rec)
                n = len(rec)
                for k in range(0, n - 1, 12):
                    seg_s = scr[k:k + 13]
                    if hid[k:k + 13].any():
                        continue
                    a = (k / max(n, 1)) ** 1.5
                    g.add(VMobject(stroke_color=GOLD, stroke_width=1.5 + 3.5 * a * s,
                                   stroke_opacity=0.15 + 0.85 * a).set_points_as_corners(seg_s))
                # mirror points, from step 4 on
                mp = smooth((t - STEPS[3]) / 0.5)
                if mp > 0.01:
                    for m in MIRR[MIRR < i][-8:]:
                        ms, _, mh = P(TRAJ[m])
                        if not mh[0]:
                            g.add(Dot(ms[0], radius=0.06, color=GOLD).set_opacity(mp))
            hp, _, hh = P(TRAJ[i])
            if not hh[0]:
                g.add(Circle(radius=0.22 * s + 0.05, stroke_width=0, fill_color=GOLD, fill_opacity=0.25).move_to(hp[0]))
                g.add(Dot(hp[0], radius=0.075 * s + 0.03, color=WHITE))
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
            [eq(r"m\,\frac{d\mathbf v}{dt}=q\,\mathbf v\times {{\mathbf B}}", 1.25),
             eq(r"\omega_c=\frac{qB}{m},\qquad r_L=\frac{m {{v_\perp}} }{qB}", 1.1)],
            [eq(r"{{\mu}} =\frac{m v_\perp^2}{2B}\;\approx\;\text{const}", 1.25)],
            [eq(r"v_\parallel^2+ v_\perp^2=v^2\ \Rightarrow\ \frac{\sin^2 {{\alpha}} }{B}=\text{const}", 1.1)],
            [eq(r"F_\parallel=- {{\mu}} \,\frac{\partial B}{\partial s}", 1.2),
             eq(r"{{v_\parallel}} =0\ \text{ at }\ {{B_m}} =\frac{B_{eq}}{\sin^2 {{\alpha}}_{eq}}", 1.1)],
            [eq(r"B(r,\lambda)=B_E\Big(\frac{R_E}{r}\Big)^3\sqrt{1+3\sin^2\lambda}", 1.05),
             eq(r"r=L\,R_E\cos^2\lambda", 1.1)],
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
            eq(r"\text{trapped}\iff\sin^2 {{\alpha}}_{eq}>\frac{B_{eq}}{B_{atm}}", 1.15),
            MathTex(r"\text{otherwise it falls into the loss cone}", color=MUTED).scale(0.72),
        ).arrange(DOWN, buff=0.36)
        panel = RoundedRectangle(corner_radius=0.08, width=box_eq.width + 1.1, height=box_eq.height + 1.0,
                                 fill_color=PANEL, fill_opacity=1, stroke_color=EDGE, stroke_width=2)
        tab_txt = Text("RESULT · MAGNETIC BOTTLE", font="Inter", weight=MEDIUM, color=MUTED).scale(0.34)
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
