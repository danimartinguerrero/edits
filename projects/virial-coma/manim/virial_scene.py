"""Virial theorem & the Coma cluster — physics + math layer (Manim CE 0.22).
Notation from the user's notes (Tema 2, §7): S = sum r.p, <T>, V, Euler lemma,
2<T> = n<V>, Zwicky's M ~ R<v^2>/G. The cluster is a real N-body Plummer
sphere (cluster.py) in virial equilibrium."""
import numpy as np
from scipy.ndimage import gaussian_filter
from manim import *

config.pixel_width = 1080
config.pixel_height = 1920
config.frame_height = 16.0
config.frame_width = 9.0
config.frame_rate = 30
config.background_color = "#000000"

INK, MUTED, PANEL, EDGE = "#F2F2F2", "#9BA2AF", "#0E1014", "#4D5466"
BLUE, LAV, GOLD, RED = "#6FA8FF", "#B7A6F5", "#E6C36A", "#F07A6A"
MATH_START, MATH_END = 12.0, 26.2
STEPS = [12.8, 14.9, 17.0, 19.1, 21.2, 23.3]

PP = np.load("P.npy"); VV = np.load("V.npy"); HH = float(np.load("h.npy")[0])
N = PP.shape[1]
rng = np.random.default_rng(2)
LUM = np.exp(rng.normal(0, 0.6, N)); LUM /= LUM.max()
_r0 = np.linalg.norm(PP[0], axis=1)
GIANTS = np.argsort(_r0)[:2]           # two giant ellipticals near the core (cf. NGC 4874/4889)
LUM[GIANTS] = 3.0
SIZE = 1.0 + 2.2 * LUM
STARS = rng.random((260, 2)); STAR_B = rng.random(260) ** 3

IMG = 1000          # raster size
SPAN = 8.6          # world units across the raster
WSCALE = 1.25       # sim units -> world units

TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage{amsmath}\usepackage{amssymb}")


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def layout(T):
    if T < MATH_START:
        a = 0.0
    elif T < MATH_END:
        a = smooth((T - MATH_START) / 0.8)
    else:
        a = 1 - smooth((T - MATH_END) / 0.7)
    w = SPAN + (3.9 - SPAN) * a
    c = np.array([0, 1.0, 0]) + (np.array([-2.2, 4.45, 0]) - np.array([0, 1.0, 0])) * a
    return w, c


def hexrgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255


CB, CR, CW = hexrgb(BLUE), hexrgb(RED), np.array([0.96, 0.95, 0.92])


def view(T):
    k = min(len(PP) - 1, int(0.6 * T / HH))
    az = 0.3 + 0.045 * T
    ca, sa = np.cos(az), np.sin(az)
    p, v = PP[k], VV[k]
    x = p[:, 0] * ca - p[:, 1] * sa
    y = p[:, 2]
    vlos = v[:, 0] * sa + v[:, 1] * ca          # toward/away from the viewer
    depth = p[:, 0] * sa + p[:, 1] * ca
    return x, y, vlos, depth


def raster(T):
    x, y, vlos, depth = view(T)
    dop = smooth((T - 3.6) / 1.0)
    halo = smooth((T - 26.5) / 1.4)
    img = np.zeros((IMG, IMG, 3))
    # faint starfield
    sx = (STARS[:, 0] * IMG).astype(int); sy = (STARS[:, 1] * IMG).astype(int)
    img[sy, sx] += 0.25 * STAR_B[:, None]
    # dark-matter halo (outro): soft lavender glow far larger than the light
    if halo > 0:
        yy, xx = (np.mgrid[0:IMG, 0:IMG] - IMG / 2) / IMG * SPAN
        rr = np.hypot(xx, yy)
        img += (0.22 * halo * np.exp(-(rr / 2.8) ** 2))[..., None] * hexrgb(LAV)
    px = (x * WSCALE / SPAN + 0.5) * IMG
    py = (0.5 - y * WSCALE / SPAN) * IMG
    t = np.clip(vlos / 0.5, -1, 1)
    col = np.where(t[:, None] < 0, CW + (CB - CW) * (-t[:, None]), CW + (CR - CW) * t[:, None])
    col = CW + (col - CW) * dop
    core = np.zeros((IMG, IMG, 3))
    ok = (px > 1) & (px < IMG - 2) & (py > 1) & (py < IMG - 2)
    ix, iy = px[ok].astype(int), py[ok].astype(int)
    w = (0.55 + 0.45 * LUM[ok])[:, None] * col[ok]
    np.add.at(core, (iy, ix), w)
    sharp = gaussian_filter(core, (1.5, 1.5, 0)) * 15
    glow = gaussian_filter(core, (8, 8, 0)) * 95
    img = img + sharp + glow * 0.9
    img = 1 - np.exp(-img * 1.4)
    rgba = np.dstack([(np.clip(img, 0, 1) * 255).astype(np.uint8), np.full((IMG, IMG), 255, np.uint8)])
    # fade edges to black so the square raster disappears
    yy, xx = (np.mgrid[0:IMG, 0:IMG] - IMG / 2) / (IMG / 2)
    edge = np.clip((1 - np.maximum(np.abs(xx), np.abs(yy))) / 0.12, 0, 1)
    rgba[..., :3] = (rgba[..., :3] * edge[..., None]).astype(np.uint8)
    return rgba


def eq(s, scale=1.0):
    m = MathTex(s, tex_template=TEX, color=INK).scale(scale)
    for tok, col in ((r"\langle T\rangle", GOLD), (r"\langle V\rangle", BLUE), (r"V", BLUE),
                     (r"\langle v^2\rangle", GOLD), (r"M", LAV)):
        try:
            m.set_color_by_tex(tok, col, substring=False)
        except Exception:
            pass
    return m


class Virial(Scene):
    def construct(self):
        T = lambda: self.renderer.time
        img = ImageMobject(raster(0.0))
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        self.lastf = -1

        def upd(m, dt):
            t = T()
            f = int(round(t * 30))
            if f != self.lastf:
                self.lastf = f
                m.pixel_array = raster(t)
            w, c = layout(t)
            m.set(width=w).move_to(c)

        img.add_updater(upd)

        # line-of-sight velocity histogram, then mass comparison
        BINS = np.linspace(-1.6, 1.6, 27)
        HY, HW = -4.55, 7.4

        def hist_f():
            t = T()
            g = VGroup()
            op = smooth((t - 3.9) / 0.6) * (1 - smooth((t - 8.3) / 0.4))
            if op > 0.01:
                _, _, vlos, _ = view(t)
                hcount, _ = np.histogram(vlos, BINS)
                bw = HW / (len(BINS) - 1)
                for i, cnt in enumerate(hcount):
                    c = (BINS[i] + BINS[i + 1]) / 2
                    tt = np.clip(c / 0.75, -1, 1)
                    col = interpolate_color(ManimColor("#F2F0EB"), ManimColor(BLUE if tt < 0 else RED), abs(tt))
                    h = cnt / N * 13.0
                    g.add(Rectangle(width=bw * 0.78, height=max(h, 0.02), stroke_width=0, fill_color=col,
                                    fill_opacity=0.9 * op).move_to([-HW / 2 + (i + 0.5) * bw, HY + h / 2, 0]))
                g.add(Line([-HW / 2, HY, 0], [HW / 2, HY, 0], stroke_color=EDGE, stroke_width=2,
                           stroke_opacity=op))
                for txt, x in ((r"\text{toward us}", -HW / 2 + 0.9), (r"\text{away}", HW / 2 - 0.55)):
                    g.add(LBL[txt].copy().move_to([x, HY - 0.35, 0]).set_opacity(op))
                g.add(LBL["sig"].copy().move_to([0, HY - 0.38, 0]).set_opacity(op))
            op2 = smooth((t - 8.6) / 0.5) * (1 - smooth((t - 11.3) / 0.4))
            if op2 > 0.01:
                grow = smooth((t - 8.9) / 0.9)
                y1, y2 = -3.35, -4.55
                g.add(LBL["light"].copy().next_to([-HW / 2, y1 + 0.42, 0], RIGHT, buff=0).set_opacity(op2))
                g.add(Rectangle(width=0.05, height=0.34, stroke_width=0, fill_color=GOLD, fill_opacity=op2)
                      .move_to([-HW / 2 + 0.025, y1, 0]))
                g.add(LBL["motion"].copy().next_to([-HW / 2, y2 + 0.42, 0], RIGHT, buff=0).set_opacity(op2))
                g.add(Rectangle(width=max(0.05, HW * grow), height=0.34, stroke_width=0, fill_color=LAV,
                                fill_opacity=op2).move_to([-HW / 2 + HW * grow / 2, y2, 0]))
                g.add(LBL["x500"].copy().move_to([HW / 2 - 0.9, y1 + 0.05, 0]).set_opacity(op2 * smooth((t - 9.6) / 0.5)))
            return g

        LBL = {r"\text{toward us}": MathTex(r"\text{toward us}", color=BLUE).scale(0.6),
               r"\text{away}": MathTex(r"\text{away}", color=RED).scale(0.6),
               "sig": MathTex(r"v_{\text{los}}", color=MUTED).scale(0.7),
               "light": Text("mass you can see (light)", font="Inter", color=GOLD).scale(0.34),
               "motion": Text("mass the motion needs (virial)", font="Inter", color=LAV).scale(0.34),
               "x500": MathTex(r"\times\,500", color=GOLD).scale(1.5)}
        ticker = Mobject().add_updater(lambda m, dt: None)
        self.add(ticker, img, always_redraw(hist_f))

        def until(t):
            dt = t - self.renderer.time
            if dt > 1e-3:
                self.wait(dt)

        until(MATH_START)
        TOP, MID = 1.6, -0.95
        steps = [
            [eq(r"S=\sum_\alpha \mathbf r_\alpha\cdot\mathbf p_\alpha,\qquad \dot S = 2T+\sum_\alpha \mathbf r_\alpha\cdot\mathbf F_\alpha", 1.0)],
            [eq(r"\langle\dot S\rangle=0\ \ \Longrightarrow\ \ {{\langle T\rangle}} =-\frac12\Big\langle\sum_\alpha \mathbf r_\alpha\cdot\mathbf F_\alpha\Big\rangle", 1.0)],
            [eq(r"\mathbf r\cdot\nabla V=n\,V\ \ \Longrightarrow\ \ 2 {{\langle T\rangle}} =n {{\langle V\rangle}}", 1.1)],
            [eq(r"\text{gravity: } n=-1\qquad 2 {{\langle T\rangle}} =- {{\langle V\rangle}}", 1.15)],
            [eq(r"T\sim\tfrac12 {{M}} \, {{\langle v^2\rangle}},\qquad V\sim-\frac{G {{M}}^2}{R}", 1.1),
             eq(r"\Longrightarrow\quad {{M}} \sim\frac{R\, {{\langle v^2\rangle}} }{G}", 1.3)],
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
            MathTex(r"R\simeq10^{6}\,\text{ly},\quad \langle v^2\rangle\simeq5\times10^{15}\,\text{cm}^2/\text{s}^2",
                    color=MUTED).scale(0.75),
            eq(r"{{M}} \approx 4\times10^{13}\,M_\odot", 1.35),
            MathTex(r"\text{about } 500\times \text{ what the light accounts for}", color=MUTED).scale(0.72),
        ).arrange(DOWN, buff=0.32)
        panel = RoundedRectangle(corner_radius=0.08, width=box_eq.width + 1.1, height=box_eq.height + 1.0,
                                 fill_color=PANEL, fill_opacity=1, stroke_color=EDGE, stroke_width=2)
        tab_txt = Text("EXAMPLE · COMA, ZWICKY 1933", font="Inter", weight=MEDIUM, color=MUTED).scale(0.34)
        tab = RoundedRectangle(corner_radius=0.05, width=tab_txt.width + 0.4, height=tab_txt.height + 0.26,
                               fill_color="#000000", fill_opacity=1, stroke_color=EDGE, stroke_width=1.6)
        theorem = VGroup(panel, box_eq).move_to([0, MID - 0.2, 0])
        tab.move_to(panel.get_corner(UL) + RIGHT * (tab.width / 2 + 0.3))
        tab_txt.move_to(tab)
        frame_box = SurroundingRectangle(box_eq[1], color=GOLD, buff=0.16, stroke_width=2.4, corner_radius=0.04)
        self.play(FadeOut(prev, run_time=0.4), cur.animate(run_time=0.5).set_opacity(0))
        self.remove(cur)
        self.play(FadeIn(theorem, shift=UP * 0.25), FadeIn(tab), FadeIn(tab_txt), run_time=0.6)
        self.play(Create(frame_box), run_time=0.5)

        until(MATH_END - 0.45)
        self.play(FadeOut(VGroup(theorem, tab, tab_txt, frame_box), shift=DOWN * 0.4), run_time=0.4)
        until(30.0 - 1.0 / 30)
