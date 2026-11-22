"""Chladni figures — physics + math layer (Manim CE 0.22).

Typography (hook, mode readout, step kicker, end card) is NOT drawn here —
it lives in the HyperFrames layer, which composes this render underneath.
Layout reserves: top band y>+6.0 (HF hook / kicker), right column of the
math section (HF step kicker), bottom TikTok UI zone y<-5.6 kept empty.
"""
import numpy as np
from manim import *

from sand import Sand, render, plate_base, W
from timeline import FPS, MODES, STEPS, mode_at, MATH_START, MATH_END

config.pixel_width = 1080
config.pixel_height = 1920
config.frame_height = 16.0
config.frame_width = 9.0
config.frame_rate = FPS
config.background_color = "#000000"

INK = "#F2F2F2"
MUTED = "#9BA2AF"
PANEL = "#0E1014"
EDGE = "#4D5466"
BLUE = "#6FA8FF"
LAV = "#B7A6F5"
GOLD = "#E6C36A"

TEX = TexTemplate()
TEX.add_to_preamble(r"\usepackage{amsmath}\usepackage{amssymb}")

P = 840                       # plate raster
BIG_W, BIG_C = 7.4, np.array([0.0, 0.35, 0])
SMALL_W, SMALL_C = 3.5, np.array([-2.25, 4.75, 0])


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def plate_layout(t):
    """plate (width, center) as a function of global time"""
    if t < MATH_START:
        a = 0.0
    elif t < MATH_END:
        a = smooth((t - MATH_START) / 0.8)
    else:
        a = 1 - smooth((t - MATH_END) / 0.7)
    w = BIG_W + (SMALL_W - BIG_W) * a
    c = BIG_C + (SMALL_C - BIG_C) * a
    return w, c


def eq(s, scale=1.0):
    m = MathTex(s, tex_template=TEX, color=INK).scale(scale)
    for tok, col in ((r"n", LAV), (r"m", LAV), (r"k", BLUE), (r"\omega", GOLD)):
        try:
            m.set_color_by_tex(tok, col, substring=False)
        except Exception:
            pass
    return m


class Chladni(Scene):
    def construct(self):
        base = plate_base(P)
        yy, xx = (np.mgrid[0:P, 0:P] + 0.5) / P
        self.sand = Sand(N=70000, seed=11)
        self.cur_mode = None
        self.wmaps = {}
        self.last_frame = -1

        def rgba(arr):
            return np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])

        img = ImageMobject(rgba(render(self.sand, P, base)))
        img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        rim = RoundedRectangle(corner_radius=0.06, width=BIG_W, height=BIG_W,
                               stroke_color=EDGE, stroke_width=2.2)
        glow_rim = rim.copy().set_stroke(LAV, width=7, opacity=0.0)

        self.T = 0.0

        def plate_updater(m, dt):
            t = self.renderer.time
            frame = int(round(t * FPS))
            if frame != self.last_frame:
                steps = frame - self.last_frame if self.last_frame >= 0 else 1
                self.last_frame = frame
                md, a, b = mode_at(t)
                drive = 0.0
                if md is not None:
                    if md != self.cur_mode:
                        if self.cur_mode is not None:
                            self.sand.scatter(0.035)
                        self.cur_mode = md
                    drive = smooth((t - a) / 0.35) * (1 - smooth((t - (b - 0.25)) / 0.25)) \
                        if b < 29.9 else smooth((t - a) / 0.35)
                    for _ in range(min(steps, 3)):
                        self.sand.step(md, drive=drive, sub=4)
                frame_rgb = render(self.sand, P, base)
                # antinode shimmer: where the plate moves most, it glows faintly
                if md is not None and drive > 0.01:
                    if md not in self.wmaps:
                        self.wmaps[md] = np.abs(W(xx, yy, *md)) / 2.0
                    flick = 0.75 + 0.25 * np.sin(t * 2 * np.pi * 9)
                    g = (self.wmaps[md] ** 2 * 0.22 * drive * flick)[..., None]
                    lav = np.array([0xB7, 0xA6, 0xF5], float)
                    frame_rgb = np.clip(frame_rgb * (1 - g) + lav * g, 0, 255).astype(np.uint8)
                m.pixel_array = rgba(frame_rgb)
                glow_rim.set_stroke(opacity=0.55 * drive)
            w, c = plate_layout(t)
            m.set(width=w).move_to(c)
            rim.set(width=w).move_to(c)
            glow_rim.set(width=w).move_to(c)

        img.add_updater(plate_updater)
        self.add(glow_rim, img, rim)

        def until(t):
            dt = t - self.renderer.time
            if dt > 1e-3:
                self.wait(dt)

        # ── physics section: 0 → 12 s (HF draws the hook + readout) ─────────
        until(MATH_START)

        # ── math section ───────────────────────────────────────────────────
        TOP, MID = 1.75, -0.65
        steps = [
            [eq(r"D\,\nabla^4 w \;+\; \rho h\,\frac{\partial^2 w}{\partial t^2} \;=\; 0", 1.25)],
            [eq(r"w = W(x,y)\cos {{\omega}} t", 1.1),
             eq(r"\nabla^4 W = {{k}}^4 W,\qquad {{k}}^4=\frac{\rho h\,{{\omega}}^2}{D}", 1.1)],
            [eq(r"\big(\nabla^2-{{k}}^2\big)\big(\nabla^2+{{k}}^2\big)\,W=0", 1.1),
             eq(r"\Longrightarrow\quad \nabla^2 W = -{{k}}^2\, W", 1.2)],
            [eq(r"W_{ {{n}} \, {{m}} } = \cos\tfrac{ {{n}} \pi x}{L}\cos\tfrac{ {{m}} \pi y}{L}"
                r"\,\pm\,\cos\tfrac{ {{m}} \pi x}{L}\cos\tfrac{ {{n}} \pi y}{L}", 0.98),
             eq(r"{{k}}^2=\frac{\pi^2}{L^2}\big( {{n}}^2 + {{m}}^2 \big)", 1.15)],
            [eq(r"f_{ {{n}} \, {{m}} }=\frac{ {{\omega}} }{2\pi}"
                r"=\frac{\pi}{2L^2}\sqrt{\frac{D}{\rho h}}\;\big( {{n}}^2 + {{m}}^2 \big)", 1.08)],
        ]
        groups = []
        for parts in steps:
            g = VGroup(*parts).arrange(DOWN, buff=0.42)
            if g.width > 8.3:
                g.scale_to_fit_width(8.3)
            groups.append(g)

        prev = None
        cur = None
        for i, g in enumerate(groups):
            until(STEPS[i][0])
            g.move_to([0, MID, 0])
            anims = [Write(g, run_time=0.9)]
            if cur is not None:
                anims.append(cur.animate(run_time=0.6).scale(0.72).move_to([0, TOP + 0.35 * (len(cur) - 1), 0])
                             .set_opacity(0.38))
            if prev is not None:
                anims.append(FadeOut(prev, shift=UP * 0.6, run_time=0.5))
            self.play(*anims)
            prev, cur = cur, g

        # step 6: nodal lines — draw W=0 over the small plate, then the theorem box
        until(STEPS[5][0])
        n, m, s = MODES[3][2]
        w, c = plate_layout(STEPS[5][0] + 0.1)
        half = w / 2

        def to_scene(u, v):
            return np.array([c[0] - half + u * w, c[1] + half - v * w, 0])

        # trace nodal set via marching squares on a fine grid
        from skimage import measure  # noqa
        G = 400
        gy, gx = (np.mgrid[0:G, 0:G] + 0.5) / G
        Z = W(gx, gy, n, m, s)
        curves = VGroup()
        for cont in measure.find_contours(Z, 0.0):
            pts = [to_scene((p[1] + 0.5) / G, (p[0] + 0.5) / G) for p in cont[::3]]
            if len(pts) > 3:
                curves.add(VMobject(stroke_color=LAV, stroke_width=3.2).set_points_smoothly(pts))

        box_eq = VGroup(
            eq(r"W_{ {{n}} \, {{m}} }(x,y) = 0", 1.35),
            eq(r"f_{ {{n}} \, {{m}} } \;\propto\; {{n}}^2 + {{m}}^2", 1.2),
        ).arrange(DOWN, buff=0.45)
        panel = RoundedRectangle(corner_radius=0.08, width=box_eq.width + 1.1,
                                 height=box_eq.height + 1.0, fill_color=PANEL,
                                 fill_opacity=1, stroke_color=EDGE, stroke_width=2)
        tab_txt = Text("THEOREM · CHLADNI", font="Inter", weight=MEDIUM,
                       color=MUTED).scale(0.34)
        tab = RoundedRectangle(corner_radius=0.05, width=tab_txt.width + 0.4,
                               height=tab_txt.height + 0.26, fill_color="#000000",
                               fill_opacity=1, stroke_color=EDGE, stroke_width=1.6)
        theorem = VGroup(panel, box_eq)
        theorem.move_to([0, MID - 0.2, 0])
        tab.move_to(panel.get_corner(UL) + RIGHT * (tab.width / 2 + 0.3))
        tab_txt.move_to(tab)
        frame_box = SurroundingRectangle(box_eq[0], color=GOLD, buff=0.16,
                                         stroke_width=2.4, corner_radius=0.04)

        self.play(Create(curves, lag_ratio=0.08, run_time=0.9),
                  FadeOut(prev, run_time=0.4),
                  cur.animate(run_time=0.5).set_opacity(0.0))
        self.remove(cur)
        self.play(FadeIn(theorem, shift=UP * 0.25), FadeIn(tab), FadeIn(tab_txt),
                  run_time=0.6)
        self.play(Create(frame_box), run_time=0.5)

        # outro: everything math leaves, the plate comes back for a last mode
        until(MATH_END - 0.45)
        self.play(FadeOut(VGroup(theorem, tab, tab_txt, frame_box), shift=DOWN * 0.4),
                  FadeOut(curves), run_time=0.4)
        until(30.0 - 1.0 / FPS)
