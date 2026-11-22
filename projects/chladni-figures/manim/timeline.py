"""Single source of truth for timing — read by the Manim scene, the audio synth
and (exported to JSON) by the HyperFrames typography layer."""
import json, pathlib

FPS = 30
DURATION = 30.0
F0 = 23.0  # Hz per unit of (n^2+m^2) — illustrative steel-plate scale

# (start, end, (n, m, sign)) — sand is driven in this window
MODES = [
    (0.45, 3.60, (1, 3, +1)),
    (3.60, 6.20, (2, 5, +1)),
    (6.20, 8.80, (3, 6, +1)),
    (8.80, 11.70, (4, 7, +1)),
    (26.40, 30.0, (5, 8, -1)),
]

# math steps (Manim owns the equations, HF owns the step kicker text)
STEPS = [
    (12.80, "plate equation", "Kirchhoff–Love"),
    (14.90, "standing wave", "separate time"),
    (17.00, "factor the operator", "Helmholtz"),
    (19.10, "try a mode", "free edges"),
    (21.20, "the pitch", "Chladni’s law"),
    (23.30, "where the sand stops", "nodal lines"),
]
MATH_START = 12.0
MATH_END = 26.2


def freq(mode):
    n, m, _ = mode
    return F0 * (n * n + m * m)


def mode_at(t):
    for a, b, md in MODES:
        if a <= t < b:
            return md, a, b
    return None, None, None


def export(path):
    data = {
        "fps": FPS, "duration": DURATION,
        "modes": [{"start": a, "end": b, "n": md[0], "m": md[1],
                   "sign": md[2], "f": round(freq(md))} for a, b, md in MODES],
        "steps": [{"t": t, "title": a, "tag": b} for t, a, b in STEPS],
        "math_start": MATH_START, "math_end": MATH_END,
    }
    pathlib.Path(path).write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    export(pathlib.Path(__file__).parents[1] / "hyperframes" / "timeline.json")
    for a, b, md in MODES:
        print(a, b, md, round(freq(md)), "Hz")
