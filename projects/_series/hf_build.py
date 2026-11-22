"""Build the HyperFrames typography layer for a 'Physics in 30s' clip.

Series identity (tag, step kicker, end tag, palette) is shared; the content
(hook, title, steps, outro) comes from each clip's config. Usage:
    python hf_build.py <clip_config.json> <out_index.html>
"""
import json, sys, html, pathlib

CSS = pathlib.Path(__file__).with_name("style.css").read_text()


def e(s):
    return html.escape(s, quote=False)


def lines_html(prefix, lines, cls):
    out = []
    for i, ln in enumerate(lines):
        txt = e(ln["text"])
        if ln.get("gold"):
            txt = txt.replace(e(ln["gold"]), f'<span class="gold">{e(ln["gold"])}</span>')
        out.append(f'<span class="line"><span class="in {cls}" id="{prefix}{i}">{txt}</span></span>')
    return "\n          ".join(out)


def build(cfg):
    D = cfg.get("duration", 30)
    hook, title, why, outro = cfg["hook"], cfg["title"], cfg["why"], cfg["outro"]
    steps = cfg["steps"]
    k_start = steps[0]["t"] - 0.2
    media = [f'<video id="bg" class="clip fill" src="assets/{cfg["video"]}" data-start="0" '
             f'data-duration="{D}" data-track-index="0" muted playsinline></video>']
    if cfg.get("audio"):
        media.append(f'<audio id="snd" src="assets/{cfg["audio"]}" data-start="0" data-duration="{D}" '
                     f'data-track-index="10" data-volume="1"></audio>')
    ksteps = "\n          ".join(
        f'<div class="kstep" id="k{i}"><div class="cap knum">Step {i+1} / {len(steps)}</div>'
        f'<div class="ktitle">{e(s["title"])}</div><div class="ktag">{e(s["tag"])}</div></div>'
        for i, s in enumerate(steps))
    hook_end = hook["end"]
    body = f"""
    <div id="root" data-composition-id="main" data-width="1080" data-height="1920" data-duration="{D}">
      {chr(10).join('      ' + m for m in media).strip()}

      <section id="seriesL" class="clip layer" data-start="0" data-duration="{D}" data-track-index="1">
        <div id="series"><i class="dot"></i><span class="cap">Physics in 30s · {e(cfg["num"])}</span></div>
      </section>

      <section id="hook" class="clip layer" data-start="0" data-duration="{hook_end + 0.2}" data-track-index="2">
        <div class="band">
          {lines_html("hk", hook["lines"], "h1")}
        </div>
      </section>

      <section id="title" class="clip layer" data-start="{hook_end - 0.2}" data-duration="{title["end"] - hook_end + 0.4}" data-track-index="3">
        <div class="band">
          <span class="line"><span class="in h1x" id="tt1">{e(title["text"])}</span></span>
          <div class="sub" id="tt2">{e(title["sub"])}</div>
          {f'<div class="cap" id="tt3" style="margin-top:26px;color:var(--lav);text-transform:none;letter-spacing:0.04em;font-size:26px">{e(title["cap"])}</div>' if title.get("cap") else ''}
        </div>
      </section>

      <section id="why" class="clip layer" data-start="{why["t"]}" data-duration="1.5" data-track-index="4">
        <div class="band">
          {lines_html("wy", why["lines"], "h1x")}
        </div>
      </section>

      <section id="kicker" class="clip layer" data-start="{k_start}" data-duration="{cfg["math_end"] - k_start + 0.1}" data-track-index="6">
        <div id="kick">
          {ksteps}
          <div class="kbar"><i id="kprog"></i></div>
          <div id="kfoot">{cfg["kfoot"]}</div>
        </div>
      </section>

      <section id="outro" class="clip layer" data-start="{outro["t"] - 0.2}" data-duration="{D - outro["t"] + 0.2}" data-track-index="7">
        <div class="band">
          {lines_html("ot", outro["lines"], "h1x")}
        </div>
        <div id="endtag" style="top:{outro.get("tag_top", 1430)}px">
          <span class="cap" id="et1">{e(outro["left"])}</span>
          <span class="next" id="et2">next → <b>{e(outro["next"])}</b></span>
        </div>
      </section>
      {cfg.get("extra_html", "")}
    </div>"""
    n_hook = len(hook["lines"])
    hook_ts = hook["times"]
    js = f"""
      window.__timelines = window.__timelines || {{}};
      const tl = gsap.timeline({{ paused: true }});
      const E = "power3.out", EI = "power2.in";
      function lineIn(sel, t, d = 0.6) {{ tl.fromTo(sel, {{ yPercent: 110 }}, {{ yPercent: 0, duration: d, ease: E }}, t); }}
      function lineOut(sel, t, d = 0.35) {{ tl.to(sel, {{ yPercent: -110, duration: d, ease: EI }}, t); }}
      tl.fromTo("#series", {{ opacity: 0, x: -16 }}, {{ opacity: 1, x: 0, duration: 0.6, ease: E }}, 0.1);
      {"".join(f'lineIn("#hk{i}", {hook_ts[i]});' for i in range(n_hook))}
      lineOut([{",".join(f'"#hk{i}"' for i in range(n_hook))}], {hook_end - 0.45});
      lineIn("#tt1", {hook_end + 0.05}, 0.7);
      tl.fromTo("#tt2", {{ opacity: 0, y: 18 }}, {{ opacity: 1, y: 0, duration: 0.6, ease: E }}, {hook_end + 0.4});
      if (document.getElementById("tt3")) tl.fromTo("#tt3", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.6 }}, {hook_end + 0.9});
      lineOut("#tt1", {title["end"] - 0.35});
      tl.to(["#tt2", "#tt3"].filter(s => document.querySelector(s)), {{ opacity: 0, y: -12, duration: 0.3, ease: EI }}, {title["end"] - 0.35});
      {"".join(f'lineIn("#wy{i}", {why["t"] + 0.05 + 0.11*i}, 0.5);' for i in range(len(why["lines"])))}
      lineOut([{",".join(f'"#wy{i}"' for i in range(len(why["lines"])))}], {why["t"] + 1.05}, 0.35);
      const steps = {json.dumps([s["t"] for s in steps])};
      steps.forEach((t, i) => tl.set(`#k${{i}}`, {{ opacity: 0 }}, {k_start}));
      steps.forEach((t, i) => {{
        tl.fromTo(`#k${{i}}`, {{ opacity: 0, y: 26 }}, {{ opacity: 1, y: 0, duration: 0.5, ease: E }}, t - 0.05);
        if (i > 0) tl.to(`#k${{i-1}}`, {{ opacity: 0, y: -20, duration: 0.22, ease: EI }}, t - 0.3);
      }});
      tl.fromTo("#kprog", {{ scaleX: 0 }}, {{ scaleX: 1, duration: {cfg["math_end"] - 0.2 - steps[0]["t"]:.2f}, ease: "none" }}, {steps[0]["t"]});
      tl.fromTo("#kfoot", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.5 }}, {steps[0]["t"] + 0.2});
      tl.to("#kick", {{ opacity: 0, duration: 0.35 }}, {cfg["math_end"] - 0.3});
      {"".join(f'lineIn("#ot{i}", {outro["t"] + 0.25 + 0.17*i}, 0.7);' for i in range(len(outro["lines"])))}
      tl.fromTo("#et1", {{ opacity: 0 }}, {{ opacity: 1, duration: 0.5 }}, {outro["t"] + 0.4});
      tl.fromTo("#et2", {{ opacity: 0, x: 20 }}, {{ opacity: 1, x: 0, duration: 0.6, ease: E }}, {D - 2.0});
      {cfg.get("extra_js", "")}
      window.__timelines["main"] = tl;"""
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1080, height=1920" />
    <title>{e(title["text"])}</title>
    <script src="vendor/gsap.min.js"></script>
{CSS}
  </head>
  <body>{body}
    <script>{js}
    </script>
  </body>
</html>
"""


if __name__ == "__main__":
    cfg = json.loads(pathlib.Path(sys.argv[1]).read_text())
    pathlib.Path(sys.argv[2]).write_text(build(cfg))
    print("wrote", sys.argv[2])
