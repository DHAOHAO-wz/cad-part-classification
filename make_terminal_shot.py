"""Render the captured terminal output to a terminal-style screenshot PNG."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "terminal_output.txt")
OUT = os.path.join(HERE, "assets", "terminal_run.png")

with open(SRC, encoding="utf-8") as f:
    lines = [ln.rstrip() for ln in f.readlines()]

# drop a single leading/trailing blank line for tighter layout
while lines and lines[0] == "":
    lines.pop(0)
while lines and lines[-1] == "":
    lines.pop()

font_size = 13
line_h = 22
pad_x = 24
title_h = 40
pad_y = 18
fig_w = 1000
fig_h = title_h + pad_y * 2 + line_h * len(lines)

fig = plt.figure(figsize=(fig_w / 100, fig_h / 100), dpi=150)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, fig_w)
ax.set_ylim(fig_h, 0)
ax.axis("off")

# window background + title bar
ax.add_patch(plt.Rectangle((0, 0), fig_w, fig_h, facecolor="#18263A", edgecolor="none"))
ax.add_patch(plt.Rectangle((0, 0), fig_w, title_h, facecolor="#1E3C78", edgecolor="none"))

# window dots
for i, c in enumerate(["#5385BD", "#519D78", "#9BC7DF"]):
    ax.add_patch(plt.Circle((28 + i * 22, title_h / 2), 6, facecolor=c, edgecolor="none"))
ax.text(fig_w / 2, title_h / 2, "bash — python3 train.py", ha="center", va="center",
        fontsize=11, color="#9BC7DF", family="DejaVu Sans Mono")

y = title_h + pad_y
for ln in lines:
    if ln.startswith("$"):
        color = "#8BCF8B"
    elif ln.startswith("["):
        color = "#9BC7DF"
    elif ln.strip() == "":
        color = "#DCE4EE"
    else:
        color = "#DCE4EE"
    ax.text(pad_x, y, ln, ha="left", va="center", fontsize=font_size,
            color=color, family="DejaVu Sans Mono")
    y += line_h

fig.savefig(OUT, dpi=150, facecolor="#18263A")
plt.close(fig)
print("saved", OUT, "lines:", len(lines))
