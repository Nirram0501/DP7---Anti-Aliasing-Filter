"""
Polplassering for Butterworth lavpassfilter (DP7).

Polene ligger jevnt fordelt på en halvsirkel med radius w0 i venstre
halvplan:
    s_k = w0 * exp(j*theta_k),  theta_k = pi/2 + (2k - 1)*pi/(2n),  k = 1..n
Vinkelavstanden mellom nabopoler er pi/n, og polene nærmest jw-aksen
ligger pi/(2n) fra aksen.

Hvert kompleks-konjugert polpar svarer til ett 2. ordens trinn med
    Q = 1 / (2 cos(psi)),
der psi er vinkelen fra den negative reelle aksen.

Figurer i ./figurer:
  poler_3par      – 3 polpar (n = 6), med vinkler og Q for hvert par
  poler_antall    – 1, 2, 3 og 4 polpar (n = 2, 4, 6, 8)
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

FIG_MAPPE = Path(__file__).resolve().parent / "figurer"
FIG_MAPPE.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 120})

PAR_FARGER = ["C0", "C1", "C2", "C3", "C4", "C5"]


def butterworth_poler(n):
    """Normerte poler (w0 = 1), sortert fra jw-aksen og innover."""
    k = np.arange(1, n + 1)
    theta = np.pi / 2 + (2 * k - 1) * np.pi / (2 * n)
    return np.exp(1j * theta)


def polpar(n):
    """Liste med (pol i øvre halvplan, psi [rad], Q) for hvert polpar."""
    p = butterworth_poler(n)
    oevre = sorted(p[p.imag > 1e-12], key=lambda z: -z.imag)  # nærmest jw først
    par = []
    for z in oevre:
        psi = np.arctan2(z.imag, -z.real)
        par.append((z, psi, 1 / (2 * np.cos(psi))))
    return par


def tegn_grunnlag(ax, akser_tekst=True):
    t = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(t), np.sin(t), color="0.8", lw=0.8, ls="--")
    t = np.linspace(np.pi / 2, 3 * np.pi / 2, 300)
    ax.plot(np.cos(t), np.sin(t), color="0.35", lw=1.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 0.45)
    ax.set_ylim(-1.25, 1.25)
    if akser_tekst:
        ax.set_xlabel(r"$\sigma / \omega_0$")
        ax.set_ylabel(r"$j\omega / \omega_0$")
    ax.grid(alpha=0.3)


def tegn_poler(ax, n, ms=10, med_linjer=True):
    for i, (z, psi, Q) in enumerate(polpar(n)):
        c = PAR_FARGER[i % len(PAR_FARGER)]
        for zz in (z, np.conj(z)):
            if med_linjer:
                ax.plot([0, zz.real], [0, zz.imag], color=c, lw=0.9, alpha=0.6)
            ax.plot(zz.real, zz.imag, "x", color=c, ms=ms, mew=2.4)
    if n % 2:  # odde orden: reell pol i -w0
        ax.plot(-1, 0, "x", color="k", ms=ms, mew=2.4)


# --- Figur 1: 3 polpar (n = 6) -----------------------------------------------
n = 6
fig, ax = plt.subplots(figsize=(7, 7.5))
tegn_grunnlag(ax)
tegn_poler(ax, n, ms=13)

# Vinkler: pi/(2n) fra jw-aksen, deretter pi/n mellom polene
vinkler = [np.pi / 2] + [np.angle(z) for z, _, _ in polpar(n)]
for a, b, r, tekst in [
    (vinkler[0], vinkler[1], 0.30, rf"${180 / (2 * n):.0f}°$"),
    (vinkler[1], vinkler[2], 0.45, rf"${180 / n:.0f}°$"),
    (vinkler[2], vinkler[3], 0.60, rf"${180 / n:.0f}°$"),
]:
    ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=np.degrees(a),
                     theta2=np.degrees(b), color="0.3", lw=1))
    m = (a + b) / 2
    ax.text((r + 0.08) * np.cos(m), (r + 0.08) * np.sin(m), tekst,
            ha="center", va="center", fontsize=10)

# Merk radius w0
ax.annotate("", xy=(-np.cos(np.pi / 4) * 1, -np.sin(np.pi / 4) * 1),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="0.3", lw=1.2))
ax.text(-0.42, -0.30, r"$\omega_0$", fontsize=13, color="0.2")

# Navn på polparene (radielt utenfor sirkelen) og fargekodet tabell
for i, (z, psi, Q) in enumerate(polpar(n)):
    c = PAR_FARGER[i]
    for zz in (z, np.conj(z)):
        r = 1.13
        ax.text(r * zz.real, r * zz.imag, f"{i + 1}", color=c, fontsize=11,
                fontweight="bold", ha="center", va="center")
    ax.text(0.08, -0.80 - 0.13 * i,
            rf"Par {i + 1}:  $\psi$ = {np.degrees(psi):.0f}°,  "
            rf"$Q = \frac{{1}}{{2\cos\psi}}$ = {Q:.3f}",
            color=c, fontsize=10, va="center", fontweight="bold")
ax.set_xlim(-1.3, 1.35)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_MAPPE / f"poler_3par.{ext}", bbox_inches="tight")


# --- Figur 2: ulikt antall polpar --------------------------------------------
antall_par = [1, 2, 3, 4]
fig, akser = plt.subplots(1, len(antall_par), figsize=(3.4 * len(antall_par), 4.6),
                          sharey=True)
for ax, m in zip(akser, antall_par):
    n = 2 * m
    tegn_grunnlag(ax, akser_tekst=False)
    tegn_poler(ax, n, ms=10)
    ax.set_title(f"{m} polpar  ($n$ = {n})", fontsize=11)
    # Radius w0: merk der halvsirkelen møter Im-aksen
    for y, tekst in [(1, r"$\omega_0$"), (-1, r"$-\omega_0$")]:
        ax.plot(0, y, "o", color="0.2", ms=4)
        ax.text(0.05, y, tekst, fontsize=12, va="center", ha="left")
    ax.set_xlabel("Re")
akser[0].set_ylabel("Im")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_MAPPE / f"poler_antall.{ext}", bbox_inches="tight")

# --- Utskrift ----------------------------------------------------------------
for m in antall_par:
    print(f"n = {2 * m}:", ", ".join(
        f"psi={np.degrees(psi):.1f}° Q={Q:.3f}" for _, psi, Q in polpar(2 * m)))

plt.show()
