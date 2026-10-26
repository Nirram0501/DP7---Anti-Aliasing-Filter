"""
Hvordan to polpar sammen gir Butterworth-respons (n = 4), DP7.

Hvert polpar gir et 2. ordens lavpasstrinn
    H_i(s) = w0^2 / (s^2 + (w0/Q_i) s + w0^2)
  - Polpar nær Im-aksen  -> høy Q  -> topp rundt f0
  - Polpar nær Re-aksen  -> lav Q  -> demper tidlig
Produktet H1*H2 blir maksimalt flat (Butterworth).

Figur i ./figurer:  polpar_bidrag
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FIG_MAPPE = Path(__file__).resolve().parent / "figurer"
FIG_MAPPE.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 120})

N = 4
FARGE_1, FARGE_2, FARGE_TOT = "C3", "C0", "k"

# Butterworth-poler (w0 = 1), øvre halvplan, nærmest Im-aksen først
k = np.arange(1, N + 1)
poler = np.exp(1j * (np.pi / 2 + (2 * k - 1) * np.pi / (2 * N)))
oevre = sorted(poler[poler.imag > 0], key=lambda z: -z.imag)
Q = [1 / (2 * np.cos(np.arctan2(z.imag, -z.real))) for z in oevre]


def H2_dB(f, q):
    """2. ordens lavpass, f normert til f0."""
    s = 1j * f
    return 20 * np.log10(np.abs(1 / (s**2 + s / q + 1)))


fig, (ax_s, ax_H) = plt.subplots(1, 2, figsize=(12, 5.2),
                                 gridspec_kw={"width_ratios": [1, 1.6]})

# --- Re/Im-planet ------------------------------------------------------------
t = np.linspace(0, 2 * np.pi, 400)
ax_s.plot(np.cos(t), np.sin(t), color="0.8", lw=0.8, ls="--")
t = np.linspace(np.pi / 2, 3 * np.pi / 2, 300)
ax_s.plot(np.cos(t), np.sin(t), color="0.35", lw=1.5)
ax_s.axhline(0, color="k", lw=0.8)
ax_s.axvline(0, color="k", lw=0.8)

for z, q, c, navn in [(oevre[0], Q[0], FARGE_1, "Polpar 1"),
                      (oevre[1], Q[1], FARGE_2, "Polpar 2")]:
    for zz in (z, np.conj(z)):
        ax_s.plot([0, zz.real], [0, zz.imag], color=c, lw=1, alpha=0.6)
        ax_s.plot(zz.real, zz.imag, "x", color=c, ms=13, mew=2.6)
    ax_s.annotate(f"{navn}\n$Q$ = {q:.2f}", (z.real, z.imag),
                  xytext=(-14, 14), textcoords="offset points", ha="right",
                  color=c, fontsize=10, fontweight="bold")

for y, tekst in [(1, r"$\omega_0$"), (-1, r"$-\omega_0$")]:
    ax_s.plot(0, y, "o", color="0.2", ms=4)
    ax_s.text(0.05, y, tekst, fontsize=12, va="center", ha="left")

ax_s.set_aspect("equal")
ax_s.set_xlim(-1.45, 0.45)
ax_s.set_ylim(-1.3, 1.3)
ax_s.set_xlabel("Re")
ax_s.set_ylabel("Im")
ax_s.grid(alpha=0.3)

# --- Amplituderespons --------------------------------------------------------
f = np.logspace(-1, 1, 1000)
H1, H2 = H2_dB(f, Q[0]), H2_dB(f, Q[1])
Htot = H1 + H2  # produkt -> sum i dB

ax_H.semilogx(f, H1, color=FARGE_1, lw=1.8,
              label=rf"Polpar 1 ($Q$ = {Q[0]:.2f}): topp ved $f_0$")
ax_H.semilogx(f, H2, color=FARGE_2, lw=1.8,
              label=rf"Polpar 2 ($Q$ = {Q[1]:.2f}): demper tidlig")
ax_H.semilogx(f, Htot, color=FARGE_TOT, lw=2.6,
              label=r"Produkt $H_1 \cdot H_2$: flat Butterworth ($n$ = 4)")

ax_H.axhline(0, color="0.5", lw=0.8)
ax_H.axhline(-3, color="0.5", lw=0.8, ls="--")
ax_H.axvline(1, color="0.5", lw=0.8, ls=":")
ax_H.text(1.04, -36, "$f_0$", fontsize=12)
ax_H.text(0.105, -2.6, "$-3$ dB", fontsize=9, color="0.4")

# Verdier ved f0: +2,3 dB  +  (-5,3 dB)  =  -3 dB
v1 = H2_dB(np.array([1.0]), Q[0])[0]
v2 = H2_dB(np.array([1.0]), Q[1])[0]
for y, c in [(v1, FARGE_1), (v2, FARGE_2), (v1 + v2, FARGE_TOT)]:
    ax_H.plot(1, y, "o", color=c, ms=6, zorder=5)
ax_H.text(9.3, 6.5,
          rf"Ved $f_0$:  ${v1:+.1f}$ dB $+$ $({v2:+.1f}$ dB$)$ $=$ ${v1 + v2:.1f}$ dB",
          ha="right", va="top", fontsize=10,
          bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax_H.set_xlim(0.1, 10)
ax_H.set_ylim(-40, 8)
ax_H.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
ax_H.set_xticklabels(["0,1", "0,2", "0,5", "1", "2", "5", "10"])
ax_H.set_xlabel("$f / f_0$")
ax_H.set_ylabel("Amplitude [dB]")
ax_H.grid(alpha=0.3, which="both")
ax_H.legend(fontsize=9.5, loc="lower left")

fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_MAPPE / f"polpar_bidrag.{ext}", bbox_inches="tight")

print("Q:", [round(q, 3) for q in Q])
print("Ved f0: H1 = %.2f dB, H2 = %.2f dB, sum = %.2f dB" %
      (H2_dB(np.array([1.0]), Q[0])[0], H2_dB(np.array([1.0]), Q[1])[0],
       H2_dB(np.array([1.0]), Q[0])[0] + H2_dB(np.array([1.0]), Q[1])[0]))
plt.show()
