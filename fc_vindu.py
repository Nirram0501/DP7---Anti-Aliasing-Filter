"""
Tillatt vindu for knekkfrekvensen fc som funksjon av filterorden n
(Butterworth lavpass), DP7.

Krav:
  1) fc >= 0.75 * fs/2
  2) demping ved fs/2 >= 10 dB

For Butterworth: |H(f)|^2 = 1 / (1 + (f/fc)^(2n)), som gir
  fc_max(n) = (fs/2) / (10^(A/10) - 1)^(1/(2n))      (A = 10 dB)

Figurer i ./figurer:
  fc_vindu_baand  – amplituderespons: bånd med alle tillatte filtre,
                    n = 4 (grått) sammenlignet med høyere orden (blått)
  fc_vindu_orden  – vinduet [fc_min, fc_max] som funksjon av n
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --- Innstillinger -----------------------------------------------------------
FS = 3200.0                 # punktprøvefrekvens [Hz]
F_NYQ = FS / 2
FC_MIN = 0.75 * F_NYQ       # krav 1
A_KRAV = 10.0               # krav 2: demping ved fs/2 [dB]
N_REF = 4                   # referanseorden (laveste som oppfyller kravene)
N_SAMMENLIGN = [5, 6, 7, 8] # ordener som sammenlignes med N_REF
FC_MAALT = 1208.0           # målt fc fra Bode-målingen [Hz] (None for å skjule)

FIG_MAPPE = Path(__file__).resolve().parent / "figurer"
FIG_MAPPE.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.35,
                     "figure.dpi": 120})

GRAA, BLAA, ROED = "0.55", "C0", "#d62728"


# --- Butterworth -------------------------------------------------------------
def H_dB(f, fc, n):
    return -10 * np.log10(1 + (f / fc) ** (2 * n))


def fc_max(n):
    return F_NYQ / (10 ** (A_KRAV / 10) - 1) ** (1 / (2 * n))


def tegn_forbudt(ax, ymin):
    """Røde felt: kurven kan ikke gå her."""
    ax.fill_between([0, FC_MIN], ymin, -3, color=ROED, alpha=0.10, lw=0)
    ax.fill_between([F_NYQ, 3000], -A_KRAV, 3, color=ROED, alpha=0.10, lw=0)
    ax.axhline(-3, color="0.4", ls="--", lw=0.8)
    ax.axhline(-A_KRAV, color="0.4", ls="--", lw=0.8)
    ax.axvline(FC_MIN, color=ROED, ls=":", lw=1)
    ax.axvline(F_NYQ, color=ROED, ls=":", lw=1)


def tegn_baand(ax, n, farge, alpha, label):
    f = np.linspace(1, 3000, 2000)
    lav, hoy = H_dB(f, FC_MIN, n), H_dB(f, fc_max(n), n)
    ax.fill_between(f, lav, hoy, color=farge, alpha=alpha, lw=0, label=label)
    ax.plot(f, lav, color=farge, lw=0.8)
    ax.plot(f, hoy, color=farge, lw=0.8)


# --- Figur 1: bånd i amplituderesponsen --------------------------------------
YMIN = -40
fig, akser = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
for ax, n in zip(akser.flat, N_SAMMENLIGN):
    tegn_forbudt(ax, YMIN)
    tegn_baand(ax, N_REF, GRAA, 0.35, f"$n$ = {N_REF}")
    tegn_baand(ax, n, BLAA, 0.30, f"$n$ = {n}")

    # «Pølsene»: vinduet for fc markert på -3 dB-linja
    for nn, farge, dy in [(N_REF, GRAA, 0), (n, BLAA, 0)]:
        ax.plot([FC_MIN, fc_max(nn)], [-3, -3], color=farge, lw=6,
                solid_capstyle="butt", alpha=0.9)

    b_ref, b = fc_max(N_REF) - FC_MIN, fc_max(n) - FC_MIN
    ax.set_title(rf"$n$ = {n}:  $f_c \in$ [{FC_MIN:.0f}, {fc_max(n):.0f}] Hz"
                 f"  ({b:.0f} Hz, vs. {b_ref:.0f} Hz for $n$ = {N_REF})",
                 fontsize=10)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(0, 3000)
    ax.set_ylim(YMIN, 2)

for ax in akser[1]:
    ax.set_xlabel("Frekvens [Hz]")
for ax in akser[:, 0]:
    ax.set_ylabel("Amplitude [dB]")
fig.suptitle("Tillatte Butterworth-filtre (bånd) for ulik orden.  "
             "Røde felt = forbudt", fontsize=12)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_MAPPE / f"fc_vindu_baand.{ext}", bbox_inches="tight")


# --- Figur 1b: zoom rundt kravene --------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
f = np.linspace(900, 2000, 1500)
ordener = [N_REF] + N_SAMMENLIGN
# Høyeste orden (bredest) tegnes først og lysest, så de smalere ligger oppå
farger = plt.cm.Blues(np.linspace(0.85, 0.3, len(N_SAMMENLIGN)))
tegn_forbudt(ax, -30)
for n, c in sorted(zip(N_SAMMENLIGN, farger), key=lambda x: -x[0]):
    ax.fill_between(f, H_dB(f, FC_MIN, n), H_dB(f, fc_max(n), n),
                    color=c, alpha=1.0, lw=0.6, ec="white", label=f"$n$ = {n}")
ax.fill_between(f, H_dB(f, FC_MIN, N_REF), H_dB(f, fc_max(N_REF), N_REF),
                color="0.35", lw=0, label=f"$n$ = {N_REF}")
ax.set_xlim(900, 2000)
ax.set_ylim(-30, 1)
ax.set_xlabel("Frekvens [Hz]")
ax.set_ylabel("Amplitude [dB]")
h, l = ax.get_legend_handles_labels()
rekkef = np.argsort([int(x.split("=")[1].strip(" $")) for x in l])
ax.legend([h[i] for i in rekkef], [l[i] for i in rekkef], fontsize=9, loc="lower left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_MAPPE / f"fc_vindu_zoom.{ext}", bbox_inches="tight")


# --- Figur 2: vinduet som funksjon av n --------------------------------------
n_akse = np.arange(3, max(N_SAMMENLIGN) + 3)
fig, ax = plt.subplots(figsize=(9, 4.8))
for n in n_akse:
    hi = fc_max(n)
    if hi < FC_MIN:
        ax.plot(hi, n, "x", color=ROED, ms=9, mew=2)
        ax.annotate(rf"umulig: $f_{{c,maks}}$ = {hi:.0f} Hz $<$ {FC_MIN:.0f} Hz",
                    (FC_MIN + 15, n), va="center", fontsize=9, color=ROED)
        continue
    farge = GRAA if n == N_REF else BLAA
    ax.barh(n, hi - FC_MIN, left=FC_MIN, height=0.55, color=farge, alpha=0.8)
    ax.annotate(f"{FC_MIN:.0f}–{hi:.0f} Hz  (bredde {hi - FC_MIN:.0f} Hz)",
                (hi, n), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9)

ax.axvline(FC_MIN, color=ROED, ls=":", lw=1.2,
           label=rf"Krav 1: $f_c \geq$ {FC_MIN:.0f} Hz")
ax.axvline(F_NYQ, color=ROED, ls="--", lw=1.2,
           label=rf"$f_s/2$ = {F_NYQ:.0f} Hz (grense når $n \to \infty$)")
if FC_MAALT:
    ax.axvline(FC_MAALT, color="C1", lw=1.5, label=rf"Målt $f_c \approx$ {FC_MAALT:.0f} Hz")
ax.set_yticks(n_akse)
ax.set_ylabel("Orden $n$")
ax.set_xlabel("Knekkfrekvens $f_c$ [Hz]")
ax.set_xlim(1080, F_NYQ + 50)
ax.set_title("Tillatt vindu for $f_c$ (Butterworth)")
ax.grid(axis="y", alpha=0)
ax.legend(fontsize=9, loc="lower right")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_MAPPE / f"fc_vindu_orden.{ext}", bbox_inches="tight")

# --- Utskrift ----------------------------------------------------------------
print(f"{'n':>3} {'fc_min':>8} {'fc_max':>8} {'bredde':>8}")
for n in n_akse:
    print(f"{n:3d} {FC_MIN:8.0f} {fc_max(n):8.1f} {fc_max(n) - FC_MIN:8.1f}")

plt.show()
