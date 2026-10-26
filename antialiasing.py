"""
Demonstrasjon av anti-aliasing (DP7).

Leser scope-målingene i ../CSV (scope<frekvens>hz.csv), der
  kanal 1 = filterets inngang (ufiltrert)
  kanal 2 = filterets utgang (filtrert)
begge punktprøvd med scopet i Decimate-modus.

Lager tre figurer i ./figurer:
  1. antialiasing_tid      – punktprøvde signaler i tid, med og uten filter
  2. antialiasing_fft      – spekter av de punktprøvde signalene
  3. antialiasing_demping  – målt demping av alias sammenlignet med Bode-målingen
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Innstillinger -----------------------------------------------------------
FS_SPEK = 3200.0          # punktprøvefrekvens fra spesifikasjonen [Hz]
TIDSVINDU = 0.015         # hvor mye av tidssignalet som vises [s]

HER = Path(__file__).resolve().parent
CSV_MAPPE = HER.parent / "CSV"
FIG_MAPPE = HER / "figurer"
FIG_MAPPE.mkdir(exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.4,
                     "figure.dpi": 120})
FARGE_UTEN, FARGE_MED = "C3", "C0"


# --- Innlesing ---------------------------------------------------------------
def les_scope(fil: Path):
    """Returnerer (fs, DataFrame) fra en WaveForms scope-CSV."""
    tekst = fil.read_text(errors="ignore")
    m = re.search(r"#Sample rate:\s*([\d.]+)\s*Hz", tekst)
    df = pd.read_csv(fil, comment="#")
    df.columns = ["t", "inn", "ut"]
    fs = float(m.group(1)) if m else 1 / np.diff(df["t"]).mean()
    return fs, df


def frekvens_fra_navn(fil: Path) -> float:
    """scope2.5khz.csv -> 2500, scope500hz.csv -> 500"""
    m = re.search(r"scope([\d.]+)\s*(k?)hz", fil.name, re.IGNORECASE)
    return float(m.group(1)) * (1000 if m.group(2).lower() == "k" else 1)


def alias(f, fs):
    """Frekvensen f havner på etter punktprøving med fs."""
    return abs(f - fs * round(f / fs))


def spekter(x, fs):
    """Amplitudespekter (V) med Blackman-vindu."""
    x = x - x.mean()
    w = np.blackman(len(x))
    X = np.abs(np.fft.rfft(x * w)) * 2 / w.sum()
    return np.fft.rfftfreq(len(x), 1 / fs), X


def amplitude_ved(x, fs, f0, bredde=5.0):
    f, X = spekter(x, fs)
    return X[np.abs(f - f0) <= bredde].max()


# --- Les alle målinger -------------------------------------------------------
maalinger = []
for fil in CSV_MAPPE.glob("scope*.csv"):
    fs, df = les_scope(fil)
    f_inn = frekvens_fra_navn(fil)
    maalinger.append(dict(f_inn=f_inn, fs=fs, df=df, f_alias=alias(f_inn, fs)))
maalinger.sort(key=lambda m: m["f_inn"])

fs_maalt = maalinger[0]["fs"]
print(f"Samplingsfrekvens i scope-målingene: {fs_maalt:.0f} Hz "
      f"(fs/2 = {fs_maalt / 2:.0f} Hz)")


def fmt_f(f):
    return f"{f / 1000:g} kHz" if f >= 1000 else f"{f:g} Hz"


# --- Figur 1: tidsplott ------------------------------------------------------
n = len(maalinger)
fig, akser = plt.subplots(n, 1, figsize=(8, 2.3 * n), sharex=True)
for ax, m in zip(np.atleast_1d(akser), maalinger):
    df = m["df"]
    t0 = df["t"].iloc[0]
    sel = df["t"] - t0 <= TIDSVINDU
    t_ms = (df["t"][sel] - t0) * 1e3
    ax.plot(t_ms, df["inn"][sel], ".-", color=FARGE_UTEN, lw=0.9, ms=4,
            label="Uten filter (inngang)")
    ax.plot(t_ms, df["ut"][sel], ".-", color=FARGE_MED, lw=0.9, ms=4,
            label="Med filter (utgang)")
    tittel = f"$f_{{inn}}$ = {fmt_f(m['f_inn'])}"
    if m["f_inn"] > m["fs"] / 2:
        tittel += rf"  $\rightarrow$  alias ved {m['f_alias']:.0f} Hz"
    ax.set_title(tittel, fontsize=10.5, loc="left")
    ax.set_ylabel("Spenning [V]")
akser[0].legend(fontsize=9, loc="upper right", ncol=2)
akser[-1].set_xlabel("Tid [ms]")
fig.suptitle(f"Punktprøvde signaler ($f_s$ = {fs_maalt:.0f} Hz)")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_MAPPE / f"antialiasing_tid.{ext}", bbox_inches="tight")


# --- Figur 2: spekter --------------------------------------------------------
fig, akser = plt.subplots(n, 1, figsize=(8, 2.3 * n), sharex=True)
resultater = []
for ax, m in zip(np.atleast_1d(akser), maalinger):
    df, fs, fa = m["df"], m["fs"], m["f_alias"]
    for kol, farge, navn in [("inn", FARGE_UTEN, "Uten filter"),
                             ("ut", FARGE_MED, "Med filter")]:
        f, X = spekter(df[kol].to_numpy(), fs)
        ax.plot(f, 20 * np.log10(X + 1e-9), color=farge, lw=1, label=navn)

    A_uten = amplitude_ved(df["inn"].to_numpy(), fs, fa)
    A_med = amplitude_ved(df["ut"].to_numpy(), fs, fa)
    demping = 20 * np.log10(A_med / A_uten)
    resultater.append((m["f_inn"], fa, A_uten, A_med, demping))

    ax.axvline(fa, color="gray", ls=":", lw=1)
    ax.annotate(f"{demping:.1f} dB", xy=(fa, 20 * np.log10(A_med)),
                xytext=(8, 0), textcoords="offset points", fontsize=9,
                va="center")
    tittel = f"$f_{{inn}}$ = {fmt_f(m['f_inn'])}"
    if m["f_inn"] > fs / 2:
        tittel += rf"  $\rightarrow$  alias ved {fa:.0f} Hz"
    ax.set_title(tittel, fontsize=10.5, loc="left")
    ax.set_ylabel("Amplitude [dBV]")
    ax.set_ylim(-110, 5)
akser[0].legend(fontsize=9, loc="upper right")
akser[-1].set_xlabel("Frekvens [Hz]")
akser[-1].set_xlim(0, fs_maalt / 2)
fig.suptitle(f"Spekter av punktprøvde signaler ($f_s$ = {fs_maalt:.0f} Hz)")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_MAPPE / f"antialiasing_fft.{ext}", bbox_inches="tight")


# --- Figur 3: målt demping vs. Bode -----------------------------------------
bode = pd.read_csv(CSV_MAPPE / "bodeplott100-6k.csv", comment="#")
bode.columns = ["f", "mag1", "mag2", "fase2"]

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.axvspan(fs_maalt / 2, bode["f"].max(), color="C3", alpha=0.08,
           label=r"Over $f_s/2$: gir aliasing")
ax.semilogx(bode["f"], bode["mag2"], color="C0", lw=1.8,
            label="Bode-måling $|H(f)|$")
ax.semilogx([r[0] for r in resultater], [r[4] for r in resultater], "o",
            color="C1", ms=8, mec="k", zorder=5,
            label="Fra punktprøving (utgang / inngang)")
for f_inn, fa, *_, d in resultater:
    ax.annotate(f"{d:.1f} dB", (f_inn, d), xytext=(7, 5),
                textcoords="offset points", fontsize=9)
ax.axvline(FS_SPEK / 2, color="C3", ls=":", lw=1.3,
           label=rf"$f_s/2$ fra spesifikasjon = {FS_SPEK / 2:.0f} Hz")
if abs(fs_maalt - FS_SPEK) > 10:
    ax.axvline(fs_maalt / 2, color="C4", ls=":", lw=1.3,
               label=rf"$f_s/2$ i scope-målingen = {fs_maalt / 2:.0f} Hz")
ax.axhline(-10, color="gray", ls="--", lw=0.8, label="Krav: 10 dB demping")
ax.set_xlabel("Inngangsfrekvens [Hz]")
ax.set_ylabel("Forsterkning [dB]")
ax.set_title("Demping av signaler som ellers ville gitt aliasing")
ax.set_xlim(bode["f"].min(), bode["f"].max())
ax.legend(fontsize=8.5, loc="lower left")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(FIG_MAPPE / f"antialiasing_demping.{ext}", bbox_inches="tight")

# --- Oppsummering ------------------------------------------------------------
print(f"\n{'f_inn':>8} {'alias':>8} {'A uten':>9} {'A med':>9} {'demping':>9}")
for f_inn, fa, a1, a2, d in resultater:
    print(f"{f_inn:8.0f} {fa:8.0f} {a1:9.4f} {a2:9.4f} {d:8.1f} dB")

plt.show()
