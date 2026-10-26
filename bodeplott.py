"""
Bodeplott av målingene fra Digilent WaveForms Network Analyzer (DP7).

Leser CSV-filene i ../CSV, plotter amplitude- og faserespons (kanal 2,
relativ til kanal 1), og markerer kravene fra oppgaven:
  - knekkfrekvens fc (-3 dB fra høyeste nivå) skal være >= 0.75 * fs/2
  - dempingen ved fs/2 skal være minst 10 dB

Figurene lagres i ./figurer som PDF og PNG.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# --- Innstillinger -----------------------------------------------------------
FS = 3200.0                 # punktprøvefrekvens [Hz] (Tandberg: 3,2 kHz)
F_NYQ = FS / 2              # fs/2
FC_KRAV = 0.75 * F_NYQ      # minste tillatte knekkfrekvens
DEMPING_KRAV = 10.0         # minste demping ved fs/2 [dB]

HER = Path(__file__).resolve().parent
CSV_MAPPE = HER.parent / "CSV"
FIG_MAPPE = HER / "figurer"
FIG_MAPPE.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "figure.dpi": 120,
})


# --- Hjelpefunksjoner --------------------------------------------------------
def les_bode(fil: Path) -> pd.DataFrame:
    """Leser en WaveForms Bode-CSV (linjer som starter med # ignoreres)."""
    df = pd.read_csv(fil, comment="#")
    df.columns = ["f", "mag1", "mag2", "fase2"]
    return df


def finn_krysning(f, y, niva):
    """Første frekvens der y krysser 'niva' (lineær interpolasjon i log f)."""
    under = np.where(y <= niva)[0]
    if len(under) == 0 or under[0] == 0:
        return None
    i = under[0]
    lf = np.interp(niva, [y[i], y[i - 1]], np.log10([f[i], f[i - 1]]))
    return 10 ** lf


def plott_bode(df: pd.DataFrame, tittel: str, filnavn: str, H_ref=None):
    """H_ref: referansenivå (passbånd) for -3 dB. Standard: maks i målingen."""
    f = df["f"].to_numpy()
    H = df["mag2"].to_numpy()
    # unwrap fjerner hoppene ved ±180°
    fase = np.rad2deg(np.unwrap(np.deg2rad(df["fase2"].to_numpy())))

    H_maks = H.max() if H_ref is None else H_ref
    fc = finn_krysning(f, H, H_maks - 3)
    H_nyq = np.interp(F_NYQ, f, H) if f.min() <= F_NYQ <= f.max() else None

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6.5))

    # Amplituderespons
    ax1.semilogx(f, H, color="C0", lw=1.8, label="Målt $|H(f)|$")
    ax1.axhline(H_maks - 3, color="gray", ls="--", lw=1,
                label=f"$-3$ dB ({H_maks - 3:.2f} dB)")
    ax1.axvline(FC_KRAV, color="C2", ls=":", lw=1.4,
                label=rf"Krav: $f_c \geq 0.75\,f_s/2$ = {FC_KRAV:.0f} Hz")
    ax1.axvline(F_NYQ, color="C3", ls=":", lw=1.4,
                label=rf"$f_s/2$ = {F_NYQ:.0f} Hz")

    if fc is not None:
        ax1.plot(fc, H_maks - 3, "o", color="C1", zorder=5,
                 label=rf"$f_c \approx$ {fc:.0f} Hz")
    if H_nyq is not None:
        ax1.plot(F_NYQ, H_nyq, "s", color="C3", zorder=5,
                 label=rf"$|H(f_s/2)| \approx$ {H_nyq:.1f} dB")
        ax1.axhline(H_maks - DEMPING_KRAV, color="C3", ls="--", lw=0.8,
                    alpha=0.6, label=f"Krav: $\\leq$ {H_maks - DEMPING_KRAV:.1f} dB")

    ax1.set_ylabel("Amplitude [dB]")
    ax1.set_title(tittel)
    ax1.legend(fontsize=8.5, loc="lower left")

    # Faserespons
    ax2.semilogx(f, fase, color="C4", lw=1.8)
    for x, c in [(FC_KRAV, "C2"), (F_NYQ, "C3")]:
        ax2.axvline(x, color=c, ls=":", lw=1.4)
    if fc is not None:
        ax2.axvline(fc, color="C1", ls="--", lw=1)
    ax2.set_ylabel("Fase [°]")
    ax2.set_xlabel("Frekvens [Hz]")
    ax2.set_xlim(f.min(), f.max())
    # Vanlige tall på frekvensaksen (ikke 8x10^2 osv.)
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax2.xaxis.set_major_formatter(fmt)
    if f.max() / f.min() < 5:   # smalt område (zoom): tick hver 100 Hz
        ax2.set_xticks(np.arange(np.ceil(f.min() / 100) * 100, f.max() + 1, 100))
    ax2.xaxis.set_minor_formatter(plt.NullFormatter())

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_MAPPE / f"{filnavn}.{ext}", bbox_inches="tight")

    # Oppsummering i terminalen
    print(f"\n{tittel}")
    print(f"  Referansenivå     : {H_maks:.2f} dB")
    if fc is not None:
        ok = "OK" if fc >= FC_KRAV else "IKKE OK"
        print(f"  fc (-3 dB)        : {fc:.1f} Hz  (krav >= {FC_KRAV:.0f} Hz) -> {ok}")
    if H_nyq is not None:
        demp = H_maks - H_nyq
        ok = "OK" if demp >= DEMPING_KRAV else "IKKE OK"
        print(f"  Demping ved fs/2  : {demp:.1f} dB  (krav >= {DEMPING_KRAV:.0f} dB) -> {ok}")


# --- Kjør --------------------------------------------------------------------
if __name__ == "__main__":
    filer = {
        "bodeplott100-6k.csv": ("Bodeplott, 100 Hz – 6 kHz", "bodeplott_100-6k"),
        "bodeplott800-1.6k.csv": ("Bodeplott, 800 Hz – 1,8 kHz (zoom)", "bodeplott_800-1.8k"),
    }
    data = {navn: les_bode(CSV_MAPPE / navn) for navn in filer}

    # Zoom-målingen starter på 800 Hz og ser ikke hele passbåndet, så
    # maksnivået fra den brede målingen brukes som referanse for begge.
    H_ref = data["bodeplott100-6k.csv"]["mag2"].max()
    for navn, (tittel, ut) in filer.items():
        plott_bode(data[navn], tittel, ut, H_ref=H_ref)

    # Begge målingene i samme amplitudeplott
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (navn, df), c in zip(data.items(), ["C0", "C1"]):
        ax.semilogx(df["f"], df["mag2"], color=c, lw=1.6, label=navn)
    ax.axvline(FC_KRAV, color="C2", ls=":", label=f"$0.75\\,f_s/2$ = {FC_KRAV:.0f} Hz")
    ax.axvline(F_NYQ, color="C3", ls=":", label=f"$f_s/2$ = {F_NYQ:.0f} Hz")
    ax.set_xlabel("Frekvens [Hz]")
    ax.set_ylabel("Amplitude [dB]")
    ax.set_title("Amplituderespons – begge målinger")
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_MAPPE / f"amplitude_begge.{ext}", bbox_inches="tight")

    plt.show()
