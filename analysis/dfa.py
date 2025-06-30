import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class DFAAnalysis:
    """Detrended Fluctuation Analysis functionality"""

    @staticmethod
    def run_dfa_analysis(data_file):
        try:
            df = pd.read_csv(data_file, parse_dates=["measurement_time"])
            df = df.sort_values("measurement_time").reset_index(drop=True)
            series = df['methane3'].values

            y = np.cumsum(series - np.mean(series))
            scales = np.logspace(np.log10(4), np.log10(len(y)/4), num=20).astype(int)

            def dfa(y, scale):
                N = len(y)
                n_seg = N // scale
                rms = []
                for i in range(n_seg):
                    seg = y[i*scale:(i+1)*scale]
                    t = np.arange(scale)
                    p = np.polyfit(t, seg, 1)
                    trend = np.polyval(p, t)
                    rms.append(np.sqrt(np.mean((seg - trend)**2)))
                return np.mean(rms)

            F = [dfa(y, s) for s in scales]
            coeffs = np.polyfit(np.log(scales), np.log(F), 1)
            H, intercept = coeffs[0], coeffs[1]

            fig, ax = plt.subplots()
            ax.loglog(scales, F, 'o', label='Datos')
            ax.loglog(scales, np.exp(intercept)*scales**H, label=f'Ajuste H={H:.3f}')
            ax.set_xlabel('Escala n')
            ax.set_ylabel('Función de fluctuación F(n)')
            note = (
                'H=0.5: uncorrelated (white noise)\n'
                'H <0.5: anti-correlated\n'
                '0.5<H <1: long-range correlations (persistent)\n'
                'H=1: 1/f noise (pink noise)\n'
                'H>1: non-stationary behavior (random walk, Brownian motion)'
            )
            ax.text(
                0.95,
                0.05,
                note,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig("DFA.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            return ["DFA.png"], None
        except Exception as e:
            return [], str(e)

