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
            ax.set_ylabel('F(n)')
            ax.legend()
            fig.tight_layout()
            fig.savefig("DFA.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

            return ["DFA.png"], None
        except Exception as e:
            return [], str(e)

