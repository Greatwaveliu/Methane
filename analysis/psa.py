class PSAAnalysis:
    """Power Spectral Analysis functionality"""

    @staticmethod
    def run_psa_analysis(data_file):
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.fft import fft, fftfreq
            from scipy.stats import linregress

            df = pd.read_csv(data_file, parse_dates=['measurement_time'])
            df.sort_values('measurement_time', inplace=True)

            df['time_numeric'] = (df['measurement_time'] - df['measurement_time'].min()).dt.total_seconds()
            slope, intercept, *_ = linregress(df['time_numeric'], df['methane3'])
            df['methane3_detrended'] = df['methane3'] - (slope * df['time_numeric'] + intercept)


            signal = df['methane3_detrended'].to_numpy()
            n = len(signal)
            dt = df['time_numeric'].diff().median()
            yf = fft(signal - np.mean(signal))
            xf = fftfreq(n, dt)[:n//2]
            power_spectrum = (2.0 / n * np.abs(yf[0:n // 2])) ** 2
            mask = (xf > 0) & (power_spectrum > 0)
            log_freq = np.log10(xf[mask])
            log_power = np.log10(power_spectrum[mask])
            slope, intercept = np.polyfit(log_freq, log_power, 1)
            beta = -slope

            plt.figure(figsize=(8, 5))
            plt.plot(log_freq, log_power, label='Log Power Spectrum')
            plt.plot(log_freq, intercept + slope * log_freq, 'r--', label=f'Fit: β = {beta:.2f}')
            plt.title('Log-Log Power Spectral Density')
            plt.xlabel('log10(Frequency [Hz])')
            plt.ylabel('log10(Power)')
            plt.legend()
            plt.grid(True)
            note = (
                'β=0: white noise\n'
                'β=1: pink noise (1/f)\n'
                'β=2: Brownian noise'
            )
            plt.gca().text(
                0.95,
                0.95,
                note,
                transform=plt.gca().transAxes,
                fontsize=8,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
            )
            plt.tight_layout()
            plt.savefig('loglog_psd_with_beta.png', dpi=300)
            plt.close()

            with open("spectral_beta_result.txt", "w") as f:
                f.write(f"Spectral exponent β (beta): {beta:.4f}\n")
                if beta < 0.3:
                    f.write("White noise-like process (uncorrelated, flat spectrum)\n")
                elif beta < 1.2:
                    f.write("Pink noise-like process (1/f scaling, correlated fluctuations)\n")
                elif beta < 2.5:
                    f.write("Brownian or red noise (integrated or persistent behavior)\n")
                else:
                    f.write("Strong low-frequency dominance or nonstationary trend\n")

            return ['loglog_psd_with_beta.png'], None

        except Exception as e:
            return [], str(e)

