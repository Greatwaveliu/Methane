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
            from statsmodels.tsa.stattools import acf, kpss

            df = pd.read_csv(data_file, parse_dates=['measurement_time'])
            df.sort_values('measurement_time', inplace=True)

            df['time_numeric'] = (df['measurement_time'] - df['measurement_time'].min()).dt.total_seconds()
            slope, intercept, *_ = linregress(df['time_numeric'], df['methane3'])
            df['methane3_detrended'] = df['methane3'] - (slope * df['time_numeric'] + intercept)

            plt.figure(figsize=(12, 4))
            plt.plot(df['measurement_time'], df['methane3'], label='Original', alpha=0.6)
            plt.plot(df['measurement_time'], df['methane3_detrended'], label='Detrended')
            plt.title('Methane Concentration: Original vs Detrended')
            plt.xlabel('Time')
            plt.ylabel('Methane3')
            plt.legend()
            plt.tight_layout()
            plt.savefig('methane_original_vs_detrended.png', dpi=300)
            plt.close()

            acf_values = acf(df['methane3_detrended'], nlags=100)
            plt.figure(figsize=(10, 4))
            plt.stem(range(len(acf_values)), acf_values)
            plt.title('Autocorrelation of Detrended Methane3')
            plt.xlabel('Lag')
            plt.ylabel('ACF')
            plt.tight_layout()
            plt.savefig('autocorrelation_detrended_methane3.png', dpi=300)
            plt.close()

            stat, p_value, _, crit_vals = kpss(df['methane3_detrended'], regression='c')
            with open('kpss_results.txt', 'w') as f:
                f.write(f'KPSS Test Statistic: {stat:.4f}\n')
                f.write(f'p-value: {p_value:.4f}\n')
                for k, v in crit_vals.items():
                    f.write(f'Critical Value {k}: {v:.4f}\n')

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

            plt.figure(figsize=(10, 4))
            plt.plot(xf, 2.0/n * np.abs(yf[0:n//2]))
            plt.title('Power Spectral Density of Detrended Methane3')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig('power_spectral_density_methane3.png', dpi=300)
            plt.close()

            return [
                'methane_original_vs_detrended.png',
                'autocorrelation_detrended_methane3.png',
                'loglog_psd_with_beta.png',
                'power_spectral_density_methane3.png'
            ], None

        except Exception as e:
            return [], str(e)

