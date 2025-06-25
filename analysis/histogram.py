import pandas as pd
import matplotlib.pyplot as plt
class HistogramAnalysis:
    """Histogram analysis functionality"""
    
    @staticmethod
    def cargar_datos(ruta_archivo):
        df = pd.read_csv(ruta_archivo)
        df = df.dropna(subset=['methane3'])
        return df

    @staticmethod
    def graficar_histograma_global(df, archivo='methane3_histogram.png'):
        """Create a histogram and overlay basic statistics."""
        plt.figure(figsize=(10, 5))
        plt.hist(df['methane3'], bins=50, color='skyblue', edgecolor='black')

        mean_val = df['methane3'].mean()
        median_val = df['methane3'].median()
        std_val = df['methane3'].std()

        plt.axvline(mean_val, color='red', linestyle='--', label=f"Media: {mean_val:.2f}")
        plt.axvline(median_val, color='green', linestyle='--', label=f"Mediana: {median_val:.2f}")

        stats_text = f"Media: {mean_val:.2f}\nMediana: {median_val:.2f}\nDesv. estándar: {std_val:.2f}"
        plt.gca().text(0.95, 0.95, stats_text,
                       transform=plt.gca().transAxes,
                       fontsize=9,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.title("Histograma global de concentración de metano (methane3)")
        plt.xlabel("Concentración de metano (ppb)")
        plt.ylabel("Frecuencia")
        plt.legend()
        plt.tight_layout()
        plt.savefig(archivo, dpi=300)
        plt.close()
        return archivo

    @staticmethod
    def generar_todos_histogramas(ruta_archivo):
        try:
            df = HistogramAnalysis.cargar_datos(ruta_archivo)
            archivos_generados = []
            
            archivo1 = HistogramAnalysis.graficar_histograma_global(df)
            archivos_generados.append(archivo1)
            
            return archivos_generados, None
            
        except Exception as e:
            error_msg = f"Error al generar histogramas: {str(e)}"
            return [], error_msg

    @staticmethod
    def run_histogram_analysis(data_file):
        return HistogramAnalysis.generar_todos_histogramas(data_file)

