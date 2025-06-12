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
        plt.figure(figsize=(10, 5))
        plt.hist(df['methane3'], bins=50, color='skyblue', edgecolor='black')
        plt.title("Histograma global de concentración de metano (methane3)")
        plt.xlabel("Concentración de metano (ppb)")
        plt.ylabel("Frecuencia")
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

