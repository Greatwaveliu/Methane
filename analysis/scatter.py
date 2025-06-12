import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
class ScatterAnalysis:
    """Scatter plot analysis functionality"""
    
    @staticmethod
    def cargar_datos(ruta_archivo):
        df = pd.read_csv(ruta_archivo, parse_dates=['measurement_time'])
        return df

    @staticmethod
    def graficar_scatter_tiempo_con_tendencia(df, archivo='scatter_tendencia_methane_tiempo.png'):
        df = df.sort_values('measurement_time')
        
        fechas_num = mdates.date2num(df['measurement_time'])
        concentracion = df['methane3'].values
        
        coef = np.polyfit(fechas_num, concentracion, 1)
        polinomio = np.poly1d(coef)
        
        tendencia = polinomio(fechas_num)
        
        pendiente = coef[0]
        ordenada = coef[1]
        ecuacion = f"y = {pendiente:.4e}·x + {ordenada:.2f}"
        
        plt.figure(figsize=(12, 6))
        plt.scatter(df['measurement_time'], concentracion, s=10, alpha=0.5, color='teal', label='Datos')
        plt.plot(df['measurement_time'], tendencia, color='orange', linewidth=2, 
                 label=f'Línea de tendencia\n{ecuacion}')
        
        plt.xlabel('Fecha y hora de medición')
        plt.ylabel('Concentración de metano (ppb)')
        plt.title('Dispersión de concentración de metano vs tiempo con línea de tendencia')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        plt.close()
        return archivo

    @staticmethod
    def graficar_scatter_lat_lon(df, archivo='scatter_lat_lon_methane.png'):
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['longitude'], df['latitude'], 
                            c=df['methane3'], s=15, alpha=0.6, 
                            cmap='viridis', edgecolors='none')
        plt.colorbar(scatter, label='Concentración de metano (ppb)')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.title('Dispersión geográfica de concentración de metano')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        plt.close()
        return archivo

    @staticmethod
    def graficar_scatter_hora_dia(df, archivo='scatter_hora_dia_methane.png'):
        df_copy = df.copy()
        df_copy['hora'] = df_copy['measurement_time'].dt.hour
        
        plt.figure(figsize=(12, 6))
        plt.scatter(df_copy['hora'], df_copy['methane3'], s=15, alpha=0.4, color='purple')
        plt.xlabel('Hora del día')
        plt.ylabel('Concentración de metano (ppb)')
        plt.title('Dispersión de concentración de metano por hora del día')
        plt.xticks(range(0, 24))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        plt.close()
        return archivo

    @staticmethod
    def graficar_scatter_filtrado(df, rango_metano=(100, 2000), archivo='scatter_filtrado_methane.png'):
        df_filtrado = df[(df['methane3'] >= rango_metano[0]) & (df['methane3'] <= rango_metano[1])]
        
        if len(df_filtrado) == 0:
            return None
        
        df_filtrado = df_filtrado.sort_values('measurement_time')
        
        plt.figure(figsize=(12, 6))
        plt.scatter(df_filtrado['measurement_time'], df_filtrado['methane3'], 
                    s=10, alpha=0.6, color='red')
        plt.xlabel('Fecha y hora de medición')
        plt.ylabel('Concentración de metano (ppb)')
        plt.title(f'Dispersión de metano filtrado ({rango_metano[0]:.0f}-{rango_metano[1]:.0f} ppb)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(archivo, dpi=300, bbox_inches='tight')
        plt.close()
        return archivo

    @staticmethod
    def generar_todos_scatter(ruta_archivo):
        try:
            df = ScatterAnalysis.cargar_datos(ruta_archivo)
            archivos_generados = []
            
            archivo1 = ScatterAnalysis.graficar_scatter_tiempo_con_tendencia(df)
            archivos_generados.append(archivo1)
            
            if 'latitude' in df.columns and 'longitude' in df.columns:
                archivo2 = ScatterAnalysis.graficar_scatter_lat_lon(df)
                archivos_generados.append(archivo2)
            
            archivo3 = ScatterAnalysis.graficar_scatter_hora_dia(df)
            archivos_generados.append(archivo3)
            
            try:
                rango_alto = (df['methane3'].quantile(0.75), df['methane3'].quantile(0.95))
                archivo4 = ScatterAnalysis.graficar_scatter_filtrado(df, rango_metano=rango_alto, 
                                                   archivo='scatter_high_concentration.png')
                if archivo4:
                    archivos_generados.append(archivo4)
            except:
                pass
            
            try:
                df_copy = df.copy()
                df_copy['dia_semana'] = df_copy['measurement_time'].dt.dayofweek
                dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
                
                plt.figure(figsize=(12, 6))
                for i in range(7):
                    data_dia = df_copy[df_copy['dia_semana'] == i]
                    if len(data_dia) > 0:
                        plt.scatter(data_dia['measurement_time'], data_dia['methane3'], 
                                   label=dias[i], s=8, alpha=0.6)
                plt.xlabel('Fecha y hora de medición')
                plt.ylabel('Concentración de metano (ppb)')
                plt.title('Dispersión de metano por día de la semana')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                archivo5 = 'scatter_dias_semana.png'
                plt.savefig(archivo5, dpi=300, bbox_inches='tight')
                plt.close()
                archivos_generados.append(archivo5)
            except:
                pass
            
            return archivos_generados, None
            
        except Exception as e:
            error_msg = f"Error al generar gráficos: {str(e)}"
            return [], error_msg

    @staticmethod
    def run_scatter_analysis(data_file):
        return ScatterAnalysis.generar_todos_scatter(data_file)

