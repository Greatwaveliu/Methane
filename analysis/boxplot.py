import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
class BoxAnalysis:
    """Box plot analysis functionality"""
    
    @staticmethod
    def cargar_datos(ruta_archivo):
        df = pd.read_csv(ruta_archivo, parse_dates=['measurement_time'])
        df = df.dropna(subset=['methane3', 'measurement_time'])
        df = df.sort_values('measurement_time').reset_index(drop=True)
        return df

    @staticmethod
    def graficar_boxplot_semanal(df, archivo='boxplot_weekly_with_mean_line.png'):
        sns.set(style="whitegrid")
        df['year_week'] = df['measurement_time'].dt.strftime('%Y-%U')
        
        plt.figure(figsize=(14, 6))
        ax = sns.boxplot(x='year_week', y='methane3', data=df)
        
        ax.set_xticks(ax.get_xticks()[::max(len(ax.get_xticks()) // 15, 1)])
        plt.xticks(rotation=45)
        plt.title('Distribución semanal de metano (methane3)')
        plt.xlabel('Semana (Año-Semana)')
        plt.ylabel('Concentración de Metano (ppb)')
        
        mean_weekly = df.groupby('year_week')['methane3'].mean().reset_index()
        x_positions = range(len(mean_weekly))
        
        plt.plot(x_positions, mean_weekly['methane3'], color='blue', marker='o', label='Media semanal')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(archivo, dpi=300)
        plt.close()
        return archivo

    @staticmethod
    def graficar_boxplot_mensual(df, archivo='boxplot_monthly_with_mean_line.png'):
        sns.set(style="whitegrid")
        df['year_month'] = df['measurement_time'].dt.strftime('%Y-%m')
        
        plt.figure(figsize=(14, 6))
        ax = sns.boxplot(x='year_month', y='methane3', data=df)
        
        ax.set_xticks(ax.get_xticks()[::max(len(ax.get_xticks()) // 15, 1)])
        plt.xticks(rotation=45)
        plt.title('Distribución mensual de metano (methane3)')
        plt.xlabel('Mes (Año-Mes)')
        plt.ylabel('Concentración de Metano (ppb)')
        
        mean_monthly = df.groupby('year_month')['methane3'].mean().reset_index()
        x_positions = range(len(mean_monthly))
        
        plt.plot(x_positions, mean_monthly['methane3'], color='green', marker='o', label='Media mensual')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(archivo, dpi=300)
        plt.close()
        return archivo

    @staticmethod
    def graficar_boxplot_anual(df, archivo='boxplot_yearly_with_mean_line.png'):
        sns.set(style="whitegrid")
        df['year'] = df['measurement_time'].dt.year.astype(str)
        
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(x='year', y='methane3', data=df)
        
        plt.title('Distribución anual de metano (methane3)')
        plt.xlabel('Año')
        plt.ylabel('Concentración de Metano (ppb)')
        
        mean_yearly = df.groupby('year')['methane3'].mean().reset_index()
        x_positions = range(len(mean_yearly))
        
        plt.plot(x_positions, mean_yearly['methane3'], color='red', marker='o', label='Media anual')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(archivo, dpi=300)
        plt.close()
        return archivo

    @staticmethod
    def generar_todos_boxplots(ruta_archivo):
        try:
            df = BoxAnalysis.cargar_datos(ruta_archivo)
            archivos_generados = []
            
            archivo1 = BoxAnalysis.graficar_boxplot_semanal(df)
            archivos_generados.append(archivo1)
            
            archivo2 = BoxAnalysis.graficar_boxplot_mensual(df)
            archivos_generados.append(archivo2)
            
            archivo3 = BoxAnalysis.graficar_boxplot_anual(df)
            archivos_generados.append(archivo3)
            
            return archivos_generados, None
            
        except Exception as e:
            error_msg = f"Error al generar boxplots: {str(e)}"
            return [], error_msg

    @staticmethod
    def run_box_analysis(data_file):
        return BoxAnalysis.generar_todos_boxplots(data_file)

