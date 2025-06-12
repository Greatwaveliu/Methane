import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import os
import glob
from PIL import Image, ImageTk
import threading
import seaborn as sns
from scipy.interpolate import griddata
import geopandas as gpd
import contextily as ctx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from prophet import Prophet

ctk.set_default_color_theme("blue")
ctk.set_appearance_mode("dark")

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

class ContourAnalysis:
    """Contour plot analysis functionality"""
    
    @staticmethod
    def cargar_datos(ruta_archivo):
        df = pd.read_csv(ruta_archivo, parse_dates=['measurement_time'])
        return df

    @staticmethod
    def preparar_cuadricula_geografica(df, columna_valor='methane3', resolucion=100):
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        lat_grid = np.linspace(lat_min, lat_max, resolucion)
        lon_grid = np.linspace(lon_min, lon_max, resolucion)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        puntos = df[['longitude', 'latitude']].values
        valores = df[columna_valor].values
        grid_z = griddata(puntos, valores, (lon_mesh, lat_mesh), method='cubic')
        
        return lon_mesh, lat_mesh, grid_z

    @staticmethod
    def preparar_cuadricula_proyectada(df, columna_valor='methane3', resolucion=200):
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        gdf = gdf.to_crs(epsg=3857)
        
        x_min, y_min, x_max, y_max = gdf.total_bounds
        x_grid = np.linspace(x_min, x_max, resolucion)
        y_grid = np.linspace(y_min, y_max, resolucion)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        
        puntos = np.vstack((gdf.geometry.x, gdf.geometry.y)).T
        valores = gdf[columna_valor].values
        grid_z = griddata(puntos, valores, (x_mesh, y_mesh), method='cubic')
        
        return x_mesh, y_mesh, grid_z, gdf

    @staticmethod
    def graficar_contorno_simple(lon_mesh, lat_mesh, grid_z, archivo='contorno_simple.png'):
        plt.figure(figsize=(10, 8))
        contorno = plt.contourf(lon_mesh, lat_mesh, grid_z, cmap='viridis', levels=15)
        plt.colorbar(contorno, label='Concentración de metano (ppb)')
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.title('Concentración de Methane3 (sin mapa base)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(archivo, dpi=300)
        plt.close()
        return archivo

    @staticmethod
    def graficar_contorno_mapa_base(x_mesh, y_mesh, grid_z, gdf, archivo='contorno_mapa_base.png'):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        contorno = ax.contourf(x_mesh, y_mesh, grid_z, cmap='viridis', levels=15, alpha=0.7)
        gdf.plot(ax=ax, markersize=10, color='black', alpha=0.3, marker='o', label='Datos')
        
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        
        ax.set_xlim(x_mesh.min(), x_mesh.max())
        ax.set_ylim(y_mesh.min(), y_mesh.max())
        ax.set_xlabel('Coordenadas Web Mercator X')
        ax.set_ylabel('Coordenadas Web Mercator Y')
        ax.set_title('Concentración de Methane3 con mapa base')
        plt.colorbar(contorno, ax=ax, label='Concentración de metano (ppb)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(archivo, dpi=300)
        plt.close()
        return archivo

    @staticmethod
    def generar_todos_contornos(ruta_archivo):
        try:
            df = ContourAnalysis.cargar_datos(ruta_archivo)
            archivos_generados = []
            
            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                return [], "No geographic data (latitude/longitude) found in dataset"
            
            lon_mesh, lat_mesh, grid_z_geo = ContourAnalysis.preparar_cuadricula_geografica(df)
            archivo1 = ContourAnalysis.graficar_contorno_simple(lon_mesh, lat_mesh, grid_z_geo)
            archivos_generados.append(archivo1)
            
            try:
                x_mesh, y_mesh, grid_z_proj, gdf = ContourAnalysis.preparar_cuadricula_proyectada(df)
                archivo2 = ContourAnalysis.graficar_contorno_mapa_base(x_mesh, y_mesh, grid_z_proj, gdf)
                archivos_generados.append(archivo2)
            except Exception as e:
                print(f"Warning: Could not create basemap contour plot: {str(e)}")
            
            return archivos_generados, None
            
        except Exception as e:
            error_msg = f"Error al generar contornos: {str(e)}"
            return [], error_msg

    @staticmethod
    def run_contour_analysis(data_file):
        return ContourAnalysis.generar_todos_contornos(data_file)

class KMeansAnalysis:
    """K-means clustering analysis functionality"""
    
    @staticmethod
    def cargar_datos(ruta_archivo, include_time=False, include_methane=False):
        columns = ['latitude', 'longitude']
        if include_time:
            columns.append('measurement_time')
        if include_methane:
            columns.append('methane3')
        df = pd.read_csv(ruta_archivo, usecols=columns, parse_dates=['measurement_time'] if include_time else None)
        df.dropna(subset=columns, inplace=True)
        return df

    @staticmethod
    def run_kmeans_analysis(data_file, mode='lat_lon'):
        try:
            os.makedirs('clusters_espaciales', exist_ok=True)
            os.makedirs('subclusters_methane3', exist_ok=True)
            
            include_time = 'time' in mode.lower()
            include_methane = 'methane' in mode.lower()
            
            df = KMeansAnalysis.cargar_datos(data_file, include_time, include_methane)
            
            if mode == 'time_lat_lon':
                df['timestamp'] = pd.to_datetime(df['measurement_time']).view('int64') // 10**9
                features = df[['latitude', 'longitude', 'timestamp']].values
                cluster_column = 'kmeans_spacetime'
            elif mode == 'time_lat_lon_methane':
                df['timestamp'] = pd.to_datetime(df['measurement_time']).view('int64') // 10**9
                features = df[['latitude', 'longitude', 'timestamp', 'methane3']].values
                cluster_column = 'kmeans_spacetime_methane'
            else:
                features = df[['latitude', 'longitude'] + (['methane3'] if include_methane else [])].values
                cluster_column = 'cluster'
            
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            wcss_list = []
            silhouette_list = []
            db_index_list = []
            K_range = range(2, 31)
            
            for k in K_range:
                kmeans_k = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels_k = kmeans_k.fit_predict(features_scaled)
                
                wcss = kmeans_k.inertia_
                silhouette = silhouette_score(features_scaled, labels_k)
                db_index = davies_bouldin_score(features_scaled, labels_k)
                
                wcss_list.append(wcss)
                silhouette_list.append(silhouette)
                db_index_list.append(db_index)
            
            best_k_candidates = np.where(silhouette_list == max(silhouette_list))[0]
            if len(best_k_candidates) > 1:
                best_k = K_range[best_k_candidates[np.argmin([db_index_list[i] for i in best_k_candidates])]]
            else:
                best_k = K_range[best_k_candidates[0]]
            
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
            df[cluster_column] = kmeans.fit_predict(features_scaled)
            
            generated_files = []
            
            output_csv = f'tula_kmeans_{mode}_clusters.csv'
            df.to_csv(output_csv, index=False)
            generated_files.append(output_csv)
            
            conteos = df[cluster_column].value_counts().sort_index()
            conteo_file = f'tula_kmeans_{mode}_conteos.txt'
            with open(conteo_file, 'w') as f:
                f.write(f"Cantidad de puntos por cluster (K-Means {mode}):\n")
                f.write(str(conteos))
            generated_files.append(conteo_file)
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(K_range, wcss_list, marker='o')
            plt.title('WCSS vs K')
            plt.xlabel('Número de clusters (K)')
            plt.ylabel('WCSS')
            plt.grid(True, linestyle='--', alpha=0.3)
            
            plt.subplot(1, 3, 2)
            plt.plot(K_range, silhouette_list, marker='o', color='green')
            plt.title('Silhouette Score vs K')
            plt.xlabel('Número de clusters (K)')
            plt.ylabel('Silhouette Score')
            plt.grid(True, linestyle='--', alpha=0.3)
            
            plt.subplot(1, 3, 3)
            plt.plot(K_range, db_index_list, marker='o', color='red')
            plt.title('Davies-Bouldin Index vs K')
            plt.xlabel('Número de clusters (K)')
            plt.ylabel('DB Index')
            plt.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            metrics_file = f'tula_kmeans_{mode}_metricas_vs_K.png'
            plt.savefig(metrics_file, dpi=300)
            plt.close()
            generated_files.append(metrics_file)
            
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df['longitude'], df['latitude']),
                crs='EPSG:4326'
            ).to_crs(epsg=3857)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            gdf.plot(
                ax=ax,
                column=cluster_column,
                cmap='tab10',
                markersize=10,
                alpha=0.6,
                legend=True
            )
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
            ax.set_title(f'Distribución espacial por cluster (K-Means, K={best_k}) - {mode}')
            ax.set_axis_off()
            plt.tight_layout()
            map_file = f'tula_kmeans_{mode}_space_scatter_basemap.png'
            plt.savefig(map_file, dpi=300)
            plt.close()
            generated_files.append(map_file)
            
            if mode in ['time_lat_lon', 'time_lat_lon_methane']:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                plt.figure(figsize=(10, 6))
                for cl in range(best_k):
                    subset = df[df[cluster_column] == cl]
                    if not subset.empty:
                        plt.scatter(
                            subset['datetime'],
                            subset[cluster_column],
                            s=5,
                            label=f'Cluster {cl}',
                            alpha=0.6
                        )
                plt.title(f'Distribución temporal por cluster (K={best_k}) - {mode}')
                plt.xlabel('Fecha')
                plt.ylabel('Etiqueta de cluster')
                plt.yticks(range(best_k))
                plt.legend(markerscale=3, fontsize=8, framealpha=0.5)
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                time_scatter_file = f'tula_kmeans_{mode}_time_scatter.png'
                plt.savefig(time_scatter_file, dpi=300)
                plt.close()
                generated_files.append(time_scatter_file)
            
            if include_methane:
                clusters = sorted(df[cluster_column].unique())
                for cl in clusters:
                    subset = df[df[cluster_column] == cl]['methane3']
                    if not subset.empty:
                        plt.figure(figsize=(6, 4))
                        plt.hist(
                            subset,
                            bins=30,
                            edgecolor='black'
                        )
                        plt.title(f'Distribución de methane3 en el cluster {cl} - {mode}')
                        plt.xlabel('methane3 (ppb)')
                        plt.ylabel('Cantidad de puntos')
                        plt.grid(axis='y', linestyle='--', alpha=0.3)
                        plt.tight_layout()
                        hist_file = f'tula_methane3_hist_{mode}_cluster_{cl}.png'
                        plt.savefig(hist_file, dpi=300)
                        plt.close()
                        generated_files.append(hist_file)
                
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    data=df,
                    x=cluster_column,
                    y='methane3',
                    palette='tab10'
                )
                plt.title(f'Distribución de methane3 por cluster (K={best_k}) - {mode}')
                plt.xlabel('Cluster')
                plt.ylabel('methane3 (ppb)')
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                box_file = f'tula_methane3_boxplot_{mode}_cluster.png'
                plt.savefig(box_file, dpi=300)
                plt.close()
                generated_files.append(box_file)
                
                plt.figure(figsize=(10, 6))
                sns.boxplot(
                    data=df,
                    x=cluster_column,
                    y='methane3',
                    palette='tab10'
                )
                sns.swarmplot(
                    data=df,
                    x=cluster_column,
                    y='methane3',
                    color='black',
                    alpha=0.5,
                    size=2
                )
                plt.title(f'Distribución combinada de methane3 por cluster (K={best_k}) - {mode}')
                plt.xlabel('Cluster')
                plt.ylabel('methane3 (ppb)')
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                swarm_file = f'tula_methane3_box_swarmplot_{mode}_cluster.png'
                plt.savefig(swarm_file, dpi=300)
                plt.close()
                generated_files.append(swarm_file)
                
                resumen = df.groupby(cluster_column)['methane3'].describe().round(2)
                resumen.reset_index(inplace=True)
                resumen.columns.name = None
                resumen.to_csv(f'tula_methane3_resumen_por_cluster_{mode}.csv', index=False)
                generated_files.append(f'tula_methane3_resumen_por_cluster_{mode}.csv')
            
            return generated_files, None
            
        except Exception as e:
            error_msg = f"Error en análisis K-means ({mode}): {str(e)}"
            return [], error_msg

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

class LSTMAnalysis:
    """LSTM time series prediction functionality"""
    
    @staticmethod
    def run_lstm_analysis(data_file):
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from sklearn.preprocessing import MinMaxScaler
            from datetime import timedelta
            import os
            
            df = pd.read_csv(data_file, parse_dates=["measurement_time"])
            df = df.sort_values("measurement_time").reset_index(drop=True)
            
            def guardar_fig(fig, nombre):
                fig.savefig(nombre, bbox_inches='tight')
                plt.close(fig)
                return nombre
                
            generated_files = []
            
            ventana = 72
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_scaled = scaler.fit_transform(df[['methane3']])
            
            def crear_seq(data, w, times):
                X, y_seq, ts = [], [], []
                for i in range(len(data) - w):
                    X.append(data[i:i+w, 0])
                    y_seq.append(data[i+w, 0])
                    ts.append(times[i+w])
                return np.array(X), np.array(y_seq), np.array(ts)
                
            X, y_seq, time_seq = crear_seq(data_scaled, ventana, df['measurement_time'])
            X = X.reshape((X.shape[0], X.shape[1], 1))
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y_seq[:split], y_seq[split:]
            time_train, time_test = time_seq[:split], time_seq[split:]
            
            model = Sequential([
                LSTM(64, input_shape=(ventana, 1), return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=75, batch_size=32, verbose=0)
            
            y_train_pred = model.predict(X_train).flatten()
            y_test_pred = model.predict(X_test).flatten()
            
            y_train_inv = scaler.inverse_transform(y_train.reshape(-1,1)).flatten()
            y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
            y_train_pred_inv = scaler.inverse_transform(y_train_pred.reshape(-1,1)).flatten()
            y_test_pred_inv = scaler.inverse_transform(y_test_pred.reshape(-1,1)).flatten()
            
            fig3, ax3 = plt.subplots()
            ax3.plot(time_train, y_train_inv, label='Real (Train)')
            ax3.plot(time_train, y_train_pred_inv, '--', label='Predicción (Train)')
            ax3.plot(time_test, y_test_inv, label='RealalternativesReal (Test)')
            ax3.plot(time_test, y_test_pred_inv, '--', label='Predicción (Test)')
            ax3.set_xlabel('Tiempo (measurement_time)')
            ax3.set_ylabel('Concentración de metano (ppb)')
            ax3.legend()
            file1 = guardar_fig(fig3, "LSTM.png")
            generated_files.append(file1)
            
            ultima_ventana = data_scaled[-ventana:].reshape(1, ventana, 1)
            predicciones_futuras = []
            fechas_futuras = []
            
            fecha_base = df['measurement_time'].iloc[-1]
            for i in range(180):
                pred = model.predict(ultima_ventana)[0, 0]
                predicciones_futuras.append(pred)
                nueva_fecha = fecha_base + timedelta(days=i + 1)
                fechas_futuras.append(nueva_fecha)
                
                nueva_entrada = np.append(ultima_ventana[0, 1:, 0], pred).reshape(1, ventana, 1)
                ultima_ventana = nueva_entrada
                
            predicciones_futuras_inv = scaler.inverse_transform(np.array(predicciones_futuras).reshape(-1, 1)).flatten()
            
            fig_future, axf = plt.subplots()
            axf.plot(df['measurement_time'], df['methane3'], label='Histórico')
            axf.plot(fechas_futuras, predicciones_futuras_inv, '--', color='red', label='Predicción futura (6 meses)')
            axf.set_xlabel('Tiempo (measurement_time)')
            axf.set_ylabel('Concentración de metano (ppb)')
            axf.set_title('Predicción futura con LSTM')
            axf.legend()
            file2 = "LSTM_Futuro.png"
            fig_future.savefig(file2, bbox_inches='tight')
            plt.close(fig_future)
            generated_files.append(file2)
            
            return generated_files, None
            
        except Exception as e:
            return [], str(e)

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

class ProphetAnalysis:
    """Prophet time series prediction functionality"""
    
    @staticmethod
    def run_prophet_analysis(data_file):
        try:
            df = pd.read_csv(data_file, parse_dates=["measurement_time"])
            df = df.sort_values("measurement_time").reset_index(drop=True)
            
            df_prop = df[['measurement_time', 'methane3']].rename(columns={'measurement_time': 'ds', 'methane3': 'y'})
            
            modelo_p = Prophet(yearly_seasonality=True)
            modelo_p.fit(df_prop)
            
            futuro = modelo_p.make_future_dataframe(periods=365, freq='D')
            forecast = modelo_p.predict(futuro)
            
            fig4 = modelo_p.plot(forecast)
            ax4 = fig4.gca()
            ax4.set_xlabel('Fecha')
            ax4.set_ylabel('Concentración de metano (ppb)')
            fig4.tight_layout()
            file1 = "Prophet.png"
            fig4.savefig(file1, bbox_inches='tight')
            plt.close(fig4)
            
            fig5 = modelo_p.plot_components(forecast)
            fig5.tight_layout()
            file2 = "Prophet components.png"
            fig5.savefig(file2, bbox_inches='tight')
            plt.close(fig5)
            
            return [file1, file2], None
            
        except Exception as e:
            return [], str(e)

class MassEstimation:
    """Methane mass estimation functionality"""
    
    @staticmethod
    def run_mass_estimation(data_file):
        try:
            df = pd.read_csv(data_file)
            
            # Check for required columns
            required_columns = ['methane3', 'longitude', 'latitude']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {', '.join(missing)}")
            
            # Constants
            M_CH4 = 16.04e-3  # kg/mol
            M_air = 28.97e-3  # kg/mol
            rho_air = 1.2     # kg/m³
            mixing_height_m = 1000  # effective mixing height
            pixel_area_km2 = 1       # each point represents 1 km²
            pixel_area_m2 = pixel_area_km2 * 1e6
            
            # Calculate mass concentration
            def ppb_to_kg_m3(ppb):
                return ppb * (1e-9 * M_CH4 * rho_air / M_air)
            
            df['methane_kg_m3'] = ppb_to_kg_m3(df['methane3'])
            
            # Calculate methane mass per pixel (kg)
            df['methane_mass_kg'] = df['methane_kg_m3'] * pixel_area_m2 * mixing_height_m
            
            # Total methane mass (tons)
            total_mass_tons = df['methane_mass_kg'].sum() / 1000
            
            # Visualization
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs='EPSG:4326')
            gdf = gdf.to_crs(epsg=3857)  # Web Mercator
            
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf.plot(column='methane_mass_kg', ax=ax, cmap='viridis', markersize=20, legend=True)
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
            ax.set_title(f"Spatial Distribution of Methane Mass (kg)\nTotal Estimated: {total_mass_tons:.2f} tons", fontsize=14)
            plt.tight_layout()
            plot_file = "methane_mass_map.png"
            plt.savefig(plot_file, dpi=300)
            plt.close()
            
            return [plot_file], total_mass_tons, None
        except Exception as e:
            return [], None, str(e)

class ImageViewer(ctk.CTkFrame):
    """Enhanced Image/Text viewer with navigation, zoom, and pan"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.image_files = []
        self.current_index = 0
        self.orig_image = None
        self.current_display_image = None
        self.tk_image = None
        self.image_id = None
        self.current_zoom = 1.0
        self.setup_ui()

    def setup_ui(self):
        nav_frame = ctk.CTkFrame(self)
        nav_frame.pack(fill="x", pady=(0, 10))

        self.prev_btn = ctk.CTkButton(nav_frame, text="◀ Previous", command=self.prev_image, width=100)
        self.prev_btn.pack(side="left", padx=5)

        self.image_label = ctk.CTkLabel(nav_frame, text="No files")
        self.zoom_label = ctk.CTkLabel(nav_frame, text="Zoom: 100%")
        self.zoom_label.pack(side="right", padx=10)
        self.image_label.pack(side="left", expand=True)

        self.next_btn = ctk.CTkButton(nav_frame, text="Next ▶", command=self.next_image, width=100)
        self.next_btn.pack(side="right", padx=5)

        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.pack(fill="both", expand=True)

        self.image_canvas = tk.Canvas(self.image_frame, bg="#1a1a1a", highlightthickness=0, xscrollincrement=1, yscrollincrement=1)
        self.image_canvas.pack(fill="both", expand=True)
        self.image_canvas.bind("<ButtonPress-1>", lambda e: self.image_canvas.scan_mark(e.x, e.y))
        self.image_canvas.bind("<B1-Motion>", lambda e: self.image_canvas.scan_dragto(e.x, e.y, gain=1))
        self.image_canvas.bind("<MouseWheel>", self._zoom_image)
        self.image_canvas.bind("<Button-4>", self._zoom_image)
        self.image_canvas.bind("<Button-5>", self._zoom_image)
        self.image_canvas.bind("<Double-Button-1>", self._reset_zoom)

        self.text_display = ctk.CTkTextbox(self.image_frame, wrap="none")
        self.text_display.configure(state="disabled")

    def _zoom_image(self, event):
        if not self.orig_image:
            return

        factor = 1.1 if (event.delta > 0 or getattr(event, 'num', 0) == 4) else 0.9
        self.current_zoom *= factor

        w, h = max(1, int(self.orig_image.width * self.current_zoom)), max(1, int(self.orig_image.height * self.current_zoom))
        self.current_display_image = self.orig_image.resize((w, h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.current_display_image)
        self.image_canvas.delete("all")
        self.image_id = self.image_canvas.create_image(self._center_x(w), self._center_y(h), anchor="nw", image=self.tk_image)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))
        self._update_zoom_label()

    def _reset_zoom(self, event=None):
        if not self.orig_image:
            return
        self.image_canvas.update_idletasks()
        canvas_w = self.image_canvas.winfo_width() or 800
        self.image_canvas.update_idletasks()
        canvas_h = self.image_canvas.winfo_height() or 600
        orig_w, orig_h = self.orig_image.size
        scale_w = canvas_w / orig_w
        scale_h = canvas_h / orig_h
        self.current_zoom = min(scale_w, scale_h, 1.0)
        w, h = max(1, int(orig_w * self.current_zoom)), max(1, int(orig_h * self.current_zoom))
        self.current_display_image = self.orig_image.resize((w, h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.current_display_image)
        self.image_canvas.delete("all")
        self.image_id = self.image_canvas.create_image(self._center_x(w), self._center_y(h), anchor="nw", image=self.tk_image)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))

    def _center_x(self, width):
        canvas_width = self.image_canvas.winfo_width() or 800
        return max((canvas_width - width) // 2, 0)

    def _center_y(self, height):
        canvas_height = self.image_canvas.winfo_height() or 600
        return max((canvas_height - height) // 2, 0)

    def load_images(self, image_files):
        self.image_files = [f for f in image_files if os.path.exists(f)]
        self.current_index = 0
        self.update_display()

    def update_display(self):
        self.image_canvas.pack_forget()
        self.text_display.pack_forget()

        if not self.image_files:
            self.image_label.configure(text="No files")
            return

        self.prev_btn.configure(state="normal" if self.current_index > 0 else "disabled")
        self.next_btn.configure(state="normal" if self.current_index < len(self.image_files) - 1 else "disabled")

        filepath = self.image_files[self.current_index]
        ext = os.path.splitext(filepath)[1].lower()
        filename = os.path.basename(filepath)
        self.image_label.configure(text=f"{filename} ({self.current_index+1}/{len(self.image_files)})")

        if ext in (".png", ".jpg", ".jpeg", ".gif"):
            self.orig_image = Image.open(filepath)

            # Resize to fit current canvas size
            canvas_w = self.image_canvas.winfo_width() or 800
            canvas_h = self.image_canvas.winfo_height() or 600

            orig_w, orig_h = self.orig_image.size
            scale_w = canvas_w / orig_w
            scale_h = canvas_h / orig_h
            initial_scale = min(scale_w, scale_h, 1.0)

            self.current_zoom = initial_scale
            w, h = max(1, int(orig_w * self.current_zoom)), max(1, int(orig_h * self.current_zoom))
            self.current_display_image = self.orig_image.resize((w, h), Image.Resampling.LANCZOS)
            self.tk_image = ImageTk.PhotoImage(self.current_display_image)

            self.image_canvas.pack(fill="both", expand=True)
            self.image_canvas.delete("all")
            self.image_id = self.image_canvas.create_image(self._center_x(w), self._center_y(h), anchor="nw", image=self.tk_image)
            self.image_canvas.config(scrollregion=self.image_canvas.bbox("all"))

        elif ext in (".csv", ".txt"):
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    data = f.read()
                self.text_display.configure(state="normal")
                self.text_display.delete("0.0", "end")
                self.text_display.insert("0.0", data)
                self.text_display.configure(state="disabled")
                self.text_display.pack(fill="both", expand=True)
            except Exception as e:
                print(f"Error loading text: {e}")

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()

    def _update_zoom_label(self):
        percent = int(self.current_zoom * 100)
        self.zoom_label.configure(text=f"Zoom: {percent}%")

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.update_display()

class HomeImage(ctk.CTkFrame):
    def __init__(self, parent, image_path, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.image = Image.open(image_path)
        self.tk_image = None
        self.image_id = None
        self.bind("<Configure>", self.on_resize)

    def on_resize(self, event):
        if not self.image:
            return
        canvas_w, canvas_h = event.width, event.height
        scale = min(canvas_w / self.image.width, canvas_h / self.image.height, 1.0)
        w, h = int(self.image.width * scale), int(self.image.height * scale)
        resized = self.image.resize((w, h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        x = (canvas_w - w) // 2
        y = (canvas_h - h) // 2
        self.image_id = self.canvas.create_image(x, y, anchor="nw", image=self.tk_image)

class MethaneAnalysisApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Methane Emission Analysis - Mexico")
        self.geometry("1400x900")

        # Load theme icons
        try:
            sun_img = Image.open("sun.png")
            moon_img = Image.open("moon.png")
            self.sun_ctk = ctk.CTkImage(light_image=sun_img, dark_image=sun_img, size=(20, 20))
            self.moon_ctk = ctk.CTkImage(light_image=moon_img, dark_image=moon_img, size=(20, 20))
        except Exception as e:
            print(f"Error loading theme icons: {e}")
            self.sun_ctk = None
            self.moon_ctk = None

        self.tab_frames = {}
        self.active_tab = None
        self.data_file = None

        self.create_widgets()

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_panel = ctk.CTkFrame(main_frame, width=300)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)

        right_panel = ctk.CTkFrame(main_frame)
        right_panel.pack(side="right", fill="both", expand=True)

        # Add theme toggle frame
        theme_frame = ctk.CTkFrame(right_panel)
        theme_frame.pack(side="top", fill="x", pady=(0, 10))

        self.theme_button = ctk.CTkButton(theme_frame, text="", width=30, height=30,
                                          command=self.toggle_theme)
        self.theme_button.pack(side="right", padx=10)
        self.update_theme_button()

        tab_button_frame = ctk.CTkFrame(right_panel)
        tab_button_frame.pack(fill="x", pady=(0, 10))

        button_names = [
            "Home", "Load Data", "Scatter Plot", "Histogram", "Box Plot", "Contour",
            "K-means: Lat-Lon", "K-means: Time-Lat-Lon", "K-means: Lat-Lon-Methane", 
            "K-means: Time-Lat-Lon-Methane", "DFA", "PSA", "LSTM", "Prophet", "Estimate Mass"
        ]

        self.tab_buttons = {}
        for i, name in enumerate(button_names):
            btn = ctk.CTkButton(
                tab_button_frame,
                text=name,
                command=lambda n=name: self.handle_tab_click(n),
                width=150,
                height=30
            )
            row = i // 6
            col = i % 6
            btn.grid(row=row, column=col, padx=5, pady=5)
            self.tab_buttons[name] = btn

        self.display_area = ctk.CTkFrame(right_panel)
        self.display_area.pack(fill="both", expand=True)

        self.create_tab_contents(button_names)
        self.switch_tab("Home")

        self.add_section(left_panel, "Data", ["Load Data"])
        self.add_section(left_panel, "Visualization", ["Scatter Plot", "Histogram", "Box Plot", "Contour"])
        self.add_section(left_panel, "Clustering", [
            "K-means: Lat-Lon", "K-means: Time-Lat-Lon",
            "K-means: Lat-Lon-Methane", "K-means: Time-Lat-Lon-Methane"
        ])
        self.add_section(left_panel, "Time Series", ["DFA", "PSA", "LSTM", "Prophet"])
        self.add_section(left_panel, "Mass Estimation", ["Estimate Mass"])

    def create_tab_contents(self, button_names):
        for name in button_names:
            frame = ctk.CTkFrame(self.display_area)
            self.tab_frames[name] = frame

            if name == "Home":
                self.create_home_tab(frame)
            elif name == "Load Data":
                self.create_load_data_tab(frame)
            elif name == "Scatter Plot":
                self.create_scatter_plot_tab(frame)
            elif name == "Histogram":
                self.create_histogram_tab(frame)
            elif name == "Box Plot":
                self.create_box_plot_tab(frame)
            elif name == "Contour":
                self.create_contour_tab(frame)
            elif name == "K-means: Lat-Lon":
                self.create_kmeans_tab(frame, mode="lat_lon")
            elif name == "K-means: Time-Lat-Lon":
                self.create_kmeans_tab(frame, mode="time_lat_lon")
            elif name == "K-means: Lat-Lon-Methane":
                self.create_kmeans_tab(frame, mode="lat_lon_methane")
            elif name == "K-means: Time-Lat-Lon-Methane":
                self.create_kmeans_tab(frame, mode="time_lat_lon_methane")
            elif name == "DFA":
                self.create_dfa_tab(frame)
            elif name == "PSA":
                self.create_psa_tab(frame)
            elif name == "LSTM":
                self.create_lstm_tab(frame)
            elif name == "Prophet":
                self.create_prophet_tab(frame)
            elif name == "Estimate Mass":
                self.create_mass_estimation_tab(frame)
            else:
                ctk.CTkLabel(frame, text=f"{name} Tab", font=("Arial", 16)).pack(pady=20)
                ctk.CTkLabel(frame, text="Feature coming soon...").pack()

    def create_home_tab(self, frame):
        ctk.CTkLabel(frame, text="🇲🇽 Methane Emission Analysis - Home", font=("Arial", 18)).pack(pady=30)
        try:
            home_tab = HomeImage(frame, image_path="mexico.png")
            home_tab.pack(fill="both", expand=True)
        except Exception as e:
            ctk.CTkLabel(frame, text=f"(Mexico image not found: {e})").pack()

    # 1) In create_load_data_tab: add a CTkTextbox for the summary
    def create_load_data_tab(self, frame):
        ctk.CTkLabel(frame, text="Load Data", font=("Arial", 18)).pack(pady=20)
        load_btn = ctk.CTkButton(frame, text="Select CSV File", 
                                command=self.load_data_file, width=200)
        load_btn.pack(pady=10)
        
        self.data_status_label = ctk.CTkLabel(frame, text="No data loaded")
        self.data_status_label.pack(pady=10)
        
        # Add a textbox to display the summary (scrollable by default)
        self.summary_text = ctk.CTkTextbox(frame, wrap="none")
        self.summary_text.configure(state="disabled")  # start as read-only
        self.summary_text.pack(fill="both", expand=True, padx=20, pady=10)

    # 2) In load_data_file: load CSV without forcing parse, then update summary_text
    def load_data_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Read CSV without parse_dates so we can catch missing columns
                df = pd.read_csv(file_path)
                self.data_file = file_path
                filename = os.path.basename(file_path)
                self.data_status_label.configure(
                    text=f"✅ Loaded: {filename} ({len(df)} rows)"
                )
                messagebox.showinfo("Success", 
                                    f"Data loaded successfully!\n{len(df)} rows loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
                self.data_status_label.configure(text="❌ Failed to load data")
                return

            # Prepare and display summary in the textbox
            self.summary_text.configure(state="normal")
            self.summary_text.delete("0.0", "end")  # clear previous text
            
            # List all column names
            cols = ", ".join(df.columns)
            self.summary_text.insert("0.0", f"Columns: {cols}\n")
            
            # Check for 'measurement_time' column
            if 'measurement_time' in df.columns:
                try:
                    # Attempt to parse measurement_time as datetime
                    df['measurement_time'] = pd.to_datetime(df['measurement_time'])
                    earliest = df['measurement_time'].min()
                    latest = df['measurement_time'].max()
                    # Format the dates for readability
                    earliest_str = earliest.strftime("%Y-%m-%d %H:%M:%S")
                    latest_str = latest.strftime("%Y-%m-%d %H:%M:%S")
                    self.summary_text.insert("end", f"Earliest measurement time: {earliest_str}\n")
                    self.summary_text.insert("end", f"Latest measurement time:   {latest_str}")
                except Exception:
                    # Parsing failed
                    self.summary_text.insert("end", 
                        "Column 'measurement_time' exists but could not be parsed as datetime.")
            else:
                # Column missing
                self.summary_text.insert("end", 
                    "Column 'measurement_time' not found in dataset.")
            
            self.summary_text.configure(state="disabled")


    def create_scatter_plot_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Scatter Plot Analysis", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_scatter_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_scatter_analysis, width=120)
        self.run_scatter_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_scatter_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.scatter_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate scatter plots")
        self.scatter_status.pack(pady=5)
        
        self.scatter_image_viewer = ImageViewer(frame)
        self.scatter_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_histogram_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Histogram Analysis", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_histogram_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_histogram_analysis, width=120)
        self.run_histogram_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_histogram_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.histogram_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate histograms")
        self.histogram_status.pack(pady=5)
        
        self.histogram_image_viewer = ImageViewer(frame)
        self.histogram_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_box_plot_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Box Plot Analysis", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_box_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_box_analysis, width=120)
        self.run_box_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_box_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.box_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate box plots")
        self.box_status.pack(pady=5)
        
        self.box_image_viewer = ImageViewer(frame)
        self.box_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_contour_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Contour Plot Analysis", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_contour_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_contour_analysis, width=120)
        self.run_contour_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_contour_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.contour_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate contour plots")
        self.contour_status.pack(pady=5)
        
        self.contour_image_viewer = ImageViewer(frame)
        self.contour_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_kmeans_tab(self, frame, mode):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text=f"K-means Clustering Analysis ({mode})", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        run_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=lambda: self.run_kmeans_analysis(mode), width=120)
        run_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=lambda: self.refresh_kmeans_images(mode), width=120)
        refresh_btn.pack(side="left", padx=5)
        
        status_label = ctk.CTkLabel(frame, text=f"Click 'Run Analysis' to perform K-means clustering ({mode})")
        status_label.pack(pady=5)
        
        image_viewer = ImageViewer(frame)
        image_viewer.pack(fill="both", expand=True, padx=10, pady=10)
        
        setattr(self, f'run_kmeans_{mode}_btn', run_btn)
        setattr(self, f'kmeans_{mode}_status', status_label)
        setattr(self, f'kmeans_{mode}_image_viewer', image_viewer)

    def create_dfa_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(title_frame, text="DFA (Detrended Fluctuation Analysis)", font=("Arial", 18)).pack(side="left", padx=20, pady=10)

        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)

        self.run_dfa_btn = ctk.CTkButton(btn_frame, text="Run Analysis", command=self.run_dfa_analysis, width=120)
        self.run_dfa_btn.pack(side="left", padx=5)

        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", command=self.refresh_dfa_images, width=120)
        refresh_btn.pack(side="left", padx=5)

        self.dfa_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate DFA plot")
        self.dfa_status.pack(pady=5)

        self.dfa_image_viewer = ImageViewer(frame)
        self.dfa_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_psa_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(title_frame, text="PSA (Power Spectral Analysis)", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        self.run_psa_btn = ctk.CTkButton(btn_frame, text="Run Analysis", command=self.run_psa_analysis, width=120)
        self.run_psa_btn.pack(side="left", padx=5)
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", command=self.refresh_psa_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        self.psa_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate PSA plots")
        self.psa_status.pack(pady=5)
        self.psa_image_viewer = ImageViewer(frame)
        self.psa_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_lstm_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="LSTM Time Series Prediction", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_lstm_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_lstm_analysis, width=120)
        self.run_lstm_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_lstm_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.lstm_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate LSTM predictions")
        self.lstm_status.pack(pady=5)
        
        self.lstm_image_viewer = ImageViewer(frame)
        self.lstm_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_prophet_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Prophet Time Series Prediction", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_prophet_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_prophet_analysis, width=120)
        self.run_prophet_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_prophet_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.prophet_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to generate Prophet predictions")
        self.prophet_status.pack(pady=5)
        
        self.prophet_image_viewer = ImageViewer(frame)
        self.prophet_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def create_mass_estimation_tab(self, frame):
        title_frame = ctk.CTkFrame(frame)
        title_frame.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(title_frame, text="Methane Mass Estimation", font=("Arial", 18)).pack(side="left", padx=20, pady=10)
        
        btn_frame = ctk.CTkFrame(title_frame)
        btn_frame.pack(side="right", padx=20, pady=10)
        
        self.run_mass_btn = ctk.CTkButton(btn_frame, text="Run Analysis", 
                               command=self.run_mass_estimation, width=120)
        self.run_mass_btn.pack(side="left", padx=5)
        
        refresh_btn = ctk.CTkButton(btn_frame, text="Refresh Images", 
                                   command=self.refresh_mass_images, width=120)
        refresh_btn.pack(side="left", padx=5)
        
        self.mass_status = ctk.CTkLabel(frame, text="Click 'Run Analysis' to estimate methane mass")
        self.mass_status.pack(pady=5)
        
        self.mass_image_viewer = ImageViewer(frame)
        self.mass_image_viewer.pack(fill="both", expand=True, padx=10, pady=10)

    def update_theme_button(self):
        if self.sun_ctk and self.moon_ctk:
            current_mode = ctk.get_appearance_mode().lower()
            if current_mode == "dark":
                self.theme_button.configure(image=self.sun_ctk)
            else:
                self.theme_button.configure(image=self.moon_ctk)
        else:
            self.theme_button.configure(text="Toggle Theme")

    def toggle_theme(self):
        current_mode = ctk.get_appearance_mode().lower()
        new_mode = "light" if current_mode == "dark" else "dark"
        ctk.set_appearance_mode(new_mode)
        self.update_theme_button()

    def add_section(self, panel, title, names):
        ctk.CTkLabel(panel, text=title, font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(20, 0))
        for name in names:
            ctk.CTkButton(panel, text=name, command=lambda n=name: self.handle_tab_click(n)).pack(fill="x", padx=10, pady=5)

    def handle_tab_click(self, name):
        tabs_requiring_data = [
            "Scatter Plot", "Histogram", "Box Plot", "Contour", 
            "K-means: Lat-Lon", "K-means: Time-Lat-Lon", 
            "K-means: Lat-Lon-Methane", "K-means: Time-Lat-Lon-Methane",
            "DFA", "PSA", "LSTM", "Prophet", "Estimate Mass"
        ]
        if name in tabs_requiring_data:
            if not self.data_file:
                messagebox.showwarning("Warning", "Please load a data file first!")
                self.switch_tab("Load Data")
                return
            
            self.switch_tab(name)
            
            if name == "Scatter Plot":
                self.refresh_scatter_images()
            elif name == "Histogram":
                self.refresh_histogram_images()
            elif name == "Box Plot":
                self.refresh_box_images()
            elif name == "Contour":
                self.refresh_contour_images()
            elif name == "K-means: Lat-Lon":
                self.refresh_kmeans_images("lat_lon")
            elif name == "K-means: Time-Lat-Lon":
                self.refresh_kmeans_images("time_lat_lon")
            elif name == "K-means: Lat-Lon-Methane":
                self.refresh_kmeans_images("lat_lon_methane")
            elif name == "K-means: Time-Lat-Lon-Methane":
                self.refresh_kmeans_images("time_lat_lon_methane")
            elif name == "DFA":
                self.refresh_dfa_images()
            elif name == "PSA":
                self.refresh_psa_images()
            elif name == "LSTM":
                self.refresh_lstm_images()
            elif name == "Prophet":
                self.refresh_prophet_images()
            elif name == "Estimate Mass":
                self.refresh_mass_images()
        else:
            self.switch_tab(name)

    def switch_tab(self, name):
        if self.active_tab:
            self.tab_frames[self.active_tab].pack_forget()
            self.tab_buttons[self.active_tab].configure(fg_color="transparent")

        self.tab_frames[name].pack(fill="both", expand=True)
        self.tab_buttons[name].configure(fg_color=("gray75", "gray25"))
        self.active_tab = name

    def refresh_scatter_images(self):
        if not hasattr(self, 'scatter_image_viewer'):
            return
            
        scatter_patterns = [
            'scatter_*.png',
            '*scatter*.png'
        ]
        
        image_files = []
        for pattern in scatter_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.scatter_image_viewer.load_images(image_files)
            self.scatter_status.configure(text=f"✅ Found {len(image_files)} scatter plot(s)")
        else:
            self.scatter_status.configure(text="No scatter plot images found")

    def refresh_histogram_images(self):
        if not hasattr(self, 'histogram_image_viewer'):
            return
            
        histogram_patterns = [
            '*histogram*.png',
            'histogram_*.png'
        ]
        
        image_files = []
        for pattern in histogram_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.histogram_image_viewer.load_images(image_files)
            self.histogram_status.configure(text=f"✅ Found {len(image_files)} histogram(s)")
        else:
            self.histogram_status.configure(text="No histogram images found")

    def refresh_box_images(self):
        if not hasattr(self, 'box_image_viewer'):
            return
            
        box_patterns = [
            '*boxplot*.png',
            'boxplot_*.png'
        ]
        
        image_files = []
        for pattern in box_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.box_image_viewer.load_images(image_files)
            self.box_status.configure(text=f"✅ Found {len(image_files)} box plot(s)")
        else:
            self.box_status.configure(text="No box plot images found")

    def refresh_contour_images(self):
        if not hasattr(self, 'contour_image_viewer'):
            return
            
        contour_patterns = [
            '*contour*.png',
            'contorno_*.png'
        ]
        
        image_files = []
        for pattern in contour_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.contour_image_viewer.load_images(image_files)
            self.contour_status.configure(text=f"✅ Found {len(image_files)} contour plot(s)")
        else:
            self.contour_status.configure(text="No contour plot images found")

    def refresh_kmeans_images(self, mode):
        viewer = getattr(self, f'kmeans_{mode}_image_viewer', None)
        if not viewer:
            return
            
        kmeans_patterns = [
            f'*kmeans_{mode}*.png',
            f'*cluster_{mode}*.png',
            f'tula_kmeans_{mode}_metricas_vs_K.png',
            f'tula_kmeans_{mode}_space_scatter_basemap.png',
            f'tula_methane3_hist_{mode}_cluster_*.png',
            f'tula_methane3_boxplot_{mode}_cluster.png',
            f'tula_methane3_box_swarmplot_{mode}_cluster.png',
            f'tula_kmeans_{mode}_time_scatter.png'
        ]
        
        image_files = []
        for pattern in kmeans_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            viewer.load_images(image_files)
            getattr(self, f'kmeans_{mode}_status').configure(text=f"✅ Found {len(image_files)} K-means plot(s)")
        else:
            getattr(self, f'kmeans_{mode}_status').configure(text="No K-means images found")

    def refresh_dfa_images(self):
        image_files = [f for f in glob.glob("DFA*.png") if os.path.exists(f)]
        if image_files:
            self.dfa_image_viewer.load_images(image_files)
            self.dfa_status.configure(text=f"✅ Found {len(image_files)} DFA plot(s)")
        else:
            self.dfa_status.configure(text="No DFA images found")

    def refresh_psa_images(self):
        image_files = [f for f in glob.glob("*psd*.png") + glob.glob("*detrended*.png") + glob.glob("*autocorrelation*.png") if os.path.exists(f)]
        if image_files:
            self.psa_image_viewer.load_images(image_files)
            self.psa_status.configure(text=f"✅ Found {len(image_files)} PSA plot(s)")
        else:
            self.psa_status.configure(text="No PSA images found")

    def refresh_lstm_images(self):
        if not hasattr(self, 'lstm_image_viewer'):
            return
            
        lstm_patterns = [
            'LSTM*.png',
            '*LSTM*.png'
        ]
        
        image_files = []
        for pattern in lstm_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.lstm_image_viewer.load_images(image_files)
            self.lstm_status.configure(text=f"✅ Found {len(image_files)} LSTM plot(s)")
        else:
            self.lstm_status.configure(text="No LSTM images found")

    def refresh_prophet_images(self):
        if not hasattr(self, 'prophet_image_viewer'):
            return
            
        prophet_patterns = [
            'Prophet*.png',
            '*Prophet*.png'
        ]
        
        image_files = []
        for pattern in prophet_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.prophet_image_viewer.load_images(image_files)
            self.prophet_status.configure(text=f"✅ Found {len(image_files)} Prophet plot(s)")
        else:
            self.prophet_status.configure(text="No Prophet images found")

    def refresh_mass_images(self):
        if not hasattr(self, 'mass_image_viewer'):
            return
            
        mass_patterns = [
            'methane_mass_map*.png',
            '*mass*.png'
        ]
        
        image_files = []
        for pattern in mass_patterns:
            image_files.extend(glob.glob(pattern))
        
        image_files = sorted(list(set(image_files)))
        
        if image_files:
            self.mass_image_viewer.load_images(image_files)
            self.mass_status.configure(text=f"✅ Found {len(image_files)} mass estimation plot(s)")
        else:
            self.mass_status.configure(text="No mass estimation images found")

    def run_scatter_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.scatter_status.configure(text="🔄 Running scatter plot analysis...")
        self.run_scatter_btn.configure(state="disabled", text="Running...")
        
        thread = threading.Thread(target=self._run_scatter_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_scatter_analysis_thread(self):
        try:
            generated_files, error = ScatterAnalysis.run_scatter_analysis(self.data_file)
            self.after(0, self._scatter_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._scatter_analysis_complete, [], str(e))

    def _scatter_analysis_complete(self, generated_files, error):
        self.run_scatter_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.scatter_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Scatter plot analysis failed:\n{error}")
        elif generated_files:
            self.scatter_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.scatter_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} scatter plots:\n" + 
                              "\n".join(plot_names))
        else:
            self.scatter_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No scatter plots were generated. Please check your data file.")

    def run_histogram_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.histogram_status.configure(text="🔄 Running histogram analysis...")
        self.run_histogram_btn.configure(state="disabled", text="Running...")
        
        thread = threading.Thread(target=self._run_histogram_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_histogram_analysis_thread(self):
        try:
            generated_files, error = HistogramAnalysis.run_histogram_analysis(self.data_file)
            self.after(0, self._histogram_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._histogram_analysis_complete, [], str(e))

    def _histogram_analysis_complete(self, generated_files, error):
        self.run_histogram_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.histogram_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Histogram analysis failed:\n{error}")
        elif generated_files:
            self.histogram_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.histogram_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} histograms:\n" + 
                              "\n".join(plot_names))
        else:
            self.histogram_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No histograms were generated. Please check your data file.")

    def run_box_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.box_status.configure(text="🔄 Running box plot analysis...")
        self.run_box_btn.configure(state="disabled", text="Running...")
        
        thread = threading.Thread(target=self._run_box_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_box_analysis_thread(self):
        try:
            generated_files, error = BoxAnalysis.run_box_analysis(self.data_file)
            self.after(0, self._box_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._box_analysis_complete, [], str(e))

    def _box_analysis_complete(self, generated_files, error):
        self.run_box_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.box_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Box plot analysis failed:\n{error}")
        elif generated_files:
            self.box_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.box_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} box plots:\n" + 
                              "\n".join(plot_names))
        else:
            self.box_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No box plots were generated. Please check your data file.")

    def run_contour_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.contour_status.configure(text="🔄 Running contour plot analysis...")
        self.run_contour_btn.configure(state="disabled", text="Running...")
        
        thread = threading.Thread(target=self._run_contour_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_contour_analysis_thread(self):
        try:
            generated_files, error = ContourAnalysis.run_contour_analysis(self.data_file)
            self.after(0, self._contour_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._contour_analysis_complete, [], str(e))

    def _contour_analysis_complete(self, generated_files, error):
        self.run_contour_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.contour_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Contour plot analysis failed:\n{error}")
        elif generated_files:
            self.contour_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.contour_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} contour plots:\n" + 
                              "\n".join(plot_names))
        else:
            self.contour_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No contour plots were generated. Please check your data file.")

    def run_kmeans_analysis(self, mode):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        status = getattr(self, f'kmeans_{mode}_status')
        btn = getattr(self, f'run_kmeans_{mode}_btn')
        
        status.configure(text=f"🔄 Running K-means analysis ({mode})...")
        btn.configure(state="disabled", text="Running...")
        
        thread = threading.Thread(target=self._run_kmeans_analysis_thread, args=(mode,))
        thread.daemon = True
        thread.start()

    def _run_kmeans_analysis_thread(self, mode):
        try:
            generated_files, error = KMeansAnalysis.run_kmeans_analysis(self.data_file, mode=mode)
            self.after(0, self._kmeans_analysis_complete, generated_files, error, mode)
        except Exception as e:
            self.after(0, self._kmeans_analysis_complete, [], str(e), mode)

    def _kmeans_analysis_complete(self, generated_files, error, mode):
        btn = getattr(self, f'run_kmeans_{mode}_btn')
        status = getattr(self, f'kmeans_{mode}_status')
        viewer = getattr(self, f'kmeans_{mode}_image_viewer')
        
        btn.configure(state="normal", text="Run Analysis")
        
        if error:
            status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"K-means analysis ({mode}) failed:\n{error}")
        elif generated_files:
            status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} K-means plots ({mode}):\n" + 
                              "\n".join(plot_names))
        else:
            status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", f"No K-means plots were generated ({mode}). Please check your data file.")

    def run_dfa_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        self.dfa_status.configure(text="🔄 Running DFA analysis...")
        self.run_dfa_btn.configure(state="disabled", text="Running...")

        thread = threading.Thread(target=self._run_dfa_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_dfa_analysis_thread(self):
        try:
            generated_files, error = DFAAnalysis.run_dfa_analysis(self.data_file)
            self.after(0, self._dfa_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._dfa_analysis_complete, [], str(e))

    def _dfa_analysis_complete(self, generated_files, error):
        self.run_dfa_btn.configure(state="normal", text="Run Analysis")
        if error:
            self.dfa_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"DFA analysis failed:\n{error}")
        elif generated_files:
            self.dfa_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.dfa_image_viewer.load_images(generated_files)
            messagebox.showinfo("Analysis Complete", f"Generated DFA plot:\n{generated_files[0]}")
        else:
            self.dfa_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No DFA plot generated.")

    def run_psa_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        self.psa_status.configure(text="🔄 Running PSA analysis...")
        self.run_psa_btn.configure(state="disabled", text="Running...")
        thread = threading.Thread(target=self._run_psa_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_psa_analysis_thread(self):
        try:
            generated_files, error = PSAAnalysis.run_psa_analysis(self.data_file)
            self.after(0, self._psa_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._psa_analysis_complete, [], str(e))

    def _psa_analysis_complete(self, generated_files, error):
        self.run_psa_btn.configure(state="normal", text="Run Analysis")
        if error:
            self.psa_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"PSA analysis failed:\n{error}")
        elif generated_files:
            self.psa_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.psa_image_viewer.load_images(generated_files)
            messagebox.showinfo("Analysis Complete", f"Generated PSA plots:\n" + "\n".join(generated_files))
        else:
            self.psa_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No PSA plot generated.")

    def run_lstm_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.lstm_status.configure(text="🔄 Running LSTM analysis...")
        self.run_lstm_btn.configure(state="disabled", text="Running...")
        
        thread = threading.Thread(target=self._run_lstm_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_lstm_analysis_thread(self):
        try:
            generated_files, error = LSTMAnalysis.run_lstm_analysis(self.data_file)
            self.after(0, self._lstm_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._lstm_analysis_complete, [], str(e))

    def _lstm_analysis_complete(self, generated_files, error):
        self.run_lstm_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.lstm_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"LSTM analysis failed:\n{error}")
        elif generated_files:
            self.lstm_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.lstm_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} LSTM plots:\n" + 
                              "\n".join(plot_names))
        else:
            self.lstm_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No LSTM plots were generated. Please check your data file.")

    def run_prophet_analysis(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        self.prophet_status.configure(text="🔄 Running Prophet analysis...")
        self.run_prophet_btn.configure(state="disabled", text="Running...")
        
        thread = threading.Thread(target=self._run_prophet_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_prophet_analysis_thread(self):
        try:
            generated_files, error = ProphetAnalysis.run_prophet_analysis(self.data_file)
            self.after(0, self._prophet_analysis_complete, generated_files, error)
        except Exception as e:
            self.after(0, self._prophet_analysis_complete, [], str(e))

    def _prophet_analysis_complete(self, generated_files, error):
        self.run_prophet_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.prophet_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Prophet analysis failed:\n{error}")
        elif generated_files:
            self.prophet_status.configure(text=f"✅ Generated {len(generated_files)} plot(s)")
            self.prophet_image_viewer.load_images(generated_files)
            plot_names = [os.path.basename(f) for f in generated_files]
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully generated {len(generated_files)} Prophet plots:\n" + 
                              "\n".join(plot_names))
        else:
            self.prophet_status.configure(text="❌ No plots generated")
            messagebox.showwarning("No Results", "No Prophet plots were generated. Please check your data file.")

    def run_mass_estimation(self):
        if not self.data_file:
            messagebox.showwarning("Warning", "Please load a data file first!")
            return
        
        try:
            df = pd.read_csv(self.data_file)
            required_columns = ['methane3', 'longitude', 'latitude']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                messagebox.showerror("Error", f"Data file missing required columns: {', '.join(missing)}")
                return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read data file: {str(e)}")
            return
        
        self.mass_status.configure(text="🔄 Running mass estimation...")
        self.run_mass_btn.configure(state="disabled", text="Running...")
        
        thread = threading.Thread(target=self._run_mass_estimation_thread)
        thread.daemon = True
        thread.start()

    def _run_mass_estimation_thread(self):
        try:
            generated_files, total_mass, error = MassEstimation.run_mass_estimation(self.data_file)
            self.after(0, self._mass_estimation_complete, generated_files, total_mass, error)
        except Exception as e:
            self.after(0, self._mass_estimation_complete, [], None, str(e))

    def _mass_estimation_complete(self, generated_files, total_mass, error):
        self.run_mass_btn.configure(state="normal", text="Run Analysis")
        
        if error:
            self.mass_status.configure(text=f"❌ Error: {error}")
            messagebox.showerror("Analysis Error", f"Mass estimation failed:\n{error}")
        elif generated_files and total_mass is not None:
            self.mass_status.configure(text=f"✅ Total methane mass: {total_mass:.2f} tons")
            self.mass_image_viewer.load_images(generated_files)
            messagebox.showinfo("Analysis Complete", 
                              f"Successfully estimated total methane mass: {total_mass:.2f} tons\n"
                              f"Generated plot: {generated_files[0]}")
        else:
            self.mass_status.configure(text="❌ No results generated")
            messagebox.showwarning("No Results", "No results were generated. Please check your data file.")

if __name__ == "__main__":
    app = MethaneAnalysisApp()
    app.mainloop()