import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns


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

