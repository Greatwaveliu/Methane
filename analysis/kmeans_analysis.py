import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import contextily as ctx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
class KMeansAnalysis:
    """K-means clustering analysis functionality"""
    
    @staticmethod
    def cargar_datos(ruta_archivo, include_time=False, include_methane=False):
        columns = ['latitude', 'longitude']
        if include_time:
            columns.append('measurement_time')

        # Always attempt to load methane3 so that histogram plots can be
        # generated even when methane is not part of the clustering features.
        optional_cols = ['methane3']

        df = pd.read_csv(
            ruta_archivo,
            usecols=lambda c: c in columns + optional_cols,
            parse_dates=['measurement_time'] if include_time else None,
        )
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
            
            silhouette_arr = np.array(silhouette_list)
            db_index_arr = np.array(db_index_list)
            best_k_candidates = np.where(
                silhouette_arr == silhouette_arr.max()
            )[0]
            if len(best_k_candidates) > 1:
                best_k = K_range[np.argmin(db_index_arr[best_k_candidates])]
            else:
                best_k = K_range[best_k_candidates[0]]
            
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
            df[cluster_column] = kmeans.fit_predict(features_scaled)
            
            generated_files = []

            if mode not in ['lat_lon', 'time_lat_lon', 'lat_lon_methane', 'time_lat_lon_methane']:
                output_csv = f'tula_kmeans_{mode}_clusters.csv'
                df.to_csv(output_csv, index=False)
                generated_files.append(output_csv)

            conteos = df[cluster_column].value_counts().sort_index()
            conteo_str = ", ".join(
                f"{cl}: {cnt}" for cl, cnt in conteos.items()
            )

            if mode not in ['lat_lon', 'time_lat_lon', 'lat_lon_methane', 'time_lat_lon_methane']:
                conteo_file = f'tula_kmeans_{mode}_conteos.txt'
                with open(conteo_file, 'w') as f:
                    f.write(
                        f"Cantidad de puntos por cluster (K-Means {mode}):\n"
                    )
                    f.write(str(conteos))
                generated_files.append(conteo_file)
            
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            best_idx = best_k - K_range.start

            axes[0].plot(K_range, wcss_list, marker='o')
            axes[0].axvline(best_k, color='purple', linestyle='--')
            axes[0].scatter(best_k, wcss_list[best_idx], color='purple', zorder=5)
            axes[0].annotate(
                f"K = {best_k}",
                xy=(best_k, wcss_list[best_idx]),
                xytext=(8, 0),
                textcoords='offset points',
                ha='left',
                va='center',
                color='purple'
            )
            axes[0].set_title('WCSS vs K')
            axes[0].set_xlabel('Número de clusters (K)')
            axes[0].set_ylabel('WCSS')
            axes[0].grid(True, linestyle='--', alpha=0.3)

            axes[1].plot(K_range, silhouette_list, marker='o', color='green')
            axes[1].axvline(best_k, color='purple', linestyle='--')
            axes[1].scatter(best_k, silhouette_list[best_idx], color='purple', zorder=5)
            axes[1].annotate(
                f"K = {best_k}",
                xy=(best_k, silhouette_list[best_idx]),
                xytext=(8, 0),
                textcoords='offset points',
                ha='left',
                va='center',
                color='purple'
            )
            axes[1].set_title('Silhouette Score vs K')
            axes[1].set_xlabel('Número de clusters (K)')
            axes[1].set_ylabel('Silhouette Score')
            axes[1].grid(True, linestyle='--', alpha=0.3)

            axes[2].plot(K_range, db_index_list, marker='o', color='red')
            axes[2].axvline(best_k, color='purple', linestyle='--')
            axes[2].scatter(best_k, db_index_list[best_idx], color='purple', zorder=5)
            axes[2].annotate(
                f"K = {best_k}",
                xy=(best_k, db_index_list[best_idx]),
                xytext=(8, 0),
                textcoords='offset points',
                ha='left',
                va='center',
                color='purple'
            )
            axes[2].set_title('Davies-Bouldin Index vs K')
            axes[2].set_xlabel('Número de clusters (K)')
            axes[2].set_ylabel('DB Index')
            axes[2].grid(True, linestyle='--', alpha=0.3)

            fig.tight_layout()
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
            title = f'Distribución espacial por cluster (K-Means, K={best_k}) - {mode}'
            if conteo_str:
                title += f"\nConteos: {conteo_str}"
            ax.set_title(title)
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
            
            has_methane = 'methane3' in df.columns

            if has_methane:
                if include_methane and mode in ['lat_lon_methane', 'time_lat_lon_methane']:
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
                    plt.title(
                        f'Distribución combinada de methane3 por cluster (K={best_k}) - {mode}'
                    )
                    plt.xlabel('Cluster')
                    plt.ylabel('methane3 (ppb)')
                    plt.grid(axis='y', linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    swarm_file = f'tula_methane3_box_swarmplot_{mode}_cluster.png'
                    plt.savefig(swarm_file, dpi=300)
                    plt.close()
                    generated_files.append(swarm_file)
                elif not include_methane:
                    if mode not in ['lat_lon', 'time_lat_lon']:
                        plt.figure(figsize=(10, 6))
                        sns.boxplot(
                            data=df,
                            x=cluster_column,
                            y='methane3',
                            palette='tab10'
                        )
                        plt.title(
                            f'Distribución de methane3 por cluster (K={best_k}) - {mode}'
                        )
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
                    plt.title(
                        f'Distribución combinada de methane3 por cluster (K={best_k}) - {mode}'
                    )
                    plt.xlabel('Cluster')
                    plt.ylabel('methane3 (ppb)')
                    plt.grid(axis='y', linestyle='--', alpha=0.3)
                    plt.tight_layout()
                    swarm_file = f'tula_methane3_box_swarmplot_{mode}_cluster.png'
                    plt.savefig(swarm_file, dpi=300)
                    plt.close()
                    generated_files.append(swarm_file)

                    if mode not in ['lat_lon', 'time_lat_lon']:
                        resumen = (
                            df.groupby(cluster_column)['methane3'].describe().round(2)
                        )
                        resumen.reset_index(inplace=True)
                        resumen.columns.name = None
                        resumen.to_csv(
                            f'tula_methane3_resumen_por_cluster_{mode}.csv', index=False
                        )
                        generated_files.append(
                            f'tula_methane3_resumen_por_cluster_{mode}.csv'
                        )

                # Stacked histogram of methane3 by cluster (common for all modes)
                clusters = sorted(df[cluster_column].unique())
                data_arrays = [df[df[cluster_column] == cl]['methane3'] for cl in clusters]
                plt.figure(figsize=(10, 6))
                colors = sns.color_palette('tab10', len(clusters))
                plt.hist(
                    data_arrays,
                    bins=30,
                    stacked=True,
                    label=[f'Cluster {cl}' for cl in clusters],
                    color=colors,
                    edgecolor='black'
                )
                plt.title(
                    f'Distribución apilada de methane3 por cluster (K={best_k}) - {mode}'
                )
                plt.xlabel('methane3 (ppb)')
                plt.ylabel('Cantidad de puntos')
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.3)
                plt.tight_layout()
                stacked_hist_file = (
                    f'tula_methane3_stacked_hist_{mode}_cluster.png'
                )
                plt.savefig(stacked_hist_file, dpi=300)
                plt.close()
                generated_files.append(stacked_hist_file)
            
            return generated_files, None
            
        except Exception as e:
            error_msg = f"Error en análisis K-means ({mode}): {str(e)}"
            return [], error_msg

