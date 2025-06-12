import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
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

