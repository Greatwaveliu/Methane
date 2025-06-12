import pandas as pd
import geopandas as gpd
import contextily as ctx
import matplotlib.pyplot as plt
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

