import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

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

