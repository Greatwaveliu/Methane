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

