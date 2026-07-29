# python -m uvicorn AQI_fastAPI:app --reload
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import pennylane as qml 
import torch.nn as nn
import joblib
import numpy as np

n_qubits = 7
dev = qml.device("lightning.qubit", wires=n_qubits)
@qml.qnode(dev, interface="torch", diff_method="adjoint")

def q_ckt(inputs,weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.clayer_in = nn.Sequential(
            nn.Linear(12,32),
            nn.ReLU(),
            nn.Linear(32,7),    # 12 se waps 7 
            nn.Tanh()    # Tanh values [-1,1] for quantum angleEmbedding
            
        )
        self.q_weights = nn.Parameter(torch.randn(3, 7))
        self.clayer_out = nn.Sequential(
            nn.Linear(7,16),
            nn.ReLU(),
            nn.Linear(16,1)   # final single aqi prediction
            
        )

    def forward(self, x):
        # Data flows: Classical Input -> Relu -> Tanh
        x = self.clayer_in(x)
        q_out = q_ckt(x, self.q_weights)

        if isinstance(q_out, (tuple, list)):
            x = torch.stack([tensor for tensor in q_out], dim=1)
        else:
            x = q_out
        x = x.to(torch.float32)
        
        x = self.clayer_out(x)   # final prediction
        return x
    
model = HybridModel()
model.load_state_dict(torch.load('aqi_qml_model.pth', map_location=torch.device('cpu')))
scaler = joblib.load('scaler.pkl')
model.eval()   # testing mode on

# ------ FASTAPI--------
app = FastAPI(title="Quantum AQI Predictor")

class UserInput(BaseModel):      # menu card h
    latitude: float
    longitude: float
    is_weekend: int
    crop_burning_season: int
    season_name: str         # summer,winter,monsoon,post_monsoon
    

@app.get("/health")
def health_check():
    return {"status": "active", "message": "Server is running perfectly."}

# prediction (api route)
@app.post("/predict_aqi")
def predict(data: UserInput):
    try:
        season = data.season_name.lower()
        season_post_monsoon = 1.0 if season == "post_monsoon" else 0.0
        season_summer = 1.0 if season == "summer" else 0.0
        season_winter = 1.0 if season == "winter" else 0.0
        
        # Step B: Chemical & Dust Features (Dataset ke average/default values)
        # (Image ke data ke aas-paas ka average lagaya hai)
        default_no2 = 16.0
        default_so2 = 13.4
        default_o3 = 80.0
        default_dust = 19.7
        default_aod = 0.5
        
        # Step C: Final 12 features ki list taiyaar karna (EXACT dataframe ke order mein)
        final_12_features = [
            default_no2,                  # 1. no2_ugm3
            default_so2,                  # 2. so2_ugm3
            default_o3,                   # 3. o3_ugm3
            default_dust,                 # 4. dust_ugm3
            float(data.crop_burning_season), # 5. crop_burning_season
            default_aod,                  # 6. aod
            data.latitude,                # 7. latitude
            data.longitude,               # 8. longitude
            float(data.is_weekend),       # 9. is_weekend
            season_post_monsoon,          # 10. season_post_monsoon
            season_summer,                # 11. season_summer
            season_winter                 # 12. season_winter
        ] 
        
        
        # 1. Pehle list ko 2D numpy array mein convert karein (1 sample, 12 features)
        features_array = np.array(final_12_features).reshape(1, -1)
        
        # 2. Scaler use karke data ko transform karein
        scaled_features = scaler.transform(features_array)
        
        # 3. Scaled data ko wapas PyTorch tensor mein convert karein
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

        with torch.no_grad():
            log_pred = model(input_tensor)
            predicted_aqi = torch.expm1(log_pred)
            
        return {
            "status": "success",
            "message": "Inference Complete",
            "predicted_aqi": round(predicted_aqi.item(), 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))   