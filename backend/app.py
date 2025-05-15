from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import plotly.graph_objects as go
import base64
from io import BytesIO
from PIL import Image
import torch
import cv2
import numpy as np
import io

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load MiDaS model
model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        print("Image read successfully")
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print("Image decoded successfully")

        # Apply transforms
        try:
            input_batch = transform(img).to(device)
            print("Transform applied successfully")
        except Exception as e:
            print(f"Transform error: {e}")
            raise

        # Prediction
        try:
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            print("Prediction completed successfully")
        except Exception as e:
            print(f"Prediction error: {e}")
            raise

        depth_map = prediction.cpu().numpy()
        print("Converted to numpy array")
        
        # Normalize depth map
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
        depth_map = depth_map.astype(np.uint8)
        print("Normalization completed")
        
        # Convert to colored depth map
        # After generating depth_map, before applying color map
        # Downsample the depth map for 3D visualization
        scale_factor = 4  # Adjust this value to balance detail vs speed
        downsampled_depth = depth_map[::scale_factor, ::scale_factor]
        y, x = np.mgrid[0:downsampled_depth.shape[0]:1, 0:downsampled_depth.shape[1]:1]
        
        # Create 3D plot with optimized settings
        fig = go.Figure(data=[go.Surface(z=downsampled_depth, x=x, y=y)])
        fig.update_layout(
            title='Depth Map 3D Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Depth',
            ),
            width=600,  # Reduced size
            height=600
        )
        
        # Convert plot to HTML
        plot_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        
        # Continue with color map conversion
        depth_map_colored = cv2.applyColorMap(depth_map.astype(np.uint8), cv2.COLORMAP_INFERNO)
        
        # Encode both images
        _, buffer = cv2.imencode(".png", depth_map_colored)
        depth_map_base64 = base64.b64encode(buffer).decode()

        return JSONResponse({
            "depth_map": f"data:image/png;base64,{depth_map_base64}",
            "plot_html": plot_html
        })
    
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        raise
