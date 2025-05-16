 🧠 Depth Map Estimator API

This project is a FastAPI-based web service that generates **depth maps** and **3D surface visualizations** from input images using the MiDaS deep learning model. Upload an image and receive both a 2D colored depth map and a 3D Plotly surface plot.



 🚀 Features

  - 📷 Upload an image and get depth estimation in real-time.
  - 🌌 2D colored depth map using OpenCV colormaps.
  - 🗻 Interactive 3D surface visualization powered by Plotly.
  - ⚡ Powered by MiDaS v3 (`DPT_Large`) for high-accuracy depth prediction.
  - 🔁 RESTful API built with FastAPI.
  - 🌐 CORS enabled for easy frontend integration.



 📁 Project Structure

 depth-estimation-app/
├── backend/
│   ├── app.py (FastAPI or Flask backend)
│   ├── model_loader.py (loads MiDaS)
│   ├── requirements.txt
│   └── static/ (for output images)
├── frontend/
│   └── index.html / React App
└── README.md

## 🛠️ Installation & Setup

### 1. Clone the Repository

git clone https://github.com/yourusername/depth-map-api.git
cd depth-map-api

2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt
If requirements.txt is not available, manually install:

pip install fastapi uvicorn torch opencv-python-headless plotly pillow
▶️ Run the API Server

uvicorn app:app --reload
Visit: http://127.0.0.1:8000/docs for Swagger UI.

📫 API Usage
Endpoint: POST /predict
Request Type: multipart/form-data

Parameter: file (image file)

Response: JSON with:

depth_map: Base64-encoded PNG image

plot_html: HTML string for 3D depth surface

Example with curl

curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@input.jpg"
🧰 Optional Utility
model_loader.py contains helper functions to load the model and run inference separately if needed.

🧠 Credits
Intel ISL MiDaS

FastAPI, OpenCV, Plotly, PyTorch

📄 License
MIT License


