import torch
import cv2
import numpy as np

# Load MiDaS model and transforms
def load_model():
    model_type = "DPT_Large"  # alternatives: "DPT_Hybrid", "MiDaS_small"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
    return model, transform

def estimate_depth(model, transform, image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_tensor = transform(img_rgb).unsqueeze(0)

    with torch.no_grad():
        prediction = model(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = depth_map.astype(np.uint8)

    return depth_map
