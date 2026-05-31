import gradio as gr
import cv2
import torch
import torch.nn as nn
import numpy as np

# 1. Define the Updated Model Architecture (V2)
class AgeGenderModel(nn.Module):
    def __init__(self):
        super(AgeGenderModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.age_output = nn.Linear(128, 1)
        self.gender_output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        age = self.age_output(x)
        # NO SIGMOID HERE ANYMORE (Matches the V2 training architecture)
        gender_logits = self.gender_output(x) 
        return age.squeeze(), gender_logits.squeeze()

# 2. Load the V2 Weights to CPU (Best for Web Servers)
model = AgeGenderModel()
# Make sure this matches the EXACT file name you saved in your Jupyter Notebook!
model.load_state_dict(torch.load('best_age_gender_model_v2.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. The Prediction Pipeline
def predict_age_gender(image):
    # SAFETY CHECK: Ignore empty or glitched frames from the webcam
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return image 
        
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # High sensitivity face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Add a 40% padding margin so the AI sees your whole head
            margin_y = int(h * 0.4)
            margin_x = int(w * 0.4)
            
            # Ensure the crop doesn't break image boundaries
            y1 = max(0, y - margin_y)
            y2 = min(image.shape[0], y + h + margin_y)
            x1 = max(0, x - margin_x)
            x2 = min(image.shape[1], x + w + margin_x)
            
            # Crop and resize
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                continue 
                
            face_resized = cv2.resize(face_crop, (128, 128))
            
            # Prepare for PyTorch Tensor (Channels, Height, Width)
            face_tensor = face_resized.transpose((2, 0, 1)) / 255.0
            face_tensor = torch.tensor(face_tensor, dtype=torch.float32).unsqueeze(0)
            
            # Predict Age and Gender Logits
            with torch.no_grad():
                pred_age, pred_gender_logits = model(face_tensor)
                
            # Age Prediction (Raw output, no offset)
            age = int(pred_age.item())
            
            # --- THE CRITICAL FIX: MANUALLY APPLY SIGMOID ---
            prob = torch.sigmoid(pred_gender_logits).item()
            
            # Gender Classification & Confidence Score
            if prob > 0.5:
                gender = "Female"
                confidence = int(prob * 100)
            else:
                gender = "Male"
                confidence = int((1.0 - prob) * 100)
            
            # Draw the final bounding box and label on the image
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Label format: "Male (88%), 24 yrs"
            label = f"{gender} ({confidence}%), {age} yrs"
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    except Exception as e:
        # Keep the server alive if a math/frame error happens
        print(f"DEBUG - Skipped a frame. Error: {e}")
        
    return image

# 4. Build and Launch the Gradio Web App
iface = gr.Interface(
    fn=predict_age_gender, 
    inputs=gr.Image(sources=["webcam"]), 
    outputs="image",
    title="Real-Time Age & Gender Predictor (V2)",
    description="Running the optimized 30-epoch model! Allow camera access, ensure good lighting, and snap a picture."
)

if __name__ == "__main__":
    iface.launch(share=True)