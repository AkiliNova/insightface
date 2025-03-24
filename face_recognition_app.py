import gradio as gr
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import os
import json
from datetime import datetime

# Initialize face analyzer
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1, det_size=(640, 640))

# Dictionary to store authorized faces
AUTHORIZED_FACES_FILE = 'authorized_faces.json'
authorized_faces = {}

def load_authorized_faces():
    global authorized_faces
    if os.path.exists(AUTHORIZED_FACES_FILE):
        with open(AUTHORIZED_FACES_FILE, 'r') as f:
            authorized_faces = json.load(f)

def save_authorized_faces():
    with open(AUTHORIZED_FACES_FILE, 'w') as f:
        json.dump(authorized_faces, f)

def register_face(image, name):
    if image is None:
        return "Please provide an image"
    
    # Detect faces in the image
    faces = app.get(image)
    if len(faces) == 0:
        return "No face detected in the image"
    if len(faces) > 1:
        return "Please provide an image with only one face"
    
    # Get face embedding
    face = faces[0]
    embedding = face.embedding.tolist()
    
    # Store the face embedding with the name
    authorized_faces[name] = embedding
    save_authorized_faces()
    return f"Successfully registered face for {name}"

def process_image(image):
    if image is None:
        return None
    
    # Detect faces in the image
    faces = app.get(image)
    
    # Create a copy of the image for drawing
    result = image.copy()
    
    for face in faces:
        # Get face embedding
        embedding = face.embedding
        
        # Check if the face is authorized
        is_authorized = False
        for name, auth_embedding in authorized_faces.items():
            # Calculate similarity between embeddings
            similarity = np.dot(embedding, auth_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(auth_embedding))
            if similarity > 0.5:  # Threshold for face recognition
                is_authorized = True
                break
        
        # Draw rectangle with appropriate color
        color = (0, 255, 0) if is_authorized else (0, 0, 255)  # Green for authorized, Red for unauthorized
        bbox = face.bbox.astype(int)
        cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Add label
        label = "Authorized" if is_authorized else "Unauthorized"
        cv2.putText(result, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result

# Load existing authorized faces
load_authorized_faces()

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Face Recognition System")
    
    with gr.Tab("Register New Face"):
        with gr.Row():
            register_image = gr.Image(label="Upload face image")
            name_input = gr.Textbox(label="Enter name for the face")
        register_button = gr.Button("Register Face")
        register_output = gr.Textbox(label="Registration Status")
        
    with gr.Tab("Face Recognition"):
        with gr.Row():
            input_image = gr.Image(label="Upload image for recognition")
            output_image = gr.Image(label="Result")
        recognize_button = gr.Button("Recognize Faces")
    
    register_button.click(
        fn=register_face,
        inputs=[register_image, name_input],
        outputs=register_output
    )
    
    recognize_button.click(
        fn=process_image,
        inputs=input_image,
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch(share=True) 