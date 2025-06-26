from flask import Flask, request, jsonify, send_file
from functools import wraps
import os
import io
from PIL import Image
import base64
import json
from datetime import datetime
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import easyocr
import google.generativeai as genai
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import shutil
from difflib import SequenceMatcher

# Import functions from main app
from app import (
    compute_embedding, compare_images, convert_to_color_binary, 
    reverse_color_binary, extract_text_from_image_easyocr,
    User, db, init_db
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-api-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# API Keys storage (in production, use a proper database)
API_KEYS = {"your_api_key_here"}

# Initialize models
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
easyocr_reader = easyocr.Reader(['en'])

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-Key') not in API_KEYS:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    })

@app.route('/api/compare', methods=['POST'])
@require_api_key
def compare():
    data = request.get_json()
    image1 = Image.open(io.BytesIO(base64.b64decode(data['image1'])))
    image2 = Image.open(io.BytesIO(base64.b64decode(data['image2'])))
    threshold = float(data.get('threshold', 70.0))
    similarity, result = compare_images(image1, image2, threshold)
    return jsonify({"similarity": similarity, "result": result})

@app.route('/api/embedding', methods=['POST'])
@require_api_key
def embedding():
    data = request.get_json()
    image = Image.open(io.BytesIO(base64.b64decode(data['image'])))
    embedding = compute_embedding(image)
    return jsonify({"embedding": embedding.tolist()})

@app.route('/api/convert_to_binary', methods=['POST'])
@require_api_key
def convert_to_binary():
    data = request.get_json()
    image = Image.open(io.BytesIO(base64.b64decode(data['image'])))
    out_file = f"temp_binary_{datetime.utcnow().timestamp()}.txt"
    convert_to_color_binary(image, out_file)
    with open(out_file, 'r') as f:
        binary_content = f.read()
    return jsonify({"binary": binary_content})

@app.route('/api/reverse_binary', methods=['POST'])
@require_api_key
def api_reverse_binary():
    """Reverse binary content back to image"""
    try:
        data = request.get_json()
        
        if not data or 'binary_content' not in data:
            return jsonify({"error": "Missing binary_content in request body"}), 400
        
        # Create temporary file
        temp_binary = f"temp_binary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
        temp_output = f"temp_output_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        
        try:
            # Save binary content
            with open(temp_binary, 'w') as f:
                f.write(data['binary_content'])
            
            # Reverse binary
            reverse_color_binary(temp_binary, temp_output)
            
            # Convert output image to base64
            if os.path.exists(temp_output):
                with open(temp_output, 'rb') as f:
                    image_data = f.read()
                image_base64 = base64.b64encode(image_data).decode()
                
                return jsonify({
                    "image": image_base64,
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                return jsonify({"error": "Failed to generate output image"}), 500
                
        finally:
            # Clean up temporary files
            if os.path.exists(temp_binary):
                os.remove(temp_binary)
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/extract_text', methods=['POST'])
@require_api_key
def api_extract_text():
    """Extract text from image using OCR"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image in request body"}), 400
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        
        # Save temporary image
        temp_image = f"temp_image_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png"
        
        try:
            image.save(temp_image)
            
            # Extract text
            extracted_text = extract_text_from_image_easyocr(temp_image)
            
            return jsonify({
                "extracted_text": extracted_text,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image):
                os.remove(temp_image)
                
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/ai_chat', methods=['POST'])
@require_api_key
def api_ai_chat():
    """AI chatbot endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({"error": "Missing message in request body"}), 400
        
        user_message = data['message']
        
        # Configure Google Generative AI
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY', 'your-google-api-key'))
        model = genai.GenerativeModel('gemini-pro')
        
        # Create context for the AI
        context = f"""
        You are an AI assistant for an image processing application. The user is asking: {user_message}
        
        Available features:
        - Image similarity comparison using CLIP model
        - Color binary conversion and reversal
        - OCR text extraction from images
        - AI-powered image analysis
        
        Please provide helpful and relevant responses about these features.
        """
        
        # Generate response
        response = model.generate_content(context)
        
        return jsonify({
            "response": response.text,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/register_key', methods=['POST'])
def register_api_key():
    """Register a new API key (admin only)"""
    try:
        data = request.get_json()
        
        if not data or 'admin_key' not in data or 'new_key' not in data:
            return jsonify({"error": "Missing admin_key or new_key in request body"}), 400
        
        # Simple admin key check (in production, use proper authentication)
        if data['admin_key'] != 'admin_secret_123':
            return jsonify({"error": "Invalid admin key"}), 401
        
        new_key = data['new_key']
        user_id = data.get('user_id', len(API_KEYS) + 1)
        
        API_KEYS.add(new_key)
        
        return jsonify({
            "message": "API key registered successfully",
            "key": new_key,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    with app.app_context():
        init_db()
    app.run(debug=True, host='0.0.0.0', port=5001) 