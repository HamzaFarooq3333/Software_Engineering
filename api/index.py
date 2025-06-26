from flask import Flask, request, jsonify
import os
import base64
import io
from PIL import Image
from datetime import datetime

app = Flask(__name__)

# API Key from environment variable
API_KEY = os.environ.get('MY_API_KEY', 'sk-2f8e1b7c-4e2a-4b8e-9c1d-3e2f1a7b6c5d')

def require_api_key(view_function):
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-Key') != API_KEY:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return view_function(*args, **kwargs)
    decorated_function.__name__ = view_function.__name__
    return decorated_function

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "message": "Flask API is running successfully on Vercel"
    })

@app.route('/api/echo', methods=['POST'])
@require_api_key
def echo():
    """Simple echo endpoint for testing"""
    data = request.get_json()
    return jsonify({
        "message": "Echo successful",
        "data": data,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/api/image_info', methods=['POST'])
@require_api_key
def image_info():
    """Get basic image information without heavy processing"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image in request body"}), 400
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        
        # Get basic image info
        width, height = image.size
        mode = image.mode
        format_info = image.format
        
        return jsonify({
            "width": width,
            "height": height,
            "mode": mode,
            "format": format_info,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/convert_image', methods=['POST'])
@require_api_key
def convert_image():
    """Convert image format (lightweight operation)"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image in request body"}), 400
        
        target_format = data.get('format', 'PNG')
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        
        # Convert to target format
        buffer = io.BytesIO()
        image.save(buffer, format=target_format)
        converted_image = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            "converted_image": converted_image,
            "format": target_format,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/api/resize_image', methods=['POST'])
@require_api_key
def resize_image():
    """Resize image (lightweight operation)"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "Missing image in request body"}), 400
        
        width = data.get('width', 300)
        height = data.get('height', 300)
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        
        # Resize image
        resized_image = image.resize((width, height))
        
        # Convert back to base64
        buffer = io.BytesIO()
        resized_image.save(buffer, format='PNG')
        resized_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            "resized_image": resized_base64,
            "new_width": width,
            "new_height": height,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# Vercel serverless function handler
def handler(environ, start_response):
    return app(environ, start_response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 