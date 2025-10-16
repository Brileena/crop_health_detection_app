import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os

# Page configuration
st.set_page_config(
    page_title="Crop Leaf Health Detection",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f9ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E7D32;
        margin: 1rem 0;
    }
    .metric-card-warning {
        background-color: #fef2f2;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #dc2626;
        margin: 1rem 0;
    }
    .recommendation-box {
        background-color: #fffbeb;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
    .recommendation-box-danger {
        background-color: #fef2f2;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #dc2626;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Health recommendations database
HEALTH_RECOMMENDATIONS = {
    "healthy": {
        "message": "✅ Your crop leaf appears healthy!",
        "recommendations": [
            "Continue with current care practices",
            "Maintain regular watering schedule",
            "Monitor for any changes in color or texture",
            "Ensure adequate sunlight exposure",
            "Keep up with preventive measures"
        ]
    },
    "diseased": {
        "message": "⚠️ Disease detected in the leaf!",
        "recommendations": [
            "Isolate affected plants immediately to prevent spread",
            "Remove and dispose of infected leaves properly (do not compost)",
            "Apply appropriate fungicide or pesticide based on disease type",
            "Improve air circulation around plants",
            "Avoid overhead watering to reduce moisture on leaves",
            "Reduce humidity if growing indoors",
            "Sanitize gardening tools between uses",
            "Consult with a local agricultural expert for specific treatment plan"
        ]
    },
    "unknown": {
        "message": "🔍 Analysis complete - Please verify results",
        "recommendations": [
            "Review the detection results carefully",
            "Compare with known disease reference images",
            "Consider taking additional images from different angles",
            "Check for visible symptoms like spots, discoloration, or wilting",
            "Consult with agricultural experts if uncertain",
            "Monitor the plant closely over the next few days"
        ]
    }
}

@st.cache_resource
def load_model(model_path):
    """Load YOLOv8 model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_health_status(class_name):
    """
    Determine health status from class name.
    Returns 'healthy' only if explicitly healthy, otherwise 'diseased'
    """
    class_lower = class_name.lower()
    
    # List of keywords that indicate healthy leaves
    healthy_keywords = ['healthy', 'normal', 'good']
    
    # List of keywords that indicate diseased leaves
    disease_keywords = [
        'disease', 'diseased', 'infected', 'infection',
        'blight', 'spot', 'rust', 'mold', 'mildew',
        'wilt', 'rot', 'scab', 'canker', 'bacterial',
        'fungal', 'viral', 'mosaic', 'leaf_curl',
        'septoria', 'anthracnose', 'powdery', 'downy',
        'early_blight', 'late_blight', 'target_spot',
        'leaf_mold', 'spider_mites', 'yellow', 'brown',
        'black', 'dead', 'dying', 'unhealthy', 'sick'
    ]
    
    # Check for healthy keywords first
    if any(keyword in class_lower for keyword in healthy_keywords):
        # Make sure it's not a false positive like "unhealthy"
        if not any(disease_keyword in class_lower for disease_keyword in disease_keywords):
            return "healthy"
    
    # Check for disease keywords
    if any(keyword in class_lower for keyword in disease_keywords):
        return "diseased"
    
    # If class name doesn't match any pattern, default to unknown
    # This is safer than assuming healthy
    return "unknown"

def run_inference(model, image):
    """Run YOLOv8 inference on the image"""
    try:
        results = model(image)
        return results[0]
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return None

def display_recommendations(health_status, class_name):
    """Display health recommendations based on status"""
    rec_data = HEALTH_RECOMMENDATIONS.get(health_status, HEALTH_RECOMMENDATIONS["unknown"])
    
    box_class = "recommendation-box-danger" if health_status == "diseased" else "recommendation-box"
    
    st.markdown(f'<div class="{box_class}">', unsafe_allow_html=True)
    st.markdown(f"### {rec_data['message']}")
    
    if health_status == "diseased":
        st.markdown(f"**Detected Condition:** {class_name}")
    
    st.markdown("**Recommendations:**")
    for rec in rec_data['recommendations']:
        st.markdown(f"- {rec}")
    st.markdown('</div>', unsafe_allow_html=True)

# Main app
def main():
    # Header
    st.markdown('<p class="main-header">🌿 Crop Leaf Health Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload or capture a leaf image for AI-powered health analysis</p>', unsafe_allow_html=True)
    
    # Sidebar for model info
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This app uses YOLOv8 to detect and classify crop leaf health conditions.")
        st.write("**Model:** best.pt")
        st.write("**Capabilities:**")
        st.write("- Healthy/Diseased classification")
        st.write("- Real-time detection")
        st.write("- Confidence scoring")
        st.divider()
        st.write("**Instructions:**")
        st.write("1. Choose input method")
        st.write("2. Upload/capture image")
        st.write("3. View results & recommendations")
        st.divider()
        
        # Debug mode toggle
        debug_mode = st.checkbox("🔧 Debug Mode", help="Show all model classes and detection details")
    
    # Check if model exists
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file '{model_path}' not found!")
        st.info("Please ensure 'best.pt' is in the same directory as app.py")
        st.stop()
    
    # Load model
    with st.spinner("Loading YOLOv8 model..."):
        model = load_model(model_path)
    
    if model is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Show model classes in debug mode
    if debug_mode:
        with st.sidebar:
            st.subheader("📋 Model Classes")
            if hasattr(model, 'names'):
                for idx, name in model.names.items():
                    health = get_health_status(name)
                    emoji = "✅" if health == "healthy" else "⚠️" if health == "diseased" else "❓"
                    st.text(f"{emoji} {idx}: {name}")
    
    # Input method selection
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a leaf image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of a crop leaf"
        )
    
    with col2:
        st.subheader("📷 Capture Image")
        camera_image = st.camera_input(
            "Take a picture",
            help="Use your camera to capture a leaf image"
        )
    
    # Process image
    input_image = None
    source_type = None
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
        source_type = "Uploaded"
    elif camera_image is not None:
        input_image = Image.open(camera_image)
        source_type = "Captured"
    
    if input_image is not None:
        st.divider()
        st.header("📊 Analysis Results")
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(input_image, use_container_width=True)
            st.caption(f"Source: {source_type} Image")
        
        # Run inference
        with st.spinner("🔍 Analyzing leaf health..."):
            result = run_inference(model, input_image)
        
        if result is not None:
            with col2:
                st.subheader("Detection Results")
                
                # Get annotated image
                annotated_img = result.plot()
                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                st.image(annotated_img_rgb, use_container_width=True)
                st.caption("Annotated with detections")
            
            # Extract predictions
            if len(result.boxes) > 0:
                # Get the detection with highest confidence
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                max_conf_idx = np.argmax(confidences)
                predicted_class_idx = int(classes[max_conf_idx])
                confidence = float(confidences[max_conf_idx])
                
                # Get class name
                class_name = result.names[predicted_class_idx]
                health_status = get_health_status(class_name)
                
                # Display metrics
                st.divider()
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    card_class = "metric-card" if health_status != "diseased" else "metric-card-warning"
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    st.metric(
                        label="🏷️ Predicted Class",
                        value=class_name.replace('_', ' ').title()
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    conf_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
                    st.metric(
                        label=f"{conf_color} Confidence",
                        value=f"{confidence:.2%}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col3:
                    card_class = "metric-card" if health_status != "diseased" else "metric-card-warning"
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    if health_status == "healthy":
                        status_emoji = "✅"
                        status_text = "Healthy"
                    elif health_status == "diseased":
                        status_emoji = "⚠️"
                        status_text = "Diseased"
                    else:
                        status_emoji = "❓"
                        status_text = "Unknown"
                    
                    st.metric(
                        label="🩺 Health Status",
                        value=f"{status_emoji} {status_text}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Debug information
                if debug_mode:
                    st.info(f"**Debug Info:** Class '{class_name}' → Status '{health_status}'")
                
                # Display all detections if multiple
                if len(result.boxes) > 1:
                    with st.expander("📋 View All Detections"):
                        for idx, (cls, conf) in enumerate(zip(classes, confidences)):
                            det_class = result.names[int(cls)]
                            det_health = get_health_status(det_class)
                            det_emoji = "✅" if det_health == "healthy" else "⚠️" if det_health == "diseased" else "❓"
                            st.write(f"**Detection {idx+1}:** {det_emoji} {det_class} ({conf:.2%} confidence) - {det_health}")
                
                # Display recommendations
                st.divider()
                st.header("💡 Health Recommendations")
                display_recommendations(health_status, class_name)
                
                # Additional warnings for low confidence
                if confidence < 0.5:
                    st.warning("⚠️ **Low Confidence Detection**: The model is uncertain about this classification. Consider taking another image with better lighting or angle.")
                
            else:
                st.warning("⚠️ No leaf detected in the image. Please try with a clearer image of a crop leaf.")
                st.info("**Tips for better results:**")
                st.write("- Ensure good lighting")
                st.write("- Focus on the leaf area")
                st.write("- Avoid blurry images")
                st.write("- Keep background simple")
    
    else:
        # Display placeholder
        st.info("👆 Please upload an image or capture one using your camera to begin analysis.")
        
        # Example section
        with st.expander("📝 Tips for Best Results"):
            st.write("""
            **For Accurate Detection:**
            - Use clear, well-lit images
            - Focus on individual leaves when possible
            - Capture the entire leaf in frame
            - Avoid excessive shadows or glare
            - Use a plain background if possible
            
            **Common Issues:**
            - Blurry images may reduce accuracy
            - Multiple leaves may confuse detection
            - Poor lighting can affect classification
            
            **Understanding Results:**
            - Green checkmark (✅) indicates healthy leaves
            - Warning sign (⚠️) indicates diseased or problematic leaves
            - Confidence score shows model certainty (higher is better)
            """)

if __name__ == "__main__":
    main()
