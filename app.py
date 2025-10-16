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
    .recommendation-box {
        background-color: #fffbeb;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
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
            "Ensure adequate sunlight exposure"
        ]
    },
    "diseased": {
        "message": "⚠️ Disease detected in the leaf!",
        "recommendations": [
            "Isolate affected plants to prevent spread",
            "Remove and dispose of infected leaves properly",
            "Consider applying appropriate fungicide or pesticide",
            "Improve air circulation around plants",
            "Avoid overhead watering to reduce moisture on leaves",
            "Consult with a local agricultural expert for specific treatment"
        ]
    },
    "default": {
        "message": "🔍 Analysis complete",
        "recommendations": [
            "Review the detection results carefully",
            "Compare with known disease patterns",
            "Consider taking additional images for verification",
            "Consult with agricultural experts if uncertain"
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
    """Determine health status from class name"""
    class_lower = class_name.lower()
    if "healthy" in class_lower or "normal" in class_lower:
        return "healthy"
    elif "disease" in class_lower or "infected" in class_lower or "blight" in class_lower or "spot" in class_lower:
        return "diseased"
    else:
        return "default"

def run_inference(model, image):
    """Run YOLOv8 inference on the image"""
    try:
        results = model(image)
        return results[0]
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return None

def display_recommendations(health_status):
    """Display health recommendations based on status"""
    rec_data = HEALTH_RECOMMENDATIONS.get(health_status, HEALTH_RECOMMENDATIONS["default"])
    
    st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
    st.markdown(f"### {rec_data['message']}")
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
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="🏷️ Predicted Class",
                        value=class_name.title()
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(
                        label="📈 Confidence",
                        value=f"{confidence:.2%}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with metric_col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    status_emoji = "✅" if health_status == "healthy" else "⚠️"
                    st.metric(
                        label="🩺 Health Status",
                        value=f"{status_emoji} {health_status.title()}"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Display all detections if multiple
                if len(result.boxes) > 1:
                    with st.expander("📋 View All Detections"):
                        for idx, (cls, conf) in enumerate(zip(classes, confidences)):
                            st.write(f"**Detection {idx+1}:** {result.names[int(cls)]} ({conf:.2%} confidence)")
                
                # Display recommendations
                st.divider()
                st.header("💡 Health Recommendations")
                display_recommendations(health_status)
                
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
            """)

if __name__ == "__main__":
    main()
