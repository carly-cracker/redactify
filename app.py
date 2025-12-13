import streamlit as st              
import tempfile                     
import os                           
import time                         

# CRITICAL FIX: Import the main processing function from your local file
from redact_video import process_video 

# Configure basic settings for the Streamlit page
st.set_page_config(
    page_title="Video Redaction (YOLO)",
    layout="centered",
    page_icon="🎥"
)

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style="text-align:center;">Video Redaction Tool</h1>
    <p style="text-align:center; color: gray;">
        Automatically detect and redact objects using YOLO
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------- FILE UPLOAD ----------
st.markdown("## 1. Upload Video")
uploaded = st.file_uploader(
    "Choose a video file",
    type=["mp4", "mov", "avi", "mkv"],
    label_visibility="collapsed"
)

# ---------- SETTINGS ----------
st.markdown("## 2. Redaction Settings")

# CATEGORY SELECTION
CATEGORY_OPTIONS = [
    'Faces', 
    'People', 
    'Cars', 
    'Trucks', 
    'Cell phones', 
    'Laptops',
    'Screens',
    'Vehicles'
]

st.markdown("### Objects to Redact")
selected_categories = st.multiselect(
    "Select the types of objects to redact",
    options=CATEGORY_OPTIONS,
    default=['Faces', 'Cars'],
    help="Select all items you want the model to detect and blur/block."
)

# Layout for Redaction Configuration
col1, col2, col3 = st.columns(3)

with col1:
    method = st.selectbox(
        "Style",
        ["Black box", "Blur", "Pixelate"],
        help="Choose how detected regions are hidden"
    )

with col2:
    conf = st.slider(
        "Confidence",
        0.1, 0.9, 0.1, # Set default to 0.1 for maximum detection visibility
        help="Lower values (e.g., 0.1) help detect low-confidence objects like far-away cars."
    )

with col3:
    padding = st.slider(
        "Padding",
        0.0, 0.5, 0.15,
        help="Extra space added around detected areas"
    )

st.markdown("### Speed Optimization")
frame_skip = st.slider(
    "Frame Skip",
    1, 10, 5,
    help="Process only 1 in N frames (1 = full speed/slowest, 5 = 5x faster/less accurate)"
)


# ---------- PROCESSING ----------
if uploaded is not None:
    # 1. Save the uploaded video to a temporary file
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp_in.write(uploaded.read())
        tmp_in.flush()
        tmp_in.close()

        # 2. Generate a unique path for the output redacted video
        tmp_out_path = os.path.join(
            tempfile.gettempdir(),
            f"out_redacted_{int(time.time())}.mp4"
        )
        
        # 3. CONSTRUCT THE SETTINGS DICTIONARY
        processing_settings = {
            'categories': selected_categories,
            'style': method,
            'frame_skip': frame_skip,
            'confidence': conf,
            'padding': padding,
        }

        st.info("Processing may take some time on CPU. Short videos work best.")

        with st.spinner("Running detection and redaction..."):
            
            # 4. CALL THE CORE PROCESSING FUNCTION
            process_video(
                input_path=tmp_in.name,
                output_path=tmp_out_path,
                settings=processing_settings
            )

            # ---------- OUTPUT (Preview and Download) ----------
            st.success(" Redaction complete!")

            # Preview
            st.markdown("### Redacted Video Preview")
            st.video(tmp_out_path)
            
            # Download
            st.markdown("### Download")
            with open(tmp_out_path, "rb") as f:
                st.download_button(
                    label="⬇ Download Redacted Video",
                    data=f,
                    file_name="redacted_video.mp4",
                    mime="video/mp4",
                    help="Click to download the processed video file."
                )

        # 5. Cleanup the temporary input file
        os.unlink(tmp_in.name)

    except Exception as e:
        st.error(f" Processing failed: {e}")
        if os.path.exists(tmp_in.name):
            os.unlink(tmp_in.name)
        if 'tmp_out_path' in locals() and os.path.exists(tmp_out_path):
             os.unlink(tmp_out_path)