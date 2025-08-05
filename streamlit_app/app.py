import os
import io
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st
import pywt
from skimage import img_as_float
from skimage.restoration import (
    denoise_nl_means,
    denoise_wavelet,
    estimate_sigma
)
import tensorflow as tf
from skimage.filters import gaussian
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy import stats

def calculate_entropy(image):
    """Calculate the entropy of an image.
    
    Args:
        image: 2D numpy array (grayscale image)
        
    Returns:
        float: Entropy value
    """
    # Calculate histogram
    hist = np.histogram(image, bins=256, range=(0, 1))[0]
    # Normalize histogram to get probabilities
    hist = hist / np.sum(hist)
    # Calculate entropy (avoid log(0) by removing zero values)
    entropy = -np.sum([p * np.log2(p) for p in hist if p > 0])
    return entropy

# Configure page
st.set_page_config(
    page_title="Image Denoiser",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model loading function
@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False  # We'll compile manually
        )
        model.compile(optimizer='adam', loss='mse')
        return model
    except Exception as e:
        st.error(f"Error loading model {model_path}: {str(e)}")
        return None

# Get available models
def get_available_models():
    models_dir = Path("models")
    models = list(models_dir.glob("*.h5"))  # Get all .h5 files in models directory
    return {model.stem: model for model in models}

# Initialize session state for model loading
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
    st.session_state.model_loaded = False

# Limit TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Sidebar for model selection
st.sidebar.title("Model Selection")
available_models = get_available_models()

if not available_models:
    st.sidebar.error("No models found in the 'models' directory!")
    st.stop()

# Create a selectbox for model selection
selected_model_name = st.sidebar.selectbox(
    "Choose a model:",
    options=list(available_models.keys()),
    index=0,  # Default to first model
    help="Select a pre-trained model for denoising"
)

# Load the selected model
if st.sidebar.button("Load Model") or st.session_state.current_model is None:
    with st.spinner(f"Loading {selected_model_name}..."):
        model_path = available_models[selected_model_name]
        st.session_state.current_model = load_model(str(model_path))
        if st.session_state.current_model is not None:
            st.session_state.model_loaded = True
            st.sidebar.success(f"Successfully loaded {selected_model_name}")
        else:
            st.session_state.model_loaded = False

# Main title
st.title("🔧 Hybrid Denoiser: Traditional → DnCNN → Optional Sharpening")
st.caption("Traditional filters reduce bulk noise; DnCNN handles residuals; crisp edges re‑introduced afterward.")

# Check if model is loaded
if not st.session_state.model_loaded:
    st.warning("Please load a model from the sidebar to continue.")
    st.stop()

model = st.session_state.current_model

# 1. Upload
uploaded = st.file_uploader("Upload ultrasound (grayscale) image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
if uploaded is None:
    st.info("Waiting for image upload…")
    st.stop()

# Read image
img = Image.open(uploaded).convert("L")
orig = img_as_float(np.array(img))

# Pre‑filter 1: Non‑Local Means
sigma_est = estimate_sigma(orig, channel_axis=None)
nlm = denoise_nl_means(orig,
                       patch_size=5,
                       patch_distance=3,
                       h=0.8 * sigma_est,
                       fast_mode=True,
                       channel_axis=None)

# Pre‑filter 2: Wavelet denoising
wavelet = denoise_wavelet(nlm,
                          sigma=sigma_est,
                          wavelet='db1',
                          method='BayesShrink',
                          mode='soft',
                          rescale_sigma=True,
                          channel_axis=None)

# Normalize to [0,1]
input_img = np.clip(wavelet, 0, 1).astype(np.float32)
# Prepare for DnCNN: shape (1, H, W, 1)
x = input_img[np.newaxis, ..., np.newaxis]

# DnCNN Predict
denoised = model.predict(x)[0, ..., 0]
denoised = np.clip(denoised, 0, 1)

# Optional: light Gaussian sharpening
# Sliders
apply_sharpening = st.checkbox("Apply edge sharpening", value=True)
amount = st.slider("Unsharp amount", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
sigma_gauss = st.slider("Gaussian blur for sharpening", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Apply sharpening if checkbox is enabled
if apply_sharpening:
    blurred = gaussian(denoised, sigma=sigma_gauss)
    sharpened = denoised + amount * (denoised - blurred)
    final = np.clip(sharpened, 0, 1)
else:
    final = denoised

# Calculate metrics
with st.spinner('Calculating metrics...'):
    # Calculate PSNR and SSIM
    psnr_val = psnr(orig, final, data_range=1.0)
    ssim_val = ssim(orig, final, data_range=1.0, win_size=7)
    
    # Calculate entropy
    orig_entropy = calculate_entropy(orig)
    final_entropy = calculate_entropy(final)

# Display images and metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.image((orig * 255).astype(np.uint8), caption="Original", use_container_width=True)
    st.metric("Entropy", f"{orig_entropy:.4f}", 
             help="Measures the amount of information/randomness in the image. Higher values indicate more texture or noise.")

with col2:
    st.image((denoised * 255).astype(np.uint8), caption="After DnCNN", use_container_width=True)
    st.metric("PSNR", f"{psnr_val:.2f} dB",
             help="Peak Signal-to-Noise Ratio. Higher values (typically 30-50 dB) indicate better quality.\n"
                   "• >30 dB: Good quality\n"
                   "• 20-30 dB: Acceptable quality\n"
                   "• <20 dB: Poor quality")

with col3:
    st.image((final * 255).astype(np.uint8), caption="Final Output", use_container_width=True)
    st.metric("SSIM", f"{ssim_val:.4f}",
             help="Structural Similarity Index. Ranges from -1 to 1, where 1 means identical to original.\n"
                   "• 0.9-1.0: Excellent quality\n"
                   "• 0.7-0.9: Good quality\n"
                   "• <0.7: Noticeable differences")

# Show entropy comparison
st.subheader("Entropy Analysis")
st.write("""
- **Entropy** measures the amount of information in the image.
- Lower entropy in the denoised image indicates reduced noise/randomness.
- A significant drop in entropy might indicate loss of important image details.
""")

col1, col2 = st.columns(2)
with col1:
    st.metric("Original Entropy", f"{orig_entropy:.4f}")
with col2:
    st.metric("Denoised Entropy", f"{final_entropy:.4f}", delta=f"{final_entropy-orig_entropy:+.4f}")
st.markdown("### 📊 Image Quality Metrics")
st.write(f"**PSNR**: {psnr_val:.2f} dB")
st.write(f"**SSIM**: {ssim_val:.4f}")

# Download button
buf = io.BytesIO()
Image.fromarray((final * 255).astype(np.uint8)).save(buf, format="PNG")
buf.seek(0)
st.download_button("⬇️ Download Output (PNG)", buf, file_name="denoised.png", mime="image/png")
