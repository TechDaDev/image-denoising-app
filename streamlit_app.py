import os
import io
import zipfile
import numpy as np
import pydicom
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
from scipy.signal import wiener
from scipy.ndimage import median_filter
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

# Sidebar: Traditional filter configuration
st.sidebar.title("Traditional Filters")

# Available filter options
FILTER_OPTIONS = [
    "Non-Local Means",
    "Wavelet Denoising",
    "Median (3x3)",
    "Wiener (3x3)",
]

pre_filters = st.sidebar.multiselect(
    "Apply before DnCNN",
    options=FILTER_OPTIONS,
    default=["Non-Local Means", "Wavelet Denoising"],
    help="Pre-filters reduce heavy noise so DnCNN focuses on residuals."
)

post_filters = st.sidebar.multiselect(
    "Apply after DnCNN",
    options=FILTER_OPTIONS,
    default=[],
    help="Post-filters are usually light to avoid oversmoothing."
)

with st.sidebar.expander("Filter settings"):
    # NLM settings
    nlm_h_factor = st.slider("NLM h × sigma", 0.3, 1.5, 0.8, 0.1,
                             help="Strength of NLM denoising relative to estimated sigma.")
    nlm_patch_size = st.select_slider("NLM patch size", options=[3,5,7], value=5)
    nlm_patch_distance = st.select_slider("NLM patch distance", options=[3,5,7,9], value=3)

    # Wavelet settings
    wavelet_type = st.selectbox("Wavelet type", options=["db1", "db2", "sym4", "haar"], index=0)
    wavelet_method = st.selectbox("Wavelet method", options=["BayesShrink", "VisuShrink"], index=0)
    wavelet_mode = st.selectbox("Threshold mode", options=["soft", "hard"], index=0)

    # Median & Wiener
    median_size = st.select_slider("Median size", options=[3,5,7], value=3)
    wiener_size = st.select_slider("Wiener size", options=[3,5,7], value=3)

# Sharpening options (sidebar for both single & batch)
st.sidebar.title("Sharpening")
apply_sharpening = st.sidebar.checkbox("Apply edge sharpening", value=True)
amount = st.sidebar.slider("Unsharp amount", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
sigma_gauss = st.sidebar.slider("Gaussian blur for sharpening", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

# Batch mode toggle
batch_mode = st.sidebar.checkbox("Batch mode (multiple images)", value=False, help="Process multiple images and download a ZIP of outputs.")

# Main title
st.title("🔧 Hybrid Denoiser: Traditional → DnCNN → Optional Sharpening")
st.caption("Traditional filters reduce bulk noise; DnCNN handles residuals; crisp edges re‑introduced afterward. Batch mode supported.")

# Check if model is loaded
if not st.session_state.model_loaded:
    st.warning("Please load a model from the sidebar to continue.")
    st.stop()

model = st.session_state.current_model

if batch_mode:
    uploaded_files = st.file_uploader(
        "Upload ultrasound images (PNG/JPG/DICOM)",
        type=["png", "jpg", "jpeg", "dcm"],
        accept_multiple_files=True
    )
    if not uploaded_files:
        st.info("Waiting for image uploads…")
        st.stop()
    if len(uploaded_files) > 25:
        st.warning("Maximum 25 images allowed; extra files will be ignored.")
        uploaded_files = uploaded_files[:25]
else:
    # Single file upload
    uploaded = st.file_uploader(
        "Upload ultrasound (grayscale) image (PNG/JPG/DICOM)",
        type=["png", "jpg", "jpeg", "dcm"],
        accept_multiple_files=False
    )
    if uploaded is None:
        st.info("Waiting for image upload…")
        st.stop()

def load_dicom_as_float(file) -> np.ndarray:
    """Load a DICOM file-like object and return a float32 image in [0,1].
    Applies basic rescale using RescaleSlope/Intercept if present and min-max normalization.
    """
    ds = pydicom.dcmread(file)
    pixel = ds.pixel_array.astype(np.float32)
    # Rescale if metadata present
    slope = getattr(ds, 'RescaleSlope', 1.0)
    intercept = getattr(ds, 'RescaleIntercept', 0.0)
    pixel = pixel * slope + intercept
    # Windowing (if WindowCenter/Width present)
    wc = getattr(ds, 'WindowCenter', None)
    ww = getattr(ds, 'WindowWidth', None)
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = float(wc[0])
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = float(ww[0])
    if wc is not None and ww is not None and ww > 1e-3:
        low = wc - ww / 2.0
        high = wc + ww / 2.0
        pixel = np.clip(pixel, low, high)
    # Normalize to [0,1]
    pmin, pmax = np.min(pixel), np.max(pixel)
    if pmax > pmin:
        pixel = (pixel - pmin) / (pmax - pmin)
    else:
        pixel = np.zeros_like(pixel, dtype=np.float32)
    return pixel.astype(np.float32)

def process_file(file_obj):
    name = file_obj.name
    # Load image
    if name.lower().endswith('.dcm'):
        try:
            orig_local = load_dicom_as_float(file_obj)
            if orig_local.ndim == 3:
                orig_local = orig_local[..., 0]
        except Exception as e:
            st.warning(f"Skipping {name}: DICOM read error ({e})")
            return None
    else:
        try:
            orig_local = img_as_float(np.array(Image.open(file_obj).convert('L')))
        except Exception as e:
            st.warning(f"Skipping {name}: read error ({e})")
            return None
    # Estimate noise
    sigma_est_local = estimate_sigma(orig_local, channel_axis=None)
    # Pre filters
    pre_local = apply_traditional_filters(orig_local, pre_filters, sigma_est_local) if pre_filters else orig_local
    x_local = np.clip(pre_local,0,1)[np.newaxis,...,np.newaxis]
    dn_out_local = model.predict(x_local, verbose=0)[0,...,0]
    den_local = np.clip(dn_out_local,0,1)
    post_local = apply_traditional_filters(den_local, post_filters, sigma_est_local) if post_filters else den_local
    if apply_sharpening:
        blurred_local = gaussian(post_local, sigma=sigma_gauss)
        sharp_local = post_local + amount * (post_local - blurred_local)
        final_local = np.clip(sharp_local,0,1)
    else:
        final_local = post_local
    # Metrics
    psnr_local = psnr(orig_local, final_local, data_range=1.0)
    ssim_local = ssim(orig_local, final_local, data_range=1.0, win_size=7)
    ent_o = calculate_entropy(orig_local)
    ent_f = calculate_entropy(final_local)
    # Side by side
    try:
        sbs = np.concatenate([
            (orig_local*255).astype(np.uint8),
            (den_local*255).astype(np.uint8),
            (final_local*255).astype(np.uint8)
        ], axis=1)
    except Exception:
        sbs = None
    return {
        'name': name,
        'orig': orig_local,
        'den': den_local,
        'final': final_local,
        'psnr': psnr_local,
        'ssim': ssim_local,
        'ent_orig': ent_o,
        'ent_final': ent_f,
        'sbs': sbs
    }

if batch_mode:
    results = []
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for f in uploaded_files:
            res = process_file(f)
            if res is None:
                continue
            results.append(res)
            # Write final image
            final_img = Image.fromarray((res['final']*255).astype(np.uint8))
            buf_img = io.BytesIO(); final_img.save(buf_img, format='PNG'); buf_img.seek(0)
            zf.writestr(f"{Path(res['name']).stem}_final.png", buf_img.read())
            # Write side-by-side
            if res['sbs'] is not None:
                buf_sbs = io.BytesIO(); Image.fromarray(res['sbs']).save(buf_sbs, format='PNG'); buf_sbs.seek(0)
                zf.writestr(f"{Path(res['name']).stem}_comparison.png", buf_sbs.read())
    if not results:
        st.error("No images processed.")
        st.stop()
    # Metrics summary
    st.subheader("Batch Results Summary")
    for r in results:
        st.write(f"{r['name']}: PSNR {r['psnr']:.2f} dB | SSIM {r['ssim']:.4f} | Entropy Δ {r['ent_final']-r['ent_orig']:+.4f}")
    st.subheader("Sample Previews")
    for r in results[:min(4,len(results))]:
        if r['sbs'] is not None:
            st.image(r['sbs'], caption=r['name'], use_container_width=True)
    # Download zip
    zip_buffer.seek(0)
    st.download_button("⬇️ Download All (ZIP)", zip_buffer, file_name="denoised_batch_results.zip", mime="application/zip")
    st.stop()

# Read image (PNG/JPG/DICOM) single mode
if uploaded.name.lower().endswith('.dcm'):
    try:
        orig = load_dicom_as_float(uploaded)
        if orig.ndim == 3:
            orig = orig[..., 0]
        img = Image.fromarray((orig * 255).astype(np.uint8), mode='L')
    except Exception as e:
        st.error(f"Failed to read DICOM: {e}")
        st.stop()
else:
    img = Image.open(uploaded).convert("L")
    orig = img_as_float(np.array(img))

# Helper: apply traditional filters in selected order
def apply_traditional_filters(image: np.ndarray, selections: list, sigma_val: float) -> np.ndarray:
    out = image.astype(np.float32)
    for f in selections:
        if f == "Non-Local Means":
            out = denoise_nl_means(
                out,
                patch_size=nlm_patch_size,
                patch_distance=nlm_patch_distance,
                h=nlm_h_factor * sigma_val,
                fast_mode=True,
                channel_axis=None,
            )
        elif f == "Wavelet Denoising":
            out = denoise_wavelet(
                out,
                sigma=sigma_val,
                wavelet=wavelet_type,
                method=wavelet_method,
                mode=wavelet_mode,
                rescale_sigma=True,
                channel_axis=None,
            )
        elif f == "Median (3x3)":
            out = median_filter(out, size=median_size)
        elif f == "Wiener (3x3)":
            out = wiener(out, mysize=wiener_size)
        # keep in [0,1]
        out = np.clip(out, 0, 1).astype(np.float32)
    return out

# Estimate noise level (used by NLM/Wavelet)
sigma_est = estimate_sigma(orig, channel_axis=None)

# Apply pre-filters
prefiltered = apply_traditional_filters(orig, pre_filters, sigma_est) if pre_filters else orig

# Normalize to [0,1] for model input
input_img = np.clip(prefiltered, 0, 1).astype(np.float32)
# Prepare for DnCNN: shape (1, H, W, 1)
x = input_img[np.newaxis, ..., np.newaxis]

# DnCNN Predict (After pre-filters)
dncnn_out = model.predict(x)[0, ..., 0]
denoised = np.clip(dncnn_out, 0, 1)

# Apply post-filters if any
post_out = apply_traditional_filters(denoised, post_filters, sigma_est) if post_filters else denoised

# Apply sharpening if enabled (after post-filters)
if apply_sharpening:
    blurred = gaussian(post_out, sigma=sigma_gauss)
    sharpened = post_out + amount * (post_out - blurred)
    final = np.clip(sharpened, 0, 1)
else:
    final = post_out

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

# Side-by-side composite (Original | After DnCNN | Final)
orig_u8 = (orig * 255).astype(np.uint8)
denoised_u8 = (denoised * 255).astype(np.uint8)
final_u8 = (final * 255).astype(np.uint8)

# Ensure same height and concatenate horizontally
try:
    side_by_side = np.concatenate([orig_u8, denoised_u8, final_u8], axis=1)
    sbs_buf = io.BytesIO()
    Image.fromarray(side_by_side).save(sbs_buf, format="PNG")
    sbs_buf.seek(0)
    st.download_button(
        "⬇️ Download Side-by-Side (PNG)",
        sbs_buf,
        file_name="comparison_side_by_side.png",
        mime="image/png"
    )
except Exception as e:
    st.warning(f"Could not create side-by-side image: {e}")
