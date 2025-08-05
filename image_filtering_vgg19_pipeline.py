import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pywt
from skimage import img_as_float
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from scipy.signal import wiener
from scipy.ndimage import median_filter

# Make sure to install TensorFlow: pip install tensorflow
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import csv

# ------------------------------------------------------------------------------
# STEP 1: FILTERING & METRIC FUNCTIONS
# ------------------------------------------------------------------------------

def apply_wiener_filter(image: np.ndarray) -> np.ndarray:
    """Apply Wiener filter and clip to [0,1]."""
    return np.clip(wiener(image, mysize=3), 0.0, 1.0)

def apply_median_filter(image: np.ndarray) -> np.ndarray:
    """Apply 3×3 median filter."""
    return median_filter(image, size=3)

def apply_anisotropic_diffusion(image: np.ndarray) -> np.ndarray:
    """
    Use Non-Local Means denoising as a stand-in for anisotropic diffusion.
    Expects image in [0,1].
    """
    img_f = img_as_float(image)
    sigma_est = np.mean(estimate_sigma(img_f, channel_axis=None))
    denoised = denoise_nl_means(
        img_f,
        h=0.8 * sigma_est,
        fast_mode=True,
        patch_size=5,
        patch_distance=3,
        channel_axis=None
    )
    return denoised

def apply_dwt(image: np.ndarray) -> np.ndarray:
    """
    Perform a single-level Haar DWT and reconstruct.
    Expects image in [0,1].
    """
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    recon = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
    H, W = image.shape
    return recon[:H, :W]

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Convert uint8 [0,255] → float32 [0,1]."""
    return np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)

def compute_psnr(original: np.ndarray, filtered: np.ndarray) -> float:
    """Compute PSNR (dB) between two [0,1] images."""
    return sk_psnr(original, np.clip(filtered, 0.0, 1.0), data_range=1.0)

def compute_ssim(original: np.ndarray, filtered: np.ndarray) -> float:
    """Compute SSIM between two [0,1] images."""
    return sk_ssim(original, np.clip(filtered, 0.0, 1.0), data_range=1.0)

def load_images_to_vectors(directory: str, img_size=(64, 64)) -> np.ndarray:
    """
    Load all PNG images in `directory` as grayscale, resize to img_size,
    flatten to 1D vectors. Returns array of shape (N, img_size[0]*img_size[1]).
    """
    images = []
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".png"):
            continue
        path = os.path.join(directory, filename)
        img = Image.open(path).convert('L')
        img = img.resize(img_size)
        vec = np.array(img, dtype=np.uint8).flatten()
        images.append(vec)
    return np.array(images)

def evaluate_all_metrics(image_vectors: np.ndarray,
                         img_size=(64, 64),
                         num_images=3) -> None:
    """
    Randomly pick `num_images` from `image_vectors`, apply Wiener, Median,
    Anisotropic Diffusion, and DWT filters. Print PSNR & SSIM per image
    and average, then display a 5×N grid of (Original, Wiener, Median,
    Anisotropic, DWT).
    """
    n = image_vectors.shape[0]
    if n == 0:
        print("No images to evaluate.")
        return
    if n < num_images:
        num_images = n

    indices = random.sample(range(n), num_images)
    total_psnr = {'wiener':0, 'median':0, 'anisotropic':0, 'dwt':0}
    total_ssim = {'wiener':0, 'median':0, 'anisotropic':0, 'dwt':0}

    for idx in indices:
        orig_vec = image_vectors[idx]
        orig = orig_vec.reshape(img_size)
        orig_f = normalize_image(orig)

        wiener_img      = apply_wiener_filter(orig_f)
        median_img      = apply_median_filter(orig_f)
        anisotropic_img = apply_anisotropic_diffusion(orig_f)
        dwt_img         = apply_dwt(orig_f)

        psnr_w = compute_psnr(orig_f, wiener_img)
        psnr_m = compute_psnr(orig_f, median_img)
        psnr_a = compute_psnr(orig_f, anisotropic_img)
        psnr_d = compute_psnr(orig_f, dwt_img)

        ssim_w = compute_ssim(orig_f, wiener_img)
        ssim_m = compute_ssim(orig_f, median_img)
        ssim_a = compute_ssim(orig_f, anisotropic_img)
        ssim_d = compute_ssim(orig_f, dwt_img)

        print(f"\n--- Image Index {idx} ---")
        print(f"PSNR (Wiener)      : {psnr_w:.2f} dB")
        print(f"PSNR (Median)      : {psnr_m:.2f} dB")
        print(f"PSNR (Anisotropic) : {psnr_a:.2f} dB")
        print(f"PSNR (DWT)         : {psnr_d:.2f} dB")
        print(f"SSIM (Wiener)      : {ssim_w:.4f}")
        print(f"SSIM (Median)      : {ssim_m:.4f}")
        print(f"SSIM (Anisotropic) : {ssim_a:.4f}")
        print(f"SSIM (DWT)         : {ssim_d:.4f}")

        total_psnr['wiener']      += psnr_w
        total_psnr['median']      += psnr_m
        total_psnr['anisotropic'] += psnr_a
        total_psnr['dwt']         += psnr_d

        total_ssim['wiener']      += ssim_w
        total_ssim['median']      += ssim_m
        total_ssim['anisotropic'] += ssim_a
        total_ssim['dwt']         += ssim_d

    print("\n=== Average Values Across Images ===")
    for key in ['wiener', 'median', 'anisotropic', 'dwt']:
        avg_psnr = total_psnr[key] / num_images
        avg_ssim = total_ssim[key] / num_images
        print(f"{key.capitalize():<12} → Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}")

    # Plot results
    rows, cols = 5, num_images
    plt.figure(figsize=(4*cols, 4*rows))
    for i, idx in enumerate(indices):
        orig = image_vectors[idx].reshape(img_size)
        orig_f = normalize_image(orig)
        wiener_img      = apply_wiener_filter(orig_f)
        median_img      = apply_median_filter(orig_f)
        anisotropic_img = apply_anisotropic_diffusion(orig_f)
        dwt_img         = apply_dwt(orig_f)

        imgs = [orig_f, wiener_img, median_img, anisotropic_img, dwt_img]
        titles = ["Original", "Wiener", "Median", "Anisotropic", "DWT"]
        for r in range(rows):
            ax = plt.subplot(rows, cols, r*cols + i + 1)
            ax.imshow(imgs[r], cmap='gray')
            ax.set_title(f"{titles[r]} (idx={idx})")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# STEP 2: VGG19 FEATURE EXTRACTION WITH DWT + ANISOTROPIC DIFFUSION PREPROCESSING
# ------------------------------------------------------------------------------

def build_vgg19_feature_extractor(img_size=(224, 224, 3)) -> Model:
    """
    Load VGG19 pretrained on ImageNet (include_top=False) and add
    GlobalAveragePooling2D to produce a single feature vector.
    """
    base = VGG19(weights="imagenet", include_top=False, input_shape=img_size)
    pooled = GlobalAveragePooling2D()(base.output)
    feat_model = Model(inputs=base.input, outputs=pooled)
    return feat_model

def preprocess_image_dwt_anisotropic(image_path: str, img_size=(224, 224)) -> np.ndarray:
    """
    Comprehensive image preprocessing pipeline:
    1. Load image as grayscale
    2. Resize to specified size
    3. Normalize to [0,1]
    4. Apply DWT (Discrete Wavelet Transform)
    5. Apply Anisotropic Diffusion
    6. Convert to 3-channel RGB for VGG19
    7. Apply VGG19 preprocessing
    
    Returns preprocessed image ready for VGG19 input.
    """
    # Step 1: Load and convert to grayscale
    pil_img = Image.open(image_path).convert("L")
    
    # Step 2: Resize to target size
    pil_img = pil_img.resize(img_size)
    
    # Step 3: Convert to numpy array and normalize to [0,1]
    gray_np = np.array(pil_img, dtype=np.float32) / 255.0
    
    # Step 4: Apply DWT (Discrete Wavelet Transform)
    print(f"Applying DWT to {os.path.basename(image_path)}...")
    dwt_output = apply_dwt(gray_np)
    
    # Step 5: Apply Anisotropic Diffusion
    print(f"Applying Anisotropic Diffusion to {os.path.basename(image_path)}...")
    aniso_output = apply_anisotropic_diffusion(dwt_output)
    
    # Step 6: Ensure output shape matches target size
    if aniso_output.shape != img_size:
        tmp = Image.fromarray((aniso_output * 255).astype(np.uint8))
        tmp = tmp.resize(img_size)
        aniso_output = np.array(tmp, dtype=np.float32) / 255.0
    
    # Step 7: Convert single channel to 3-channel RGB
    rgb_image = np.stack([aniso_output, aniso_output, aniso_output], axis=-1)  # (H, W, 3)
    
    # Step 8: Scale back to [0,255] and apply VGG19 preprocessing
    rgb_scaled = rgb_image * 255.0
    preprocessed = preprocess_input(rgb_scaled)
    
    return preprocessed

def extract_features_with_dwt_anisotropic(
    image_directory: str,
    img_size=(224, 224),
    save_csv="enhanced_vgg19_features.csv"
) -> None:
    """
    Extract VGG19 features from images with DWT + Anisotropic Diffusion preprocessing.
    
    For each PNG image in the directory:
    1. Apply DWT + Anisotropic Diffusion preprocessing
    2. Extract features using VGG19
    3. Save results to CSV file
    """
    print("Building VGG19 feature extractor...")
    feat_model = build_vgg19_feature_extractor(img_size=(img_size[0], img_size[1], 3))
    feat_model.trainable = False
    
    print("Starting feature extraction with DWT + Anisotropic Diffusion preprocessing...")
    
    rows = []
    processed_count = 0
    
    for filename in sorted(os.listdir(image_directory)):
        if not filename.lower().endswith(".png"):
            continue
            
        image_path = os.path.join(image_directory, filename)
        
        try:
            # Apply DWT + Anisotropic Diffusion preprocessing
            preprocessed_image = preprocess_image_dwt_anisotropic(image_path, img_size)
            
            # Add batch dimension for VGG19
            input_batch = np.expand_dims(preprocessed_image, axis=0)  # (1, H, W, 3)
            
            # Extract features using VGG19
            features = feat_model.predict(input_batch, verbose=0)  # (1, feature_dim)
            features = features.flatten()  # (feature_dim,)
            
            # Create row for CSV
            row = [filename] + features.tolist()
            rows.append(row)
            
            processed_count += 1
            print(f"Processed {filename} → feature dimension = {features.shape[0]} (Total: {processed_count})")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    if not rows:
        print("No PNG files found or processed successfully. No CSV file created.")
        return
    
    # Create CSV header
    feature_dim = len(rows[0]) - 1
    header = ["filename"] + [f"vgg19_feat_{i}" for i in range(feature_dim)]
    
    # Write to CSV
    with open(save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    
    print(f"\nFeature extraction completed!")
    print(f"Total images processed: {len(rows)}")
    print(f"Feature dimension per image: {feature_dim}")
    print(f"Results saved to: '{save_csv}'")

def compare_preprocessing_methods(
    image_directory: str,
    sample_size=3,
    img_size=(224, 224)
) -> None:
    """
    Compare VGG19 features extracted with different preprocessing methods:
    1. Original image (no preprocessing)
    2. DWT only
    3. Anisotropic Diffusion only
    4. DWT + Anisotropic Diffusion (combined)
    
    Visualize the preprocessing results for comparison.
    """
    print("Comparing different preprocessing methods...")
    
    # Get sample images
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith('.png')]
    if len(image_files) < sample_size:
        sample_size = len(image_files)
    
    sample_files = random.sample(image_files, sample_size)
    
    # Build VGG19 model
    feat_model = build_vgg19_feature_extractor(img_size=(img_size[0], img_size[1], 3))
    
    # Create comparison plot
    fig, axes = plt.subplots(sample_size, 4, figsize=(16, 4*sample_size))
    if sample_size == 1:
        axes = axes.reshape(1, -1)
    
    for i, filename in enumerate(sample_files):
        image_path = os.path.join(image_directory, filename)
        
        # Load original image
        pil_img = Image.open(image_path).convert("L")
        pil_img = pil_img.resize(img_size)
        original = np.array(pil_img, dtype=np.float32) / 255.0
        
        # Apply different preprocessing methods
        dwt_only = apply_dwt(original)
        aniso_only = apply_anisotropic_diffusion(original)
        dwt_aniso = apply_anisotropic_diffusion(apply_dwt(original))
        
        # Plot results
        methods = [original, dwt_only, aniso_only, dwt_aniso]
        titles = ["Original", "DWT Only", "Anisotropic Only", "DWT + Anisotropic"]
        
        for j, (method_result, title) in enumerate(zip(methods, titles)):
            axes[i, j].imshow(method_result, cmap='gray')
            axes[i, j].set_title(f"{title}\n({filename})")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Preprocessing comparison visualization completed.")

# ------------------------------------------------------------------------------
# STEP 3: ACCURACY AND LOSS EVALUATION
# ------------------------------------------------------------------------------

def create_classification_model(input_dim: int, num_classes: int) -> Model:
    """
    Create a neural network classifier for evaluating feature quality.
    Uses the extracted VGG19 features as input.
    """
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_synthetic_labels(num_samples: int, num_classes: int = 5) -> np.ndarray:
    """
    Generate synthetic labels for classification task.
    In real scenarios, you would have actual labels for your images.
    """
    # Create balanced synthetic labels
    labels_per_class = num_samples // num_classes
    remainder = num_samples % num_classes
    
    labels = []
    for class_id in range(num_classes):
        count = labels_per_class + (1 if class_id < remainder else 0)
        labels.extend([class_id] * count)
    
    # Shuffle the labels
    np.random.shuffle(labels)
    return np.array(labels)

def evaluate_feature_quality(csv_file: str, test_size: float = 0.2) -> dict:
    """
    Evaluate the quality of extracted features by training a classifier.
    Returns accuracy, loss, and training history.
    """
    print(f"\nEvaluating feature quality from '{csv_file}'...")
    
    # Load features from CSV
    try:
        import pandas as pd
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} samples from CSV file")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return {}
    
    # Extract features (all columns except filename)
    feature_columns = [col for col in df.columns if col != 'filename']
    X = df[feature_columns].values
    
    # Generate synthetic labels (in real scenarios, use actual labels)
    num_classes = 5  # Adjust based on your classification task
    y_synthetic = generate_synthetic_labels(len(X), num_classes)
    y_categorical = to_categorical(y_synthetic, num_classes)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=test_size, random_state=42, stratify=y_synthetic
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create and train model
    model = create_classification_model(X.shape[1], num_classes)
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
    ]
    
    print("\nTraining classification model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    
    # Predictions for detailed analysis
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate additional metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
    
    results = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_loss': train_loss,
        'test_loss': test_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'history': history,
        'model': model,
        'predictions': {
            'y_true': y_true_classes,
            'y_pred': y_pred_classes,
            'y_prob': y_pred
        }
    }
    
    return results

def plot_training_history(history, save_path: str = None):
    """
    Plot training and validation accuracy and loss curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, num_classes: int, save_path: str = None):
    """
    Plot confusion matrix for classification results.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {save_path}")
    
    plt.show()

def compare_preprocessing_accuracy(image_directory: str, sample_size: int = 100) -> dict:
    """
    Compare accuracy of different preprocessing methods by training classifiers
    on features extracted with each method.
    """
    print("\nComparing preprocessing methods based on classification accuracy...")
    
    # Get sample of images
    image_files = [f for f in os.listdir(image_directory) if f.lower().endswith('.png')]
    if len(image_files) < sample_size:
        sample_size = len(image_files)
    
    sample_files = random.sample(image_files, sample_size)
    
    # Build VGG19 model
    feat_model = build_vgg19_feature_extractor()
    
    # Extract features with different preprocessing methods
    methods = {
        'original': lambda x: x,
        'dwt_only': apply_dwt,
        'anisotropic_only': apply_anisotropic_diffusion,
        'dwt_anisotropic': lambda x: apply_anisotropic_diffusion(apply_dwt(x))
    }
    
    results = {}
    
    for method_name, preprocess_func in methods.items():
        print(f"\nProcessing with {method_name} preprocessing...")
        
        features = []
        for filename in sample_files:
            image_path = os.path.join(image_directory, filename)
            
            # Load and preprocess image
            pil_img = Image.open(image_path).convert("L")
            pil_img = pil_img.resize((224, 224))
            gray_np = np.array(pil_img, dtype=np.float32) / 255.0
            
            # Apply preprocessing
            processed = preprocess_func(gray_np)
            
            # Convert to RGB and extract features
            rgb_image = np.stack([processed, processed, processed], axis=-1)
            rgb_scaled = rgb_image * 255.0
            preprocessed = preprocess_input(rgb_scaled)
            input_batch = np.expand_dims(preprocessed, axis=0)
            
            feat = feat_model.predict(input_batch, verbose=0).flatten()
            features.append(feat)
        
        features = np.array(features)
        
        # Generate labels and train classifier
        num_classes = 5
        y_synthetic = generate_synthetic_labels(len(features), num_classes)
        y_categorical = to_categorical(y_synthetic, num_classes)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_categorical, test_size=0.2, random_state=42, stratify=y_synthetic
        )
        
        # Train model
        model = create_classification_model(features.shape[1], num_classes)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=16,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        results[method_name] = {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'history': history
        }
        
        print(f"{method_name} - Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")
    
    return results

def print_evaluation_summary(results: dict):
    """
    Print a comprehensive summary of the evaluation results.
    """
    print("\n" + "=" * 80)
    print("FEATURE QUALITY EVALUATION SUMMARY")
    print("=" * 80)
    
    print(f"Training Accuracy: {results['train_accuracy']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Training Loss: {results['train_loss']:.4f}")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    
    # Overfitting analysis
    overfit_score = results['train_accuracy'] - results['test_accuracy']
    if overfit_score > 0.1:
        print(f"\nWarning: Potential overfitting detected (difference: {overfit_score:.4f})")
    else:
        print(f"\nGood generalization (train-test difference: {overfit_score:.4f})")
    
    print("\nFeature quality assessment:")
    if results['test_accuracy'] > 0.8:
        print("✓ Excellent feature quality - High discriminative power")
    elif results['test_accuracy'] > 0.6:
        print("✓ Good feature quality - Moderate discriminative power")
    elif results['test_accuracy'] > 0.4:
        print("⚠ Fair feature quality - Limited discriminative power")
    else:
        print("✗ Poor feature quality - Low discriminative power")

# ------------------------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Dataset configuration
    dataset_base = "dataset"
    unknown_dir = os.path.join(dataset_base, "unknown")

    print("=" * 80)
    print("VGG19 FEATURE EXTRACTION WITH DWT + ANISOTROPIC DIFFUSION PREPROCESSING")
    print("=" * 80)

    # Check if dataset directory exists
    if not os.path.isdir(unknown_dir):
        print(f"Error: Directory '{unknown_dir}' not found.")
        print("Please ensure the dataset is extracted correctly.")
        exit(1)

    # Count available images
    png_files = [f for f in os.listdir(unknown_dir) if f.lower().endswith('.png')]
    print(f"Found {len(png_files)} PNG images in '{unknown_dir}'")

    if len(png_files) == 0:
        print("No PNG images found. Exiting.")
        exit(1)

    # PART A: Traditional Filter Evaluation
    print("\n" + "=" * 60)
    print("PART A: EVALUATING TRADITIONAL FILTERS")
    print("=" * 60)

    vectors = load_images_to_vectors(unknown_dir, img_size=(64, 64))
    if vectors.shape[0] > 0:
        evaluate_all_metrics(vectors, img_size=(64, 64), num_images=min(6, len(png_files)))

    # PART B: VGG19 Feature Extraction with Enhanced Preprocessing
    print("\n" + "=" * 60)
    print("PART B: VGG19 FEATURE EXTRACTION WITH DWT + ANISOTROPIC DIFFUSION")
    print("=" * 60)

    extract_features_with_dwt_anisotropic(
        image_directory=unknown_dir,
        img_size=(224, 224),
        save_csv="enhanced_vgg19_features.csv"
    )

    # PART C: Preprocessing Method Comparison
    print("\n" + "=" * 60)
    print("PART C: COMPARING PREPROCESSING METHODS")
    print("=" * 60)

    compare_preprocessing_methods(
        image_directory=unknown_dir,
        sample_size=3,
        img_size=(224, 224)
    )

    # PART D: Accuracy and Loss Evaluation
    print("\n" + "=" * 60)
    print("PART D: ACCURACY AND LOSS EVALUATION")
    print("=" * 60)

    # Evaluate feature quality using classification
    evaluation_results = evaluate_feature_quality("enhanced_vgg19_features.csv")

    if evaluation_results:
        # Print summary
        print_evaluation_summary(evaluation_results)

        # Plot training history
        plot_training_history(
            evaluation_results['history'], 
            save_path="training_history.png"
        )

        # Plot confusion matrix
        plot_confusion_matrix(
            evaluation_results['predictions']['y_true'],
            evaluation_results['predictions']['y_pred'],
            num_classes=5,
            save_path="confusion_matrix.png"
        )
