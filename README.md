# DnCNN Image Denoising Web App

An interactive web application for image denoising using a Deep Convolutional Neural Network (DnCNN) with a Streamlit interface. This implementation is based on the paper [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](http://www4.comp.polyu.edu.hk/~cslzhang/paper/DnCNN.pdf).

## 🚀 Features

- **Hybrid Denoising Pipeline**: Combines traditional denoising methods with DnCNN
- **Multiple Pre-trained Models**: Choose from different trained DnCNN models
- **Image Quality Metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - Entropy Analysis
- **Real-time Adjustments**: Fine-tune denoising parameters
- **Download Results**: Save processed images

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/image-denoising-app.git
   cd image-denoising-app/streamlit_app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📂 Project Structure

```
streamlit_app/
├── app.py                # Main Streamlit application
├── models/               # Directory containing pre-trained DnCNN models
│   ├── model_1.h5
│   └── model_2.h5
└── requirements.txt      # Python dependencies
```

## 🚀 Running the Application

1. Start the Streamlit server:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Use the sidebar to select a model and click "Load Model"

4. Upload an image and adjust the parameters as needed

## 📊 Metrics Explanation

- **PSNR (Peak Signal-to-Noise Ratio)**:
  - >30 dB: Good quality
  - 20-30 dB: Acceptable quality
  - <20 dB: Poor quality

- **SSIM (Structural Similarity Index)**:
  - 0.9-1.0: Excellent quality
  - 0.7-0.9: Good quality
  - <0.7: Noticeable differences

- **Entropy**:
  - Measures image information/randomness
  - Higher values indicate more texture or noise

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For any questions or suggestions, please open an issue or contact the repository owner.






