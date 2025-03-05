# Human-Segmentation-Background-Removal
This project uses YOLO-Seg to detect humans in an image, isolate the most confidently detected individual, and offer background removal options: ✅ White Background ✅ Transparent Sticker ✅ Detection View ✅ Save Option. The interface enables real-time selection and runs efficiently on GPU (CUDA) or CPU.

## Requirements
- Python 3.10 or greater
- CUDA 12.1 (or compatible version for GPU acceleration)

## Installation & Setup

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/human-subject-extractor.git
cd human-subject-extractor
```

### 2. Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

**Note:** This setup assumes Torch and TorchVision are configured for CUDA 12.1. If your CUDA version differs, install the appropriate PyTorch version from the official PyTorch site:

[PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

### 4. Check Your CUDA Version
To verify your CUDA version, run:
```sh
nvcc --version
```
If CUDA is not available, the program will automatically run on the CPU.

## Running the Program
To start the program, run:
```sh
python subject_extractor.py
```

## Features & Options
- **Detect Human Subject** – Finds the most confidently detected human in an image.
- **Background Removal Options:**
  - ✅ White Background – Replaces the background with plain white.
  - ✅ Transparent Sticker – Creates a sticker-like output with a checkered transparency effect.
  - ✅ Detection View – Displays the detected human with bounding boxes.
  - ✅ Save Option – Saves the final processed image.
- **Runs on GPU (CUDA) or CPU automatically** based on system configuration.

Enjoy using the Human Subject Extractor! 🚀

