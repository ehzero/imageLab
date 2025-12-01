# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an integrated image processing tool that combines background removal using U²-Net and image upscaling using Real-ESRGAN. The tool can perform both operations together or individually based on command-line options.

**Note**: The README and most code comments are in Korean. The project uses pre-trained U²-Net and Real-ESRGAN models.

## Key Commands

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Tool

**Default: Both background removal and upscaling**

```bash
# Basic usage (background removal + 4x upscaling)
python main.py --input image.jpg

# 2x upscaling
python main.py --input image.jpg --scale 2

# Specify output path
python main.py --input image.jpg --output result.png --scale 4
```

**Background removal only**

```bash
python main.py --input image.jpg --bg-only
```

**Upscaling only**

```bash
python main.py --input image.jpg --upscale-only --scale 2
```

### Model Files Required

**Background Removal Models** (one required):

- `saved_models/u2net/u2net.pth` (176.3 MB) - Full model
- `saved_models/u2net/u2netp.pth` (4.7 MB) - Lightweight model
- Download from: https://github.com/xuebinqin/U-2-Net

**Upscaling Model** (required):

- `saved_models/realesrgan/RealESRGAN_x4plus.pth` (64 MB)
- Download: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

## Architecture

### Core Components

**main.py** (Main integrated script)

- `process_bg_only()`: Performs only background removal
- `process_upscale_only()`: Performs only upscaling
- `process_both()`: Performs both operations sequentially
- `save_image_cv2()`: Saves OpenCV format images
- `save_image_pil()`: Saves PIL Image objects
- Command-line argument parsing with `--bg-only` and `--upscale-only` options

**background_remover.py** (Background removal module)

- `load_model()`: Loads pre-trained U²-Net model (u2net or u2netp)
- `normalize_image()`: Preprocesses images to 320x320, applies ImageNet normalization
- `remove_background_image()`: Main inference function - runs model, extracts mask from d1 output, creates RGBA image
- `pil_to_cv2()`: Converts PIL Image to OpenCV format

**upscaler.py** (Upscaling module)

- `load_model()`: Loads Real-ESRGAN x4plus model (supports 2x, 3x, 4x via outscale parameter)
- `upscale_image()`: Processes image with configurable output scale
- Uses RRDBNet architecture with scale=4 (native scale), supports outscale 2, 3, 4

**model/u2net.py** (Model architecture)

- Implements the U²-Net architecture with nested U-structures
- `U2NET`: Full model (high accuracy, slower)
- `U2NETP`: Lightweight version (faster, lower accuracy)

### Data Flow

**Background Removal:**

1. Input image → RGB conversion → Resize to 320x320 → Normalize with ImageNet stats
2. Forward pass through U²-Net → 7 outputs (d1, d2, ..., d7)
3. Extract d1 (primary output) → Convert to numpy → Scale 0-1 to 0-255
4. Resize mask to original image dimensions using bilinear interpolation
5. Apply mask as alpha channel to original RGB image → Save as RGBA PNG

**Upscaling:**

1. Load image via OpenCV (preserves alpha channel for PNG)
2. Real-ESRGAN enhancement with outscale parameter
3. If outscale ≠ netscale (4), applies bilinear interpolation
4. Save result maintaining original format and transparency

**Integrated Processing:**

1. Background removal → PIL Image (RGBA)
2. Convert PIL Image to OpenCV format (BGRA)
3. Upscaling → OpenCV Image
4. Save as PNG

## Project Structure

```
imageLab/
├── model/
│   └── u2net.py          # Complete U²-Net model architecture
├── saved_models/
│   ├── u2net/            # Pre-trained weights (must download)
│   │   ├── u2net.pth     # Full model weights
│   │   └── u2netp.pth    # Lightweight model weights
│   └── realesrgan/       # Real-ESRGAN weights (must download)
│       └── RealESRGAN_x4plus.pth
├── background_remover.py # Background removal module
├── upscaler.py           # Upscaling module
├── main.py               # Integrated main entry point and CLI
├── requirements.txt      # Dependencies
└── README.md             # Documentation (in Korean)
```

## Technical Notes

### Device Handling

- Automatically detects CUDA availability
- Falls back to CPU if GPU not available
- Model weights loaded with appropriate device mapping

### Image Processing

- **Input**: Any PIL/OpenCV-supported format (JPG, PNG, BMP, TIFF, WebP, etc.)
- **Output**: Always PNG with alpha channel (transparency)
- **Background removal**: Fixed 320x320 internal resolution, output matches original dimensions
- **Upscaling**: Processes at original resolution, outputs at specified scale

### Model Selection

- **Background removal**: `u2net` (higher accuracy) or `u2netp` (faster)
- **Upscaling**: Always uses `x4plus` model, supports 2x, 3x, 4x via outscale parameter

### Output File Naming

- Background removal only: `{input}_nobg.png`
- Upscaling only: `{input}_upscaled_{scale}x.png`
- Both operations: `{input}_processed_{scale}x.png`

## Programmatic Usage

```python
from background_remover import load_model as load_bg_model, remove_background_image
from upscaler import load_model as load_upscale_model, upscale_image
from main import pil_to_cv2

# Load models once (expensive operation)
bg_model = load_bg_model('u2net')  # or 'u2netp'
upscale_model = load_upscale_model()

# Process multiple images
for image_path in image_list:
    # Option 1: Both operations
    no_bg_image = remove_background_image(image_path, bg_model)
    cv2_image = pil_to_cv2(no_bg_image)
    upscaled_image = upscale_image(cv2_image, upscale_model, scale=4)

    # Option 2: Background removal only
    no_bg_image = remove_background_image(image_path, bg_model)

    # Option 3: Upscaling only
    import cv2
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    upscaled_image = upscale_image(image, upscale_model, scale=4)
```

## Common Issues

**FileNotFoundError**: Model weights not downloaded - see README for download links
**CUDA out of memory**: Use `u2netp` (lightweight model) or ensure GPU has sufficient memory
**Poor quality results**: Try `u2net` (full model) instead of `u2netp`, or check if input image has clear foreground/background separation
**Slow processing**: Ensure GPU is being used (check for CUDA availability)
**Scale mismatch error**: Real-ESRGAN model structure is fixed at scale=4, use outscale parameter for different scales (2, 3, 4)

## Integration Patterns

### Sequential Processing (via main.py)

```bash
# Both operations (default)
python main.py --input image.jpg --scale 4

# Background removal only
python main.py --input image.jpg --bg-only

# Upscaling only
python main.py --input image.jpg --upscale-only --scale 2
```

### Batch Processing

```bash
# Process entire directory
for img in images/*.jpg; do
    python main.py --input "$img" --scale 4
done
```

### Python Integration

```python
from main import process_bg_only, process_upscale_only, process_both
from background_remover import load_model as load_bg_model
from upscaler import load_model as load_upscale_model

# Load models once
bg_model = load_bg_model('u2net')
upscale_model = load_upscale_model()

# Process workflow
process_both('input.jpg', bg_model, upscale_model, scale=4, output_path='output.png')
```

## Model Selection Guide

**Background Removal:**

- `u2net`: Higher accuracy, larger file (176.3 MB), slower inference
- `u2netp`: Faster inference, smaller file (4.7 MB), slightly lower quality

**Upscaling:**

- `x4plus`: Single model supports 2x, 3x, 4x upscaling via outscale parameter
- Model structure is fixed at scale=4 (native scale)
- For 2x or 3x, model performs 4x upscaling then downscales to target scale
