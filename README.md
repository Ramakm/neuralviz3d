# Neural Network Visualizer

**Interactive 3D visualization of neural networks performing MNIST handwritten digit recognition in real-time.**

Draw digits and watch activations flow through the network layers in beautiful 3D!

> This project is inspired from with enhanced documentation & functionalities which is gonna be added.

![Neural Network Visualization](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Three.js](https://img.shields.io/badge/Three.js-r128-orange)

## Features

- **Interactive Drawing Canvas**: Draw digits on a 28×28 grid with mouse controls
- **Real-time 3D Visualization**: Watch neural activations flow through network layers
- **Live Predictions**: See probability distributions for all digit classes
- **Customizable Settings**: Adjust visualization parameters in real-time
- **Pre-trained Samples**: Quick-test with sample digits
- **Responsive Design**: Works on desktop and mobile devices
- **Hardware Acceleration**: Supports Apple Metal (MPS), CUDA, and CPU

## Quick Start

### Prerequisites

- Python 3.8+ (for training)
- Modern web browser with WebGL support
- (Optional) CUDA or Apple Silicon for accelerated training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/NN-visualizer.git
   cd NN-visualizer
   ```

2. **Install Python dependencies** (only for training)
   ```bash
   pip install -r requirements.txt
   ```

3. **Start a local web server**
   ```bash
   # Using Python
   python3 -m http.server 8000
   
   # Or using Node.js
   npx http-server -p 8000
   ```

4. **Open in browser**
   ```
   http://localhost:8000
   ```

## Training Your Own Model

The visualizer comes with a training script to create custom models:

```bash
python training/mlp_train.py \
  --epochs 10 \
  --hidden-dims 64 32 \
  --batch-size 128 \
  --export-path exports/mlp_weights.json
```

### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 5 | Number of training epochs |
| `--batch-size` | 128 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--hidden-dims` | 64 32 | Hidden layer dimensions |
| `--device` | auto | Device (auto/mps/cuda/cpu) |
| `--export-path` | exports/mlp_weights.json | Output path |
| `--data-dir` | ./data | MNIST dataset directory |

### Example Training Output

```bash
Using device: mps

Loading MNIST dataset...
  - Training samples: 60,000
  - Test samples: 10,000

Building model with architecture: 784 → 64 → 32 → 10
  - Total parameters: 51,946

Training for 10 epochs...

Epoch 1/10
----------------------------------------------------------
  Batch 100/469 | Loss: 0.5234 | Acc: 85.32%
  ...
  Train - Loss: 0.3421 | Accuracy: 89.45%
  Test  - Loss: 0.2156 | Accuracy: 93.67%
  New best accuracy: 93.67%

...

Training complete!
  - Best test accuracy: 97.82%
```

## Project Structure

```
NN-visualizer/
├── index.html              # Main HTML file
├── assets/
│   ├── main.css           # Modern styling
│   └── main.js            # Application logic & 3D visualization
├── training/
│   └── mlp_train.py       # PyTorch training script
├── exports/
│   └── mlp_weights.json   # Trained model weights
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Architecture

### Default Network

- **Input Layer**: 784 neurons (28×28 pixels)
- **Hidden Layer 1**: 64 neurons (ReLU activation)
- **Hidden Layer 2**: 32 neurons (ReLU activation)
- **Output Layer**: 10 neurons (Softmax for digit classes 0-9)

Total Parameters: ~51,946

### Customization

You can modify the network architecture by training with different `--hidden-dims`:

```bash
# Larger network
python training/mlp_train.py --hidden-dims 128 64 32

# Smaller network
python training/mlp_train.py --hidden-dims 32 16

# Deep network
python training/mlp_train.py --hidden-dims 128 64 64 32
```

## How to Use

### Drawing

- **Left-click + Drag**: Draw on the canvas
- **Right-click + Drag**: Erase
- **Reset Button**: Clear the canvas

### 3D Controls

- **Left-click + Drag**: Rotate view
- **Right-click + Drag**: Pan view
- **Scroll Wheel**: Zoom in/out

### Sample Digits

Click any number button (0-9) to load a sample digit for quick testing.

### Advanced Settings

Click the settings icon to adjust:

- **Max Connections Per Neuron**: Visualization detail level
- **Connection Weight Threshold**: Hide weak connections
- **Connection Line Thickness**: Visual appearance
- **Drawing Brush Size**: Canvas drawing size
- **Drawing Intensity**: Stroke opacity

## Color Coding

### Neurons

- **Blue tones**: Low or negative activations
- **Purple/Pink tones**: Strong positive activations
- **Size**: Scales with activation strength

### Connections

- **Intensity**: Represents weight magnitude
- **Color gradient**: Shows contribution strength

## Technical Details

### Frontend

- **Three.js (r128)**: 3D graphics rendering
- **Vanilla JavaScript**: No framework dependencies
- **Modern CSS**: Custom properties, Grid, Flexbox
- **WebGL**: Hardware-accelerated graphics

### Backend (Training)

- **PyTorch**: Deep learning framework
- **MNIST Dataset**: 70,000 handwritten digit images
- **Optimizers**: Adam optimizer
- **Loss**: Cross-entropy loss

### Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance

- **Training**: 2-5 minutes on Apple M1/M2
- **Inference**: Real-time (<16ms per frame)
- **Memory**: ~100MB for visualization

## Contributing

Contributions are welcome! Here are some ways you can help:

- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original concept inspired by [DFin/Neural-Network-Visualisation]
- MNIST dataset by Yann LeCun
- Three.js community

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Made with care for education and learning**
