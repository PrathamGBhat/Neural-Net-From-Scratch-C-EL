# Neural Net Visualizer — C Implementation

A from-scratch implementation of a multi-layer feedforward neural network written in pure C. The network learns the **4-bit parity (XOR) function** — a classic non-linear classification problem — using backpropagation and gradient descent, with no external ML libraries.

---

## Architecture

```
Input (4)  →  Hidden Layer 1 (8)  →  Hidden Layer 2 (4)  →  Output (1)
```

| Layer          | Neurons | Activation |
|----------------|---------|------------|
| Input          | 4       | —          |
| Hidden Layer 1 | 8       | Sigmoid    |
| Hidden Layer 2 | 4       | Sigmoid    |
| Output         | 1       | Sigmoid    |

- **Loss function**: Mean Squared Error (MSE)
- **Optimizer**: Vanilla Gradient Descent
- **Default Learning Rate**: `0.5`
- **Default Epochs**: `100,000`

---

## Dataset

The network trains on all **16 possible 4-bit binary input vectors** and learns to predict the **parity bit** (1 if an odd number of inputs are 1, else 0).

| Input          | Label |
|----------------|-------|
| `{0, 0, 0, 0}` | 0     |
| `{0, 0, 0, 1}` | 1     |
| `{0, 0, 1, 0}` | 1     |
| `{0, 0, 1, 1}` | 0     |
| ...            | ...   |

---

## Project Structure

```
Neural-Net-Visualizer-C-EL-main/
├── main.c                    # Training loop and evaluation
├── layers/
│   ├── dataset.h / .c        # Input data and labels (16 samples)
│   ├── layer1.h  / .c        # Hidden Layer 1 (8 neurons)
│   ├── layer2.h  / .c        # Hidden Layer 2 (4 neurons)
│   └── output_layer.h / .c   # Output Layer (1 neuron)
└── utils/
    ├── utils.h  / .c         # sigmoid, sigmoid_deriv, randf helpers
```

---

## Building & Running

### Prerequisites
- GCC (or any C99-compatible compiler)
- `libm` (standard math library, included on macOS/Linux by default)

### Compile

```bash
gcc main.c layers/dataset.c layers/layer1.c layers/layer2.c layers/output_layer.c utils/utils.c -o main -lm
```

### Run

```bash
./main
```

### Expected Output

```
Epoch 500    Loss: 0.124998  Accuracy: 50.0%
...
Epoch 21500  Loss: 0.000696  Accuracy: 100.0%
...
Epoch 100000 Loss: 0.000004  Accuracy: 100.0%
```

The network converges to **100% accuracy** typically within ~22,000 epochs.

---

## Configuration

Edit the defines at the top of `main.c` to tune training:

```c
#define EPOCHS 100000   // Number of training epochs
#define LR     0.5f     // Learning rate
```

---

## Key Implementation Details

### Weight Initialization
Weights are randomly initialized in the range `[-1.0, 1.0]` using:
```c
// utils/utils.c
float randf(void) {
    return (((float)rand() / (float)RAND_MAX) - 0.5f) * 2.0f;
}
```
> **Note:** A wider initialization range is critical for this XOR/parity problem. A narrow range (e.g., `±0.25`) causes the network to get stuck in local minima at ~93.8% accuracy.

### Backpropagation
Each layer implements three functions:
- `layer_forward()` — computes activations via sigmoid
- `layer_backward()` — computes deltas via chain rule
- `layer_update()` — applies weight updates via gradient descent

---

## Results

| Metric         | Value          |
|----------------|----------------|
| Final Accuracy | **100%**       |
| Final Loss     | ~0.000004      |
| Convergence    | ~22,000 epochs |
| Dataset Size   | 16 samples     |

---

## License

This project is open source and available under the [MIT License](LICENSE).
