# Deep Learning Architectures & Pure Object-Oriented Framework
**Research Internship — Week 2 Technical Documentation**  
**Sardar Ahmed**  
**Framework:** TensorFlow / Keras 3.x with Pure Python OOP Architecture

---

## 1. Overview & Objectives

This project establishes a comprehensive, production-grade Deep Learning benchmark and architectural framework across four foundational neural network architectures:
1. **Artificial Neural Network (ANN / Multi-Layer Perceptron)** — MNIST Digit Classification
2. **Convolutional Neural Network (CNN)** — MNIST Spatial Feature Classification
3. **Recurrent Neural Network (SimpleRNN)** — IMDB Movie Review Sentiment Analysis
4. **Long Short-Term Memory (LSTM)** — IMDB Long-Range Sequential Sentiment Analysis

The codebase adheres strictly to **Object-Oriented Programming (OOP)** principles:
- **Single Responsibility Principle (SRP):** Each neural network architecture is encapsulated in its own dedicated, reusable class with modular lifecycle methods (`load_data`, `preprocess_data`, `build_model`, `compile_model`, `train`, `evaluate`, `predict`, `plot_training_history`, `plot_sample_predictions`, `save_model`, `run`).
- **Master Pipeline Orchestrator (`Main.py`):** The `DeepLearningPipeline` class coordinates sequential execution, captures metrics, benchmarks performance across domains, and exports serialized artifacts and plots.
- **Reproducibility & Modularity:** Configurable hyperparameters in `__init__`, automated artifact directory creation (`Models/`, `Visualizations/`, `ModelComparison/`), and dual execution support (standalone module vs. pipeline).

---

## 2. Architectural Deep-Dive

### 2.1 Artificial Neural Network (ANN / MLP)
- **Dataset:** MNIST ($70,000$ grayscale handwritten digit images of size $28 \times 28$, $10$ classes: digits 0–9).
- **Mathematical Pipeline:**
  $$\mathbf{x}_{\text{flat}} = \text{vec}(\mathbf{X}) \in \mathbb{R}^{784}$$
  $$\mathbf{h}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{x}_{\text{flat}} + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{128 \times 784}$$
  $$\mathbf{h}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{64 \times 128}$$
  $$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3), \quad \mathbf{W}_3 \in \mathbb{R}^{10 \times 64}$$
- **Limitations:** Treats spatial 2D grid pixels as an unrolled 1D vector, discarding 2D spatial correlations and translation invariance.

### 2.2 Convolutional Neural Network (CNN)
- **Dataset:** MNIST (reshaped to $\mathbb{R}^{N \times 28 \times 28 \times 1}$).
- **Mathematical Pipeline:**
  - **Conv2D Layer 1:** 32 filters, kernel size $3 \times 3$, ReLU activation:
    $$S(i, j) = (I * K)(i, j) = \sum_m \sum_n I(i-m, j-n) K(m, n)$$
  - **MaxPooling2D Layer 1:** $2 \times 2$ window reducing spatial dimensions by $2\times$.
  - **Conv2D Layer 2:** 64 filters, kernel size $3 \times 3$, ReLU activation.
  - **MaxPooling2D Layer 2:** $2 \times 2$ window.
  - **Dense Feature & Regularization:** Flatten $\rightarrow$ Dense(128, ReLU) $\rightarrow$ Dropout(rate = 0.5) $\rightarrow$ Dense(10, Softmax).
- **Advantages:** Weight sharing and local receptive fields achieve translation invariance and superior generalization.

### 2.3 Recurrent Neural Network (SimpleRNN)
- **Dataset:** IMDB Movie Reviews ($50,000$ reviews, vocabulary size = 10,000, padded to sequence length $T = 200$).
- **Mathematical Pipeline:**
  - **Embedding:** Maps discrete token indices $w_t \in \{1, \dots, V\}$ to dense vectors $\mathbf{x}_t \in \mathbb{R}^{32}$.
  - **Hidden State Recurrence:**
    $$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h), \quad \mathbf{h}_t \in \mathbb{R}^{64}$$
  - **Output Classification:**
    $$\hat{y} = \sigma(\mathbf{W}_{hy} \mathbf{h}_T + b_y) \in [0, 1]$$
- **Limitations:** Suffers from vanishing and exploding gradients when backpropagating through long sequences ($T = 200$), hindering memory of early tokens.

### 2.4 Long Short-Term Memory (LSTM)
- **Mathematical Pipeline:**
  - **Forget Gate:** Decides what past memory to discard:
    $$\mathbf{f}_t = \sigma(\mathbf{W}_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$
  - **Input Gate & Candidate Cell:** Decides what new information to store:
    $$\mathbf{i}_t = \sigma(\mathbf{W}_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$
    $$\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$
  - **Cell State Update (Constant Error Carousel):**
    $$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
  - **Output Gate & Hidden State:**
    $$\mathbf{o}_t = \sigma(\mathbf{W}_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$
    $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$
  - **Output Classification:** $\hat{y} = \sigma(\mathbf{W}_y \mathbf{h}_T + b_y)$
- **Advantages:** The additive cell state update $\mathbf{C}_t$ provides a highway for gradient flow, eliminating vanishing gradients over extended review sequences.

---

## 3. Directory Layout & Artifacts

```
Week 2/
├── Models/                                  # Serialized Trained Model Artifacts (.keras)
│   ├── ann_mnist_model.keras
│   ├── cnn_mnist_model.keras
│   ├── rnn_imdb_model.keras
│   └── lstm_imdb_model.keras
├── Visualizations/                          # Training curves, sample predictions, confusion matrices
│   ├── ann_training_history.png
│   ├── ann_sample_predictions.png
│   ├── cnn_training_history.png
│   ├── cnn_sample_predictions.png
│   ├── rnn_training_history.png
│   ├── rnn_sample_predictions.png
│   ├── lstm_training_history.png
│   └── lstm_sample_predictions.png
├── ModelComparison/                         # Comparative Benchmarks & Charts
│   ├── deep_learning_benchmark_results.csv
│   ├── vision_models_comparison.png
│   ├── sequence_models_comparison.png
│   └── overall_dl_benchmark_comparison.png
│
├── Step_1_ANN.py                            # class ArtificialNeuralNetwork
├── Step_2_CNN.py                            # class ConvolutionalNeuralNetwork
├── Step_3_RNN.py                            # class RecurrentNeuralNetwork
├── Step_4_LSTM.py                           # class LSTMNeuralNetwork
├── Step_5_Model_Comparison.py               # class DLModelComparator
├── Main.py                                  # class DeepLearningPipeline (Master Orchestrator)
│
├── ANN.py                                   # Entry Point for ANN
├── CNN.py                                   # Entry Point for CNN
├── RNN.py                                   # Entry Point for RNN
├── LSTM.py                                  # Entry Point for LSTM
│
├── DeepLearningImplementations.ipynb        # Master Interactive Jupyter Notebook
├── README.md                                # Full Project Documentation & Mermaid Diagrams
├── Documentation.md                         # Detailed Architectural & Mathematical Breakdown
├── Report.pdf                               # Formal Publication-Ready PDF Report
└── requirements.txt                         # Dependency specifications
```

---

## 4. Execution Guide

### Running Master Pipeline:
```bash
python Main.py
```

### Running Individual Modular Steps:
```bash
python Step_1_ANN.py
python Step_2_CNN.py
python Step_3_RNN.py
python Step_4_LSTM.py
python Step_5_Model_Comparison.py
```
