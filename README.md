# Multimodal Contrastive Learning for Clinical Data Alignment via Fat Composition Representations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Website](https://img.shields.io/badge/Website-Online-brightgreen.svg)](https://tech4dlab.github.io/multimodal-contrastive-learning-for-clinical-data-alignment-via-fat-composition-representations)

## Overview

![Methodology Overview](docs/static/images/methodology.png)

This repository implements a multimodal contrastive learning framework inspired by CLIP, specifically designed to align seven heterogeneous clinical modalities related to obesity research. The framework uses fat composition as an anchor modality to create meaningful representations across different types of clinical data.

### Key Features

- **Seven Clinical Modalities**: Fat composition, muscle composition, biochemistry, anthropometry, demographics, metabolic profile, and cardiovascular physiology
- **CLIP-inspired Architecture**: Contrastive learning framework with dedicated neural network encoders for each modality
- **Controllable Training**: Alpha parameter to balance anchor-modality alignment vs. inter-modality relationships
- **Missing Data Imputation**: Similarity-based reconstruction of missing modalities in latent space
- **Comprehensive Evaluation**: Benchmarked against traditional imputation methods (KNN, MICE)

## Repository Structure

```
multimodal-contrastive-learning-for-clinical-data-alignment-via-fat-composition-representations/
├── README.md                          # This file
├── LICENSE                           # MIT License
├── .gitignore                       # Git ignore rules
├── index.html                       # Main project website
├── project_site/                    # Academic project page
│   ├── index.html                  # Project presentation page
│   └── static/                     # Web assets (CSS, JS, images, etc.)
├── src/                            # Source code
│   ├── code/
│   │   ├── clip_tab.py            # Main CLIP-TAB implementation
│   │   └── methods_comparison.py  # Comparison with baseline methods
│   ├── embeddings/                # Generated embeddings storage
│   │   └── exp10.1/               # Experiment results
│   ├── figures/                   # Generated figures and plots
│   └── models_weights/            # Trained model weights
│       └── exp10.1/
└── static/                        # Static assets (duplicate for GitHub Pages)
    ├── css/                       # Stylesheets
    ├── images/                    # Research images and visualizations
    ├── js/                        # JavaScript files
```

## Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Minimum 8GB RAM
- 2GB free disk space

### Python Dependencies

```bash
# Core ML libraries
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Data processing
standardscaler
tsne

# Utilities
dotenv
rich
argparse
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bernabe19/multimodal-contrastive-learning-for-clinical-data-alignment-via-fat-composition-representations.git
   cd multimodal-contrastive-learning-for-clinical-data-alignment-via-fat-composition-representations
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn plotly python-dotenv rich
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   DATA_PATH=path/to/your/clinical/data
   EMBEDDINGS_PATH=src/embeddings/
   FIGURES_PATH=src/figures/
   MODELS_PATH=src/models_weights/
   ```

## Usage

### Data Preparation

Your clinical data should be organized in a CSV file with the following modalities:

- **Fat Composition**: `["Visceral Fat", "Total Fat", "Fat Right Leg", "Fat Left Leg", "Fat Right Arm", "Fat Left Arm", "Fat Trunk"]`
- **Biochemical**: `["Cholesterol", "Glucose"]`
- **Anthropometric**: `["Height", "Weight", "Total Muscle", "Wrist", "Waist", "Hip", "WHR"]`
- **Muscle Composition**: `["Total Muscle", "Muscle Right Leg", "Muscle Left Leg", "Muscle Right Arm", "Muscle Left Arm", "Muscle Trunk"]`
- **Demographics**: `["Age", "Gender"]`
- **Metabolic Profile**: `["BMR", "TEE", "Activity"]`
- **Cardiovascular Physiology**: `["Systolic", "Diastolic"]`

### Training the Model

1. **Basic training** (single alpha value):
   ```bash
   cd src/code
   python clip_tab.py
   ```

2. **Hyperparameter search** (multiple alpha values):
   ```bash
   # Edit clip_tab.py and set: search = True
   python clip_tab.py
   ```

### Key Parameters

- `alpha`: Controls balance between anchor-modality loss and inter-modality loss (0.0-1.0)
- `epochs`: Number of training epochs (default: 2000)
- `batch_size`: Training batch size (default: 64)
- `temperature`: Temperature parameter for contrastive loss (default: 0.1)
- `hidden_dim`: Hidden dimension for MLP encoders (default: 1024)
- `out_dim`: Output embedding dimension (default: 128)

### Evaluating Imputation Performance

```bash
cd src/code
python methods_comparison.py --modality bioq --num_ret 4 --method cosine
```

Parameters:
- `--modality`: Modality to impute (`bioq`, `antro`, `muscle`, `demo`, `meta`, `physio`)
- `--num_ret`: Number of neighbors for retrieval (default: 4)
- `--method`: Similarity method (`cosine`, `euclidean`)

## Model Architecture

### MLP Encoder
Each modality is processed by a dedicated MLP encoder:
```
Input → Linear(input_dim, 1024) → BatchNorm → ReLU → Dropout(0.3) → Linear(1024, 128)
```

### Contrastive Loss
The framework uses a symmetric CLIP-style contrastive loss with two components:
1. **Anchor Loss**: Fat composition aligned with each other modality
2. **Full Loss**: All pairwise modality alignments

Final loss: `L = α × L_anchor + (1-α) × L_full`

## Experimental Results

The framework demonstrates competitive or superior performance compared to traditional imputation methods:
- **KNN Imputation**: Using k-nearest neighbors on individual or all modalities
- **MICE**: Multiple Imputation by Chained Equations
- **Our Method**: Similarity-based imputation in learned embedding space

## Restrictions and Limitations

### ⚠️ Important Restrictions

1. **Research Use Only**:
   - This implementation is for research purposes
   - Not validated for clinical decision-making
   - Requires proper medical supervision for any clinical applications

2. **Data Requirements**:
   - Requires specific clinical modalities as defined above
   - Data must be properly normalized and preprocessed
   - Missing data patterns may affect performance

3. **Hyperparameter Sensitivity**:
   - Alpha parameter significantly affects alignment quality
   - Temperature parameter affects contrastive learning dynamics
   - May require dataset-specific tuning

### Technical Limitations

- **Fixed Architecture**: MLP encoders may not capture complex relationships
- **Scalability**: Performance on very large datasets not extensively tested
- **Generalization**: Trained on specific clinical domains and populations
- **Evaluation Metrics**: Limited to MAE and MRE for imputation quality


## Citation

If you use this code in your research, please cite:

```bibtex
@InProceedings{10.1007/978-3-032-16992-1_5,
author="S{\'a}nchez-Sos, Bernab{\'e}
and Tom{\'e}-Fern{\'a}ndez, Mario
and Bevi{\'a}-Ballesteros, Ismael
and Jerez-Tall{\'o}n, Mario
and Garc{\'i}a-D'Urso, Nahuel
and Azor{\'i}n-L{\'o}pez, Jorge
and Saval-Calvo, Marcelo
and Fuster-Guill{\'o}, Andr{\'e}s
and S{\'a}nchez-SanSegundo, Miriam",
editor="Bravo, Jos{\'e}
and Fontecha, Jes{\'u}s
and Ballesteros, Joaqu{\'i}n",
title="Multimodal Contrastive Learning for Clinical Data Alignment via Fat Composition Representations",
booktitle="Proceedings of the International Conference on Ubiquitous Computing and Ambient Intelligence (UCAmI 2025), Volume 1",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="45--50",
abstract="This work proposes a multimodal contrastive learning framework inspired by CLIP, designed to align seven heterogeneous clinical modalities related to obesity: fat composition, muscle composition, biochemistry, anthropometry, demographics, metabolic profile, and cardiovascular physiology. Each modality is encoded through a dedicated neural network and projected into a shared latent space, with fat composition used as the anchor modality during training. A control parameter is introduced to balance the intra-modality contribution and fat-modality alignment, enhancing cluster separability and semantic coherence. Furthermore, the framework is evaluated for its capacity to reconstruct missing modalities in the latent space through similarity-based imputation. The method is benchmarked against traditional imputation approaches such as KNN and MICE, showing competitive or superior performance across multiple clinical targets. The results demonstrate the framework's potential to effectively align complex clinical data and enable meaningful patient stratification, contributing to more personalized and accurate obesity phenotyping in clinical and research settings. https://tech4dlab.github.io/multimodal-contrastive-learning-for-clinical-data-alignment-via-fat-composition-representations",
isbn="978-3-032-16992-1"
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was supported by grant PID2023-149562OB-I00 funded by MICIU/AEI/
10.13039/501100011033 and by ERDF/EU and by grant CIAICO/2022/132 funded
by Conselleria d’Innovació, Universitat, Ciència i Societat Digital. Predoctoral support
(PREP2023-001644) was provided by MICIU/AEI/10.13039/501100011033 and ESF+

## Contact

- **Bernabé Sánchez-Sos** - [LinkedIn Profile](https://www.linkedin.com/in/bernabé-sánchez-591479257/)
- **Mario Tomé Fernández** - [LinkedIn Profile](https://www.linkedin.com/in/mario-tomé-fernández-86a308352/)

---

**Disclaimer**: This research code is provided for educational and research purposes only. It should not be used for clinical decision-making without proper validation and medical oversight.
