# Multimodal Contrastive Learning for Clinical Data Alignment via Fat Composition Representations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## Overview

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
    ├── pdfs/                      # Papers and documentation
    └── videos/                    # Demo videos
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

1. **Data Privacy**: 
   - This code handles sensitive clinical data
   - Ensure compliance with HIPAA, GDPR, or relevant data protection regulations
   - Never commit real patient data to version control
   - Use anonymized/synthetic data for testing

2. **Research Use Only**:
   - This implementation is for research purposes
   - Not validated for clinical decision-making
   - Requires proper medical supervision for any clinical applications

3. **Data Requirements**:
   - Requires specific clinical modalities as defined above
   - Data must be properly normalized and preprocessed
   - Missing data patterns may affect performance

4. **Computational Constraints**:
   - Training requires significant computational resources
   - GPU recommended for reasonable training times
   - Memory requirements scale with dataset size

5. **Hyperparameter Sensitivity**:
   - Alpha parameter significantly affects alignment quality
   - Temperature parameter affects contrastive learning dynamics
   - May require dataset-specific tuning

### Technical Limitations

- **Fixed Architecture**: MLP encoders may not capture complex relationships
- **Scalability**: Performance on very large datasets not extensively tested
- **Generalization**: Trained on specific clinical domains and populations
- **Evaluation Metrics**: Limited to MAE and MRE for imputation quality

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings for all functions and classes
- Include unit tests for new functionality
- Update documentation for any API changes

## Citation

If you use this code in your research, please cite:

```bibtex
@article{SanchezSos2054MultimodalContrastive,
  title={Multimodal Contrastive Learning for Clinical Data Alignment via Fat Composition Representations},
  author={Bernabé Sánchez-Sos and Mario Tomé-Fernández and Ismael Beviá-Ballesteros and Mario Jerez-Tallón and Nahuel García-D'Urso and Jorge Azorín-López and Marcelo Saval-Calvo and Andrés Fuster-Guilló and Miriam Sánchez-SanSegundo},
  journal={UCAmI 2025},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- University of Alicante - Department of Computer Technology
- University of Alicante - Department of Health Psychology
- UCAmI 2025 Conference
- CLIP paper authors for the original contrastive learning framework

## Contact

- **Bernabé Sánchez-Sos** - [ResearchGate Profile](https://www.researchgate.net/profile/Bernabe-Sanchez-Sos)
- **Mario Tomé Fernández** - [LinkedIn Profile](https://www.linkedin.com/in/mario-tomé-fernández-86a308352/)
- **Project Repository** - [GitHub](https://github.com/Bernabe19/multimodal-contrastive-learning-for-clinical-data-alignment-via-fat-composition-representations)

---

**Disclaimer**: This research code is provided for educational and research purposes only. It should not be used for clinical decision-making without proper validation and medical oversight.
