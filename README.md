# Machine Learning - Sunspot Characteristics & Flare Forecasting

**Author**: Aoife McCloskey  
**Email**: mccloska@tcd.ie

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Description](#project-description)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data & Methodology](#data--methodology)
- [Models & Algorithms](#models--algorithms)
- [Results & Evaluation](#results--evaluation)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Overview

This repository contains machine learning implementations for predicting solar flares based on sunspot characteristics. The project builds upon traditional McIntosh classification-based forecasting methods by applying modern machine learning techniques to improve prediction accuracy and performance.

**Primary Goal**: Develop and evaluate ML-based solar flare prediction models to enhance operational flare forecast services.

---

## Project Description

### Background

Historically, McIntosh classifications of sunspots have been utilized for the prediction of solar flares. Modern operational flare forecast services still rely upon these classifications for their predictions. This project extends this approach by applying contemporary machine learning techniques.

### Objectives

1. **Build ML Models**: Construct a set of machine learning models to predict solar flares within a 24-hour period
2. **Compare Performance**: Evaluate ML approaches against traditional Poisson-based forecasting models
3. **Validate Results**: Train and test algorithms using data from multiple independent solar cycle periods
4. **Feature Analysis**: Explore the importance of individual McIntosh components on model performance
5. **Address Solar Cycle Dependency**: Investigate and mitigate solar cycle dependencies in predictions

### Key Contributions

- Application of multiple ML techniques (algorithms compared across comprehensive metrics)
- Cross-validation across multiple solar cycles to ensure robustness
- Skill score calculations for direct comparison with Poisson-based forecasts
- Feature importance analysis with physical interpretation
- Investigation of solar cycle effects on model generalization

---

## Key Features

✅ Multiple Machine Learning Algorithms  
✅ Cross-Solar-Cycle Validation  
✅ Comprehensive Performance Metrics (Skill Scores, etc.)  
✅ Feature Importance Analysis  
✅ Solar Cycle Dependency Analysis  
✅ McIntosh Classification-based Features  
✅ 24-hour Flare Prediction Window  

---

## Project Structure

```
ml_sunspots/
├── README.md                 # Project documentation (this file)
├── data/                     # Data files and datasets
│   ├── raw/                  # Original raw data from solar cycles
│   └── processed/            # Processed features and labels
├── src/                      # Source code
│   ├── preprocessing.py      # Data preprocessing and feature engineering
│   ├── models.py             # ML model definitions and training
│   ├── evaluation.py         # Performance metrics and evaluation functions
│   ├── feature_importance.py # Feature importance analysis
│   └── utils.py              # Utility functions
├── notebooks/                # Jupyter notebooks for exploration and analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_feature_analysis.ipynb
├── results/                  # Output results and figures
│   ├── models/               # Trained model files
│   ├── figures/              # Generated plots and visualizations
│   └── metrics/              # Performance metrics and reports
├── requirements.txt          # Python dependencies
└── config.py                 # Configuration parameters
```

---

## Getting Started

### Prerequisites

- Python 3.7+
- pip or conda package manager
- Basic familiarity with machine learning and solar physics concepts

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mccloska/ml_sunspots.git
   cd ml_sunspots
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the Pipeline

1. **Data Preprocessing**:
   ```python
   from src.preprocessing import prepare_data
   X_train, X_test, y_train, y_test = prepare_data('data/raw/')
   ```

2. **Model Training**:
   ```python
   from src.models import train_models
   models = train_models(X_train, y_train)
   ```

3. **Evaluation**:
   ```python
   from src.evaluation import evaluate_models
   results = evaluate_models(models, X_test, y_test)
   ```

4. **Feature Analysis**:
   ```python
   from src.feature_importance import analyze_features
   importance = analyze_features(models, feature_names)
   ```

### Jupyter Notebooks

Execute analysis notebooks in sequence:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_model_training.ipynb
jupyter notebook notebooks/03_evaluation.ipynb
jupyter notebook notebooks/04_feature_analysis.ipynb
```

---

## Data & Methodology

### Data Sources

- **Solar cycles**: Data from multiple independent solar cycle periods
- **McIntosh classifications**: Sunspot classification attributes
- **Flare labels**: Historical solar flare occurrence within 24-hour windows

### Feature Engineering

Features are derived from McIntosh classification components:
- **Sunspot classification codes**
- **Magnetic field properties**
- **Area and complexity metrics**
- **Historical flare frequency**

### Cross-Validation Strategy

- **Solar cycle-based splitting**: Models trained on certain cycles, tested on others
- **Time-series considerations**: Respects temporal dependencies in solar data
- **Multiple fold validation**: Ensures robust performance estimates

---

## Models & Algorithms

The project evaluates and compares multiple machine learning algorithms:

- **Logistic Regression**: Baseline linear classifier
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Advanced ensemble techniques
- **Support Vector Machines**: Non-linear classification
- **Neural Networks**: Deep learning approaches (where applicable)

Each model is evaluated against:
- **Poisson-based baseline forecasts** (operational standard)
- **Multiple performance metrics**
- **Cross-cycle generalization**

---

## Results & Evaluation

### Metrics

- **Skill Scores**: Direct comparison with operational Poisson forecasts
- **Accuracy, Precision, Recall**: Standard classification metrics
- **ROC-AUC**: Discrimination ability assessment
- **F1-Score**: Balanced performance metric
- **Confusion Matrices**: Detailed error analysis

### Key Findings

- ML models demonstrate improved performance across multiple metrics
- Feature importance analysis reveals critical McIntosh components
- Solar cycle dependency is significant; models show cycle-specific patterns
- Some models generalize well across solar cycles; others show cycle bias

### Visualizations

Results include:
- Performance comparison plots
- Feature importance rankings
- ROC curves
- Confusion matrices
- Solar cycle analysis visualizations

---

## Documentation

### Code Documentation

Each module includes:
- **Docstrings**: Function and class documentation
- **Type hints**: Parameter and return type annotations
- **Comments**: Explanations of complex logic

### Configuration

Modify `config.py` to adjust:
- Model hyperparameters
- Data paths
- Cross-validation folds
- Feature selection
- Output directories

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes with clear messages
4. Push to your branch
5. Open a Pull Request with a description of changes

### Guidelines

- Follow PEP 8 code style
- Add docstrings to new functions
- Include unit tests for new features
- Update README if adding new functionality
- Ensure all tests pass before submitting PR

---

## License

This project is provided as-is for research and educational purposes. Please cite appropriately if used in academic work.

---

## References

### Key Publications & Concepts

- **McIntosh Classification System**: Standard for sunspot characterization
- **Poisson-based Flare Forecasting**: Traditional operational approach
- **Solar Cycle Effects**: Understanding cycle-dependent solar activity patterns
- **Machine Learning in Solar Physics**: Emerging applications of ML to space weather prediction

### Recommended Reading

- Space Weather prediction literature
- Solar flare prediction studies
- Machine learning classification techniques
- Cross-validation methodologies for time-series data

---

## Contact & Support

For questions, issues, or suggestions regarding this project:

- **Author**: Aoife McCloskey
- **Email**: mccloska@tcd.ie
- **Repository**: https://github.com/mccloska/ml_sunspots

---

**Last Updated**: May 20, 2026  
**Status**: Active Research Project
