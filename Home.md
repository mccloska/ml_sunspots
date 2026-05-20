# Machine Learning - Sunspot Characteristics & Flare Forecasting

**Author**: Aoife McCloskey; mccloska@tcd.ie

## Project Overview

This project applies machine learning algorithms to predict solar flares using McIntosh classifications of sunspots. Historically, McIntosh classifications have been utilized for flare prediction, and modern operational services still rely on these classifications. This project explores how contemporary machine learning techniques can improve upon traditional approaches.

## Key Objectives

- **Train and test ML algorithms** using data from independent solar cycle periods
- **Apply cross-validation techniques** to ensure robust model evaluation
- **Compare relative performance** of different algorithms
- **Improve flare forecasting accuracy** beyond traditional classification methods

## Quick Links

- [[Project Structure|Project-Structure]]
- [[Data Description|Data-Description]]
- [[Algorithms & Methods|Algorithms-and-Methods]]
- [[Results & Evaluation|Results-and-Evaluation]]
- [[Getting Started|Getting-Started]]
- [[References|References]]

## Repository Structure

```
ml_sunspots/
├── calibration_curve_plot.py      # Calibration curve visualization
├── get_data.py                    # FTP data retrieval from NOAA
├── machine_learn_kfold.py         # K-Fold cross-validation ML pipeline
├── mci_classification_ml.py       # McIntosh classification utilities
├── ss_custom.py                   # Custom scoring functions
├── data/                          # Data directory (SRS and events data)
└── README.md                      # Project documentation
```

## Technology Stack

- **Language**: Python 100%
- **ML Framework**: scikit-learn
- **Data Processing**: pandas, NumPy
- **Visualization**: matplotlib
- **Data Source**: NOAA SWPC FTP server

## Key Features

✅ **K-Fold Cross-Validation** - Stratified sampling across solar cycles
✅ **Multiple Algorithms** - LR, LDA, KNN, CART, RFC
✅ **Custom Metrics** - BSS (Brier Skill Score) and TSS (True Skill Statistic)
✅ **Calibration Analysis** - Reliability diagrams and probability calibration
✅ **Automated Data Retrieval** - Direct FTP access to NOAA solar data
✅ **Feature Importance Analysis** - Understanding model decision-making

## Getting Started

1. Clone the repository
2. Set up data folder and retrieve data using `get_data.py`
3. Run `machine_learn_kfold.py` with your preferred method
4. View results and calibration curves

For detailed instructions, see [[Getting Started|Getting-Started]].

---

*For more information, see the [[References|References]] section for papers and resources on solar flare prediction and McIntosh classifications.*
