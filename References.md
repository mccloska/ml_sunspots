# References

## Solar Flare Prediction Literature

### Key Papers on McIntosh Classifications

1. **McIntosh, P. S. (1990).** "The Physics of Sunspot Magnetic Fields." *Solar Physics*
   - Original McIntosh classification system paper
   - Defines Zurich, Penumbra, and Spot Distribution classes
   - Foundational reference for all work using these classifications

2. **Klobucar, J., et al.** "Multimethod evaluation of sunspot detection and classification"
   - Analysis of McIntosh classification accuracy and variability
   - Discusses reliability of classification for forecasting

3. **Bloomfield, D. S., et al.** "Solar flare prediction using machine learning"
   - Modern ML approaches to solar flare forecasting
   - Comparison of algorithms and feature selection
   - Validation on multiple solar cycles

### Solar Flare Forecasting

4. **Abramenko, V. (2005).** "Magnetic Energy of a Sunspot as a Function of the Zurich Classification."
   - Relationship between McIntosh class and magnetic properties
   - Why Zurich class predicts flares

5. **Hanssen, A. C., & Kuipers, W. H. (1965).** "On the relationship between the frequency of rain and various meteorological parameters."
   - Original TSS (True Skill Statistic) definition
   - Now standard in forecast verification

6. **Brier, G. W. (1950).** "Verification of forecasts expressed in terms of probability."
   - Original definition of Brier Score
   - Basis for Brier Skill Score (BSS)

### Machine Learning for Solar Physics

7. **Sammis, I., et al. (2000).** "A comparison of fuzzy neural network and support vector machine classifiers for the prediction of solar flares."
   - Early ML approaches to flare prediction
   - Compares neural networks with traditional methods

8. **Akhaury, S., et al.** "Deep learning approaches for solar flare prediction."
   - Modern deep learning applications
   - Comparison with traditional ML

### Scikit-learn Documentation

9. **Pedregosa et al. (2011).** "Scikit-learn: Machine Learning in Python"
   - Primary reference for algorithms and metrics used
   - Available at: https://scikit-learn.org/
   - Comprehensive documentation for all ML algorithms

### Statistical Methods

10. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning*
    - Comprehensive ML theory
    - Cross-validation and regularization methods
    - Feature selection techniques

11. **James, G., et al. (2013).** *An Introduction to Statistical Learning*
    - More accessible version of ESLRP
    - Practical applications and interpretations

## Data Sources

### NOAA Space Weather Prediction Center

- **FTP Server:** ftp.swpc.noaa.gov
- **Data Products:**
  - Solar Region Summaries (SRS)
  - Solar Event Reports
  - Magnetometer data
  - Space weather forecasts

**Citation:** 
National Oceanic and Atmospheric Administration, Space Weather Prediction Center. Solar Region Summary Data. Available at https://www.swpc.noaa.gov/

### Solar Monitoring Resources

- **Solar Monitor:** https://www.solarmonitor.org/
  - Real-time solar observations
  - HMI, AIA, and other instruments
  - Current space weather forecasts

- **NOAA Space Weather:** https://www.swpc.noaa.gov/
  - Operational space weather products
  - 3-day forecasts
  - Historical data archives

## Software and Tools

### Machine Learning Libraries

- **scikit-learn:** https://scikit-learn.org/
  - ML algorithms used in this project
  - Documentation and examples

- **Pandas:** https://pandas.pydata.org/
  - Data manipulation and analysis
  - CSV reading and DataFrame operations

- **NumPy:** https://numpy.org/
  - Numerical computing
  - Array operations

- **Matplotlib:** https://matplotlib.org/
  - Plotting and visualization
  - ROC curves, box plots, histograms

- **SciPy:** https://www.scipy.org/
  - Scientific computing
  - Interpolation functions used in ROC analysis

### Python Documentation

- **Python 3:** https://docs.python.org/3/
- **Pickle module:** https://docs.python.org/3/library/pickle.html
- **FTP library:** https://docs.python.org/3/library/ftplib.html

## Related Projects

### Similar Solar ML Research

- **SHARP dataset:** Machine Learning-ready magnetic field data from HMI
- **FLARECAST:** EU-funded flare forecasting project
- **SODA:** Solar Orbiter Data Analysis package

## Statistical References

### Forecast Verification

12. **Mason, I. (1982).** "A Model for Assessment of Weather Forecasts"
    - Basis for skill score definitions
    - Comparison metrics between forecasts

13. **Jolliffe, I. T., & Stephenson, D. B. (2012).** *Forecast Verification: A Practitioner's Guide in Atmospheric Science*
    - Comprehensive verification methodology
    - Choice and interpretation of metrics

### Class Imbalance and Evaluation

14. **Chawla, N. V., et al. (2002).** "SMOTE: Synthetic Minority Over-sampling Technique"
    - Handling imbalanced datasets
    - Relevant for flare prediction (few events, many quiet periods)

15. **Fawcett, T. (2006).** "An Introduction to ROC Analysis"
    - Detailed ROC curve methodology
    - AUC interpretation and use

## Cross-Validation and Model Selection

16. **Kohavi, R. (1995).** "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection"
    - K-Fold cross-validation theory
    - Why stratification matters for imbalanced data

17. **Bergstra, J., & Bengio, Y. (2012).** "Random Search for Hyper-Parameter Optimization"
    - Hyperparameter tuning approaches
    - Relevant for algorithm improvement

## Related Operational Forecasting

### Space Weather Prediction Operations

- **NOAA Forecast Centers:** https://www.swpc.noaa.gov/
  - Operational flare forecasts
  - Real-time prediction accuracy verification
  - 3-day and 27-day forecasts

- **ISES (International Space Environment Service):** https://www.ises-spaceweather.org/
  - International forecast coordination
  - Benchmark forecasts
  - Performance metrics

## Suggested Reading Order

**For getting started:**
1. McIntosh (1990) - Understand the classification system
2. This project's README and wiki pages
3. Scikit-learn documentation - Understand algorithms

**For understanding forecasting:**
1. Hanssen & Kuipers (1965) - TSS definition
2. Brier (1950) - BSS definition
3. Jolliffe & Stephenson (2012) - Verification methods
4. Fawcett (2006) - ROC analysis

**For improving the project:**
1. Hastie et al. (2009) - ML theory
2. Kohavi (1995) - Cross-validation
3. Related ML papers on solar flares

**For operational deployment:**
1. NOAA operational procedures
2. Mason (1982) - Skill score comparison
3. Chawla et al. (2002) - Handling class imbalance

## How to Cite This Project

If using this code in research:

```bibtex
@software{mccloska2020mlsunspots,
  author = {McCloskey, Aoife},
  title = {Machine Learning for Sunspot Characteristics and Flare Forecasting},
  year = {2020},
  url = {https://github.com/mccloska/ml_sunspots},
  note = {GitHub repository}
}
```

Or in text:
> McCloskey, A. (2020). Machine Learning for Sunspot Characteristics and Flare Forecasting. Retrieved from https://github.com/mccloska/ml_sunspots

## Contact and Collaboration

**Author:** Aoife McCloskey  
**Email:** mccloska@tcd.ie

## License

Check repository for license information.

---

*Last updated: 2024*
*For the latest references and resources, see the main project repository.*
