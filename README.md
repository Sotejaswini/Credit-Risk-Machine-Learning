# Credit Risk Modelling & Analytics Platform

End-to-End credit risk machine learning project built with Python, scikit-learn, XGBoost, Jupyter Notebook and Streamlit.

## Project objectives

- Estimate Probability of Default (PD) from historical borrower attributes.
- Compare Logistic Regression, Random Forest and XGBoost.
- Evaluate discrimination using ROC-AUC, Gini and KS.
- Review probability quality using Brier score and calibration plots.
- Engineer repayment behaviour and credit utilization features.
- Demonstrate PD-to-score mapping and risk bands.
- Monitor population distribution shift using PSI.
- Demonstrate IFRS 9 staging concepts and scenario-weighted ECL arithmetic.

## Dataset

- UCI Default of Credit Card Clients: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

- The included raw ZIP is the dataset supplied for this project. It contains 30,000 historical Taiwan credit-card customer records. The target is whether the customer defaulted in the following month.

**Limitations:** this is not an Indian NBFC dataset and it is not a production credit underwriting dataset. The target is a next-month default indicator. It does not provide a reliable monthly observation timeline, contractual loan cash flows, lifetime default curves, initial-recognition risk, EIR, or macroeconomic scenarios needed for a production IFRS 9 engine.

## Folder structure

```text
Credit-Risk-ML/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   ├── 05_PSI_Monitoring.ipynb
│   └── 06_IFRS9_ECL.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluation.py
│   ├── credit_scoring.py
│   ├── monitoring.py
│   └── ifrs9.py
├── models/
├── reports/
├── dashboard/
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup (Ubuntu / WSL Terminal)

```bash
cd Credit-Risk-ML
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Run notebooks

Start Jupyter from the project root:
```bash
jupyter notebook
```

Run notebooks in this order:
1. `01_EDA.ipynb`
2. `02_Feature_Engineering.ipynb`
3. `03_Model_Training.ipynb`
4. `04_Model_Evaluation.ipynb`
5. `05_PSI_Monitoring.ipynb`
6. `06_IFRS9_ECL.ipynb`

In Jupyter, ensure the working directory is the project root or open the notebook from `notebooks/`. The notebooks include a path setup cell for both cases.

## Train from terminal (optional)

```bash
python -m src.train
```

## Run dashboard

After running notebook 03:
```bash
streamlit run dashboard/app.py
```

## Methodology notes

- ID is excluded from model predictors.
- Numeric missing-value imputation and scaling are fitted inside sklearn pipelines on training data.
- Categorical variables are one-hot encoded in the pipeline.
- A stratified random holdout is used as a baseline because the dataset does not contain a trustworthy observation date. This is not a substitute for out-of-time validation.
- The default 0.50 classification threshold is illustrative and not an approval policy.
- PSI thresholds are heuristics and require institution-specific governance.
- Credit score mapping is illustrative and not a calibrated bureau scorecard.
- IFRS 9 functions are educational only. Production ECL requires institution-specific methodology, lifetime PD, EAD, LGD, discounting, forward-looking scenarios, SICR governance and validation.

## Suggested future enhancements

- Obtain a loan-level dataset with origination date and performance windows.
- Build out-of-time validation and vintage analysis.
- Calibrate PD by segment and assess calibration drift.
- Add cost-sensitive threshold optimization.
- Add SHAP explanations and model documentation.
- Add lifetime PD term structures and scenario-weighted discounted cash shortfalls for a properly governed IFRS 9 implementation.

## License

This project is licensed under the MIT License.
