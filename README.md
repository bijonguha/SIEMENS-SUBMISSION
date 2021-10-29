# SIEMENS-SUBMISSION

The gas turbine is the engine at the heart of the power plant that produces electric current. A gas turbine is a combustion engine that can convert natural gas or other liquid fuels to mechanical energy. This energy then drives a generator that produces electrical energy.

**This is a Machine Learning project on Gas Turbine Compressors and Turbine decay evaluation for Propulsion Plants.**

### Goal / Objective:

To build Machine learning models that can predict the GT Compressor decay state coefficient.

### Outcome: 

Achieved an RMSE(root-mean squared error) score of 0.000844 for our Compressor model.

## **File description:**


- **report.html:** The Exploratory Data Analysis report generated with Pandas-Profiling. Result of which was used on narrowing down on model.

- **siem_bijon_guha_submission.ipynb:** The Jupyter notebook where 7 different ML models were evaluated for the GT Compressor decay state coefficient.

- **CatBoostRegressor_GT_Compressor.pickle:** Pickle file of the CatBoost model for Compressor

- **Model_Comparison_GT_Compressor.csv:** CSV file that contains the comparison chart of all the models tested for the Compressor decay state coefficient

## Resources:

**Python version :** 3.7
**Packages Used:** pandas, numpy, matplotlib, seaborn, sklearn, xgboost, lightgbm, catboost

## Machine Learning models tested:

Seven different Machine Learning models were used for this project.

- Linear regression

- Random Forest

- K Nearest neighbours

- Support vector machine

- XGBoost

- Light GBM(LGB)

- CatBoost

## <u>GT Compressor Decay State Evaluation</u>

### Model Comparison chart:

![](images/model_comparison_compressor.png)

## Conclusion:

CatBoost model performs exceptionally well. It is fast, efficient and simple to implement as well.
