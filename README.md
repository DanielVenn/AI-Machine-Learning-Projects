# Apziva Program Projects
---
This repository contains the project work for my AI Residency Program. Throughout the program, I have worked on various projects to explore and develop my skills in artificial intelligence and machine learning.

# Projects
## **Project 1:** *Happy Customers*
---
- Description: Presented with a subset of customer feedback data, predict if a customer is happy or not based on the answers they give to questions asked.
- Technologies used: 
    - Jupyter notebook 
    - Pandas
    - NumPy
    - Matplotlib
    - SciPy
    - scikit-learn
    - Hyperopt
- Results: 
    - This project highlighted a success metric of 73% accuracy. This was achieved using both **Logistic Regression** ('*Applied Regularization Parameter*' [13]) and **Random Forests** ('*Minimum Samples Split*' [26]).
    - It was found that the **X2** feature had a negative feature importance ('*Feature Selection*' [15]), and thus **X2** could be safely removed in the next survey.
    - Generally, the default parameters were sufficient for achieving the requested accuracy score, and further attempts at Hyperparameter Tuning via Grid Search and Bayesian Optimization proved ineffective.

## **Project 2:** *Term Deposit Marketing*
---
- Description: Presented with customer data, predict if a customer will subscribe to a term deposit. Further, find the customers most likely to buy the investment product, and which segment should be prioritized by sales.
- Concepts used: 
    - Class Imbalance 
    - Minority Upscaling
    - Class Weights
    - Model Analysis
    - Feature Importance
    - Hyperparameter Tuning
- Results: 
We have applied the following preprocessing steps:

- Upsampling. This is to account for the class imbalance.
- Class Weights. This is also to account for the class imbalance.

Both steps have been shown to improve the tracked metrics. Additional, Probability Prediction was used to enhance the effectiveness in the models. Comparing three models, the average performance score was calculated and shown below:

| Model               | Average Performance Score |
|---------------------|---------------------------|
| Logistic Regression | 0.87                      |
| Decision Tree       | 0.95                      |
| Random Forest       | 0.97                      |

The metrics for accuracy, precision, recall, and F1 score can be found below, including the XGBoost feature:

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.92     | 0.46      | 0.52   | 0.49     |
| Decision Tree       | 0.92     | 0.41      | 0.41   | 0.41     |
| Random Forest       | 0.92     | 0.47      | 0.77   | 0.58     |
| XGBoost             | 0.93     | 0.51      | 0.62   | 0.56     |

There were 3 Goals given for this problem:

1. **Accuracy of 81%.** We have achieved an Accuracy Score of 92%.
2. **Evaluate with 5-fold cross validation.** Evaluated above, the outputs are printed for each model.
3. **Report the Average Performance Score.** Shown below as 87%.

And 2 Bonus Goals:

1. **Determine the segment(s) of customers our client should prioritize.**
2. **What makes the customers buy?** For this, we used Feature Importance to determine the most important feature in whether or not a customer buys. As found in the 'Feature Importance' section, we find that the 'day' value is the most important feature, and the highest positive result rate occurs on the first and last days of the month. Therefore, it is important to focus on those days of the month for selling to customers.

## **Project 3:** *Potential Talents*
---
- Description: Predict how fit the candidate is based on their available information.

- In Progress
