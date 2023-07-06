# Apziva Program Projects

This repository contains the project work for my AI Residency Program. Throughout the program, I have worked on various projects to explore and develop my skills in artificial intelligence and machine learning.

## Projects
**Project 1:** *Happy Customers*

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

**Project 2:** *Term Deposit Marketing*

- Description: Presented with customer data, predict if a customer will subscribe to a term deposit. Further, find the customers most likely to buy the investment product, and which segment should be prioritized by sales.
- Concepts used: 
    - Class Imbalance 
    - Minority Upscaling
    - Class Weights
    - Model Analysis
    - Feature Importance
    - Hyperparameter Tuning
- Results: 
    - This project highlighted a success metric of 81% accuracy. An accuracy score of 87% was achieved - and by identifying a Class Imbalance, and adressing it with both Upsampling and Class Weights, the score was made to be more reliable for the minority case.
    - Feature Importance analysis was used to determine the most important feature that should be prioritized by Sales.
