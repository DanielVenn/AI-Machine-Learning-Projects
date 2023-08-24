# Apziva Program Projects
---
This repository contains the project work for my AI Residency Program. Throughout the program, I have worked on various projects to explore and develop my skills in artificial intelligence and machine learning.

# Projects
## **Project 4:** *Computer Vision*
---
**Data Description:**

We collected page flipping video from smart phones and labelled them as flipping and not flipping.

We clipped the videos as short videos and labelled them as flipping or not flipping. The extracted frames are then saved to disk in a sequential order with the following naming structure: VideoID_FrameNumber

**Goal(s):**

Predict if the page is being flipped using a single image.

**Techniques Used**
- Convolutional Neural Networks (CNN): Initially, we utilized CNNs, a deep learning architecture well-suited for image classification tasks. CNNs have been foundational in the rise of deep learning, thanks to their proficiency in capturing spatial hierarchies in images.

- Vision Transformers (ViT): In addition to traditional CNNs, we explored the capabilities of Vision Transformers, especially the ViT model.

- Linformer: A variant of the transformer architecture, it efficiently handles long sequences without much computational overhead, making it ideal for our task.

- PyTorch: We used PyTorch, a popular deep learning library, for constructing, training, and evaluating our model.

- GPU Acceleration: Given the computationally intensive nature of Vision Transformers, we utilized the capabilities of an Nvidia GPU for faster model training.

**Results:**
For context, our deep learning models were designed to classify specific image characteristics, namely, whether an image is 'flipped' or 'not flipped'. A representative example of such classifications can be observed below:

![Flip or Not Flip](4%20MonReader/images/flipnotflip.png)

**Convolutional Neural Networks (CNN):**

The traditional Convolutional Neural Network (CNN) approach demonstrated impressive effectiveness in distinguishing between the two image types. After multiple training epochs, the model settled with a commendable balance between precision, recall, and overall accuracy. The model's performance metrics can be further visualized below:

![CNN Model Metrics](4%20MonReader/images/modelMetricsCNN.png)

The CNN's efficiency stems from its ability to extract hierarchical features from images and has proven to be a robust method for our binary classification task.

**PyTorch Vision Transformer:**

On the other end of the spectrum, we also explored the capabilities of the Vision Transformer (ViT) using PyTorch. While the ViT showcased decent performance, it was notably more computationally intensive. Despite employing GPU acceleration, the training times were significantly longer than its CNN counterpart.

![PyTorch Model Metrics](4%20MonReader/images/modelMetricsPyTorch.png)

Vision Transformers have recently been the focal point in the world of computer vision due to their potential in understanding global image features. However, in our specific use-case, its advantages didn't necessarily translate to a marked increase in performance over CNNs, especially considering the computational costs.

**Summary**

In summary, while both the CNN and Vision Transformer models achieved satisfactory results, the CNN proved to be the more efficient choice for this particular binary classification problem. It demonstrated not only great accuracy but also computational efficiency. The Vision Transformer, although promising, might be better suited for more complex and large-scale image tasks where its unique strengths in capturing global patterns can shine. For this specific dataset and classification goal, the simplicity and efficiency of the CNN was clear.

## **Project 3:** *Potential Talents*
---
- Description: Given candidates and their job titles, use Natural Language Processing techniques to rank candidates based on their job titles. Then, given manual rankings, train and evaluate a model that predicts how fit a given candidate is for a role.
- Concepts used: 
    - **Natural Language Processing (NLP):** This project heavily involves NLP techniques, specifically for generating embeddings from textual data in resumes. Techniques used include Doc2Vec, BERT, GloVe, and ELMo.
    - **Learning to Rank (LTR):** LTR is a branch of machine learning that focuses on training models for ranking items in a list or a group. Three specific LTR models, RankNet, LambdaRank, and LambdaMART are trained and compared.
    - **Model Evaluation:** The models are evaluated using the Normalized Discounted Cumulative Gain (NDCG) metric, a popular choice for measuring the effectiveness of ranking models.
    - **Data Visualization:** Data visualization is used to aid in understanding the model performances, particularly through plotting the NDCG scores of the models over different iterations.
- Results: 

Leveraging multiple text embedding techniques, cosine similarity measures were generated as fitness scores to establish a preliminary ranking of job candidates. Subsequently, a manual 'starring' process was undertaken to mimic the real-world evaluation of these candidates. This created a binary outcome, used as the target for training a series of Learning-to-Rank (LTR) models.

These LTR models, specifically RankNet, LambdaRank, and LambdaMART, were then deployed to re-rank the candidates based on their predicted 'starred' status. Each model was evaluated and compared using the Normalized Discounted Cumulative Gain (NDCG) metric to ascertain the most effective approach.

Furthermore, this project addressed practical considerations regarding the determination of threshold values for deeming candidates unfit, in both specific and general terms. It also discussed potential automation techniques that could be employed in future iterations to enhance efficiency while maintaining measures to prevent human bias. This project thereby offers an insightful perspective into the utility of machine learning and natural language processing in the context of an automated hiring process.

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
