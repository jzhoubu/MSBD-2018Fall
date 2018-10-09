# 5001 Midterm
Short-answer Questions and Computation Questions(Calculator is required).

## Top Question Prediction
- Given formula and data, compute Least Squre Fit / kNN process / Gradient Descent / Entropy to construct decision tree
- Given situation, compare different methods among machine learning, or ensemble learning


## Lecture1
- Describing and Summarizing Data
> Basic concept of mean, min, max, median, mode, range, IQR, var, standard deviation.(P12/59)
- Identifying Relationship between Variables
    - Eyeball Fit
    - Least Squre Fit
    - Correlation Coefficient
- Forecasting Outcomes
    + Concept of experiment, sample space, event
    + Probability
    + Independence
    + Conditional Probability
    + Bayes Theorem
    + Mean, Expected Value, or Expectation
    + Variance
    + Binomial Probability Model (Mean and Standard Deviation)
    > Noted: Example in P49/59 is a Multinomial distribution but not Binomial distribution.
    + Probability Models for Continuous Data

## Lecture 2
- Introduction to Supervised Learning
    + Machine Learning:   3   Steps
        1.  Collect data,   extract features
        2.  Determine a model
        3.  Train the model with the data
    + Derivative/Gradient Descent Method
    >Noted: Example in P16,  the learning rate `r` should be 0.5 but not 0.1.
- A few Supervised Learning Methods
    - Linear Regression/Classification
        + Logistic Regression
        + Support Vector Machine (SVM)
    - Nearest Neighbor Methods
        + K Nearest Neighbor Methods (kNN)
        + LR vs kNN(P40)
    - Decision Tree
        + Information Gain: Entropy(P50)

## Lecture 3
- Generalization  of  models
    - Bias-Variance-Model Complexity (P24)
- Selecting good models
    - Cross Validation
- Improving models
    + Regularization
    + Ensemble learning
        + Averaging
        + Bagging
        > Generates m new training sets by subsampling from samples uniformly with replacement, then train m models and conbined by averaging the output.
        + Random forest
        > Generateing different trees by using subset of samples to construct the tree, and using subset of features each time split the tree.  
        + Boosting
        > Train all data with weight, then re-weight based on the classification outcome. The final prediction is based on the whole sequential of models(voting).
        - Ensemble Learning Summary(P60)

## Lecture 4
- Clustering
    - K-means clustering
    - Hierarchical clustering
- Dimensionality Reduction
    - Principle Component Analysis
- Ranking
    - Google’s PageRank
