1. Instructions to reproduce model:

 a) Create anaconda environment: conda create --name test_environment python=3.7.5
 b) Within Anaconda prompt install all required packages from requirements.txt using command:
    pip install -r requirements.txt

2. Code structure:
 Main class XGBEstimator is at src.models

3. Short questions:
 a) Imagine the credit risk use case above. You have two models: a logistic regression
    model with an F1-score of 0.60 and a neural network with an F1-score of 0.63. Which
    model would you recommend for the bank and why?

    In the credit use case above we have imbalanced dataset, so F1-score (or alternatively roc-auc) are most
    suitable metrics, because they are combined precision and recall metrics. Surely we want to credit those
    who are very likely to return credit (high precision), but not to miss a chance to make
    money on giving loans to those who are not so likely to return credit, but are likely enough (high recall).
    So I would suggest using neural network, because of the higher precision-recall tradeoff.

 b) A customer wants to know which features matter for their dataset. They have several
    models created in the DataRobot platform such as random forest, linear regression, and
    neural network regressor. They also are pretty good at coding in Python so wouldn't
    mind using another library or function. How do you suggest this customer gets feature
    importance for their data?

    If customer prefers random forrest or any other tree-based ensemble method or single tree-based method
    I would recommend scikit-learn built in feature_importances_ method which can be called on estimator instance (Also available in XGBoost).
    Feature importance is calculated as the decrease in node impurity weighted by the probability of reaching that node,
    which is highly informative, especially if we are dealing with ensembles.

    If customer preffers differentiable models, such as linear regression or neural network, I would recommend
    using permutation feature importance method which is also implemented in scikit-learn. Also don't hesitate
    using simple correlation of individual variables with targets or coefficients of lasso regression.

    Also I would recommend vizualizing individual features to target relations. (I.e percentage of positive examples
    per feature category) using matplotlib and seaborn libraries. Pandas also supports some vizualising abilities.

 4. Topics for further research and testing :
    a) Most effective generic feature selection.
    b) More robust generic missing values imputation (using KNN i.e.)
    c) Using different categorical encoders, others than one-hot (Sum Encoder, Helmert Encoder,
    Frequency Encoder and so on)
    d) Making Bayesian Hyperparameter Search more reproducible.
    ...
