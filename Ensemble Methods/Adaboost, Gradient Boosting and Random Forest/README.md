# Theory Questions

## 1. An analysis of the mistakes of these models

### Random Forests
Overfitting and Sensitivity to Hyperparameters:

* Issue: Random Forests are susceptible to overfitting, especially when the number of trees in the ensemble is large. Hyperparameters, such as the depth of individual trees and the number of features considered at each split, must be carefully tuned to prevent overfitting.
- Mitigation: Practitioners should perform thorough hyperparameter tuning and consider techniques like cross-validation to find the optimal settings. Regularization methods, such as limiting tree depth, can also be employed to control overfitting.

Lack of Interpretability:

- Issue: Random Forest models, being ensembles of decision trees, are inherently complex and may lack interpretability. Understanding the underlying reasoning behind predictions can be challenging, making it difficult to extract actionable insights from the model.
- Mitigation: While the ensemble nature of Random Forests contributes to robust predictions, interpreting individual trees or using techniques like feature importance analysis can provide some insights. For tasks where interpretability is crucial, simpler models or model-agnostic interpretability methods might be preferred over Random Forests.

### AdaBoost + Decision Trees
Sensitivity to Noisy Data and Outliers:

- Issue: Adaboost + Decision Trees is sensitive to noisy data and outliers, as misclassified instances receive higher weights during training. Outliers can have a significant impact on the model's focus, potentially leading to suboptimal performance.
- Mitigation: Careful preprocessing, such as outlier detection and removal, is essential to handle noisy data. Additionally, using robust decision trees or incorporating techniques to reduce the influence of outliers can be beneficial.

Vulnerability to Overfitting and Hyperparameter Tuning Complexity:

- Issue: There is a risk of overfitting, particularly if the base decision trees are too complex or if the boosting process iterates for too many rounds. Tuning hyperparameters, such as the learning rate and the number of iterations, can be complex and requires careful consideration.
- Mitigation: Hyperparameter tuning through techniques like cross-validation is crucial to prevent overfitting. Experimenting with different combinations of hyperparameters helps find the right balance between model complexity and generalization performance. Regularization techniques and limiting the depth of decision trees are strategies to control overfitting.

### Gradient Boosted Decision Trees (GBDT)
Sensitivity to Noisy Data and Outliers:

- Issue: GBDT can be sensitive to noisy data and outliers during the training process, as it may assign high weights to misclassified instances, leading to suboptimal performance.
- Mitigation: Preprocessing steps such as outlier detection and removal, or the use of robust loss functions, are important to handle noisy data and outliers effectively.

Risk of Overfitting and Hyperparameter Tuning:

- Issue: GBDT has the potential to overfit the training data, particularly if the model is too complex or the number of trees is too high. Tuning hyperparameters, such as the learning rate, tree depth, and the number of trees, is crucial.
- Mitigation: Implement regularization techniques like limiting the tree depth, adjusting the learning rate, and performing careful hyperparameter tuning through techniques like cross-validation to prevent overfitting and achieve optimal model performance.


## 2. Feature Similarity between the 3 ensemble techniques

- All three ensemble methods are sensitive to noisy data and outliers. Noisy or outlier-laden instances can disrupt the training process by influencing the decisions of individual trees in the ensemble.

- Random Forests, Adaboost + Decision Trees, and GBDT all face the risk of overfitting, particularly when the individual base learners (trees) are too complex or when the ensemble is allowed to become excessively large

- Random Forests, Adaboost + Decision Trees, and GBDT can exhibit computational complexity, especially when dealing with a large number of trees or deep trees.

- The ensemble nature of these methods, combining multiple trees, can make model interpretation challenging.



### Code Structure
- All the tasks (Hyperparameter tuning, comparision with other ensemble techniques, etc ) with respect to Random Forests are in `Random_Forest.ipynb`
- Rest of the files contain the analysis and implementation of the respective algorithms

