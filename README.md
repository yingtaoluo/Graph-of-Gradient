# Graph-of-Gradient
Please see newly added experiments "New Experiment Table 4.png", which is a hyperparameter sensitivity study on "K" of the KNN algorithm. It shows that "K" generally does not affect the model's performance.

![image](https://github.com/yingtaoluo/Graph-of-Gradient/blob/main/New%20Experiment%20Table%204.png) 

Please see newly added visualization "New Visualization Figure 3.png". It is using t-SNE to visualize the distribution of input features and last-layer gradients upon different race groups for MIMIC-III dataset (Purple: White, Yellow: 'BLACK', Green: 'HISPANIC', Others are discarded for the convenience of visualization and presentation). Figure 3 shows that while it is impossible to separate people by race based on input features, we find that the least populated subpopulation group (colored by green) is distributed closely on the graph of last-layer gradient. This case study demonstrates that gradient is indeed a better representation of the true demographics, when the true demographics is unknown.

![image](https://github.com/yingtaoluo/Graph-of-Gradient/blob/main/New%20Visualization%20Figure%203.png) 

"Normal" folder contains Baselines (without fairness algorithms). 

"models" folder together with "train.py" is an example implementation of GoG.
