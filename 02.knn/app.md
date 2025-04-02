K-Nearest Neighbors

What is KNN?
-> is a machine learning algorithm which makes predictions based on the proximity of data points
-> it assumes that similar data points are located close to each other
-> for a new unlabeled data point, knn looks at the 'K' closest data points (the neighbours) in the training dataset
-> it then uses the labels of these neighbors to determine the prediction for the new data point
-> accuracy refers to how the % of times knn algorithm makes the right prediction
-> use x_test to get the accuracy of the model
-> precision focuses on the accuracy of positive predictions.
-> precision measures the correctly predicted positive cases out of all the cases predicted as positive.
-> precision = True Positives/ (True Positives + False Positives) where true positives are number of correctly identified as positive
and false positive numbers of cases falsely identified as positive
