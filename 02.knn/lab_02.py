from sklearn import neighbors, metrics



x = [[5,6],[3,5],[8,4],[1,8],[3,4],[7,6],[9,3],[9,8],[3,8],[8,9],[9,2],[5,7]]
y = ['GOOD','BAD','GOOD','BAD','GOOD','BAD','GOOD','BAD','GOOD','BAD','GOOD','BAD']

x_train = x[:6]
x_test = x[6:]
y_train = y[:6]
y_test = y[6:]



knn_one = neighbors.KNeighborsClassifier(n_neighbors=1)
knn_three = neighbors.KNeighborsClassifier(n_neighbors=3)
knn_five = neighbors.KNeighborsClassifier(n_neighbors=5)

knn_one.fit(x_train, y_train)
knn_three.fit(x_train, y_train)
knn_five.fit(x_train, y_train)

y_predict_one = knn_one.predict(x_test)
y_predict_three = knn_three.predict(x_test)
y_predict_five = knn_five.predict(x_test)

a_y_one = metrics.accuracy_score(y_predict_one, y_test)
a_y_three = metrics.accuracy_score(y_predict_three, y_test)
a_y_five = metrics.accuracy_score(y_predict_five, y_test)
confusion_matrix_one = metrics.confusion_matrix(y_predict_one,y_test)
confusion_matrix_three = metrics.confusion_matrix(y_predict_three,y_test)
confusion_matrix_five = metrics.confusion_matrix(y_predict_five,y_test)


print('\n y_actual\t: ', y_test)
print('\n Accuracy one\t: ', a_y_one)
print('\n Accuracy three\t: ', a_y_three)
print('\n Accuracy five\t: ', a_y_five)

print('\n Confusion matrix one', confusion_matrix_one)
print('\n Confusion matrix three', confusion_matrix_three)
print('\n Confusion matrix five', confusion_matrix_five)





