import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
from ucimlrepo import fetch_ucirepo

# fetch dataset
student_performance = fetch_ucirepo(id=320)

# data (as pandas dataframes)
X = student_performance.data.features
y = student_performance.data.targets
#would  ot run unluss this is here took alot of trial to get this working
X = pd.get_dummies(X, drop_first=True)

#same for this
y = y["G3"].astype(float).values.ravel()


# metadata
#print(student_performance.metadata)
# variable information
#print(student_performance.variables)

X, y = shuffle(X, y, random_state=7)

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

y_train = y_train.ravel()



# Create and train the Support Vector Regressor using a linear kernel
sv_regressor = SVR(kernel='linear')
sv_regressor.fit(X_train, y_train)

# Run the regressor on the testing data and predict the output (predicted labels). y_test_pred= sv_regressor.predict(X_test)
y_test_pred = sv_regressor.predict(X_test)

# Evaluate the performance of the regressor and print the initial metrics.
mse = mean_squared_error(y_test, y_test_pred)
ev_score = explained_variance_score(y_test, y_test_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Explained Variance Score: {ev_score:.2f}")

# binarize the predicted values & the actual values using threshold of 12.0.
y_pred_label = (y_test_pred >= 12.0).astype(int)
y_test_label = (y_test.ravel() >= 12.0).astype(int)

# Create the confusion matrix using the predicted labels and the actual labels.
confusion_mat = confusion_matrix(y_test_label, y_pred_label)
print("Confusion Matrix:")
print(confusion_mat)

# Visualize the confusion matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion Matrix')
plt.colorbar()
ticks = np.arange(2)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()


# Print the classification report based on the confusion matrix.
print("Classification Report:")
print(classification_report(y_test_label, y_pred_label))
