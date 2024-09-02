import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations_with_replacement
from itertools import product

#DATASET LOADING
file_path = "//Users//paolaloi//PycharmProjects//MachineLearningPr//your_dataset.csv"
data = pd.read_csv(file_path)

pd.set_option('display.max_columns', None)
print(data.head())
print(data.describe())

## DATA DISTRIBUTION
data.hist(figsize=(15, 10), bins=20)
plt.suptitle("Distribution of Variables")
plt.show()

##BOXPLOT
plt.figure(figsize=(15, 10))
sns.boxplot(data=data)
plt.title("Boxplot of Variables")
plt.show()

#remove outliers
def remove_outliers_iqr(data, columns):
    for column in columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

columns_to_check = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
clean_data = remove_outliers_iqr(data, columns_to_check)

#CORRELATION MATRIX
corr_matrix = clean_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features (Cleaned Data)')
plt.show()

#NO HIGH, NUOVA MATRIX
data_tot = clean_data.drop(columns=['x6', 'x10'])
corr_matrix = data_tot.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Features Cleaned')
plt.show()

# Division of the dataset into features and lables
X = data_tot.drop(columns=['y']).values
y = data_tot['y'].values

##Division of the dataset into training and testing
def train_test_split(X, y, test_size=0.2, seed=None):
    n_samples = X.shape[0]
    test_size = int(n_samples * test_size)

    if seed is not None:
        np.random.seed(seed)

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    X_train = X[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    y_train = y[indices[:-test_size]]
    y_test = y[indices[-test_size:]]

    return X_train, X_test, y_train, y_test


# initialization
seed = 41
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Calculate the mean and standard deviation only on the training set
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Apply standardization to the training set
X_train = (X_train - mean) / std

# Apply the same standardization to the test set, using the parameters calculated on the training set
X_test = (X_test - mean) / std


# DEF PERCEPTRON
class Perceptron:
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.weights = None
        self.converged = False

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for epoch in range(self.max_iter):
            errors = 0
            for i in range(n_samples):
                if y[i] * np.dot(X[i], self.weights) <= 0:
                    self.weights += y[i] * X[i]
                    errors += 1

            if errors == 0:
                self.converged = True
                print(f"Convergence reached after {epoch + 1} epochs.")
                break

        if not self.converged:
            print("No convergence achieved after maximum number of iterations.")

    def predict(self, X):
        return np.sign(np.dot(X, self.weights))

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


## DEF CROSS VALIDATION
def cross_validation_score(model, X, y, cv=5):
    fold_size = len(X) // cv
    scores = []

    for i in range(cv):
        start, end = i * fold_size, (i + 1) * fold_size
        X_test_cv = X[start:end]
        y_test_cv = y[start:end]

        X_train_cv = np.concatenate((X[:start], X[end:]), axis=0)
        y_train_cv = np.concatenate((y[:start], y[end:]), axis=0)

        model.fit(X_train_cv, y_train_cv)
        score = model.score(X_test_cv, y_test_cv)
        scores.append(score)
        print(f"Fold {i + 1}, Accuracy: {score:.4f}")

    return np.mean(scores)


# DEF SVM with PEGASOS
class PegasosSVM:
    def __init__(self, lambda_param=0.01, T=1000):
        self.lambda_param = lambda_param
        self.T = T
        self.weights = None

    def _hinge_loss_gradient(self, x_i, y_i, w):
        """Calculate the gradient of the hinge loss function."""
        condition = y_i * np.dot(w, x_i)
        if condition < 1:
            return -y_i * x_i
        else:
            return np.zeros_like(w)

    def fit(self, X, y):
        """Trains the SVM model using the Pegasos algorithm."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for t in range(1, self.T + 1):
            eta_t = 1 / (self.lambda_param * t)

            # Random draw of a sample
            i = np.random.randint(0, n_samples)
            x_i = X[i]
            y_i = y[i]

            # Weight update
            gradient = self._hinge_loss_gradient(x_i, y_i, self.weights)
            self.weights = (1 - eta_t * self.lambda_param) * self.weights - eta_t * gradient

            # projections
            norm_w = np.linalg.norm(self.weights)
            if norm_w > (1 / np.sqrt(self.lambda_param)):
                self.weights = self.weights / (norm_w * np.sqrt(self.lambda_param))

    def predict(self, X):
        return np.sign(np.dot(X, self.weights))

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


## DEF LOGISTIC CLASSIFICATION
class LogisticClassification:
    def __init__(self, lambda_param=0.01, T=1000):
        self.lambda_param = lambda_param
        self.T = T
        self.weights = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for t in range(1, self.T + 1):
            i = np.random.randint(0, n_samples)
            xi = X[i]
            yi = y[i]

            # Weight update
            eta_t = 1 / (self.lambda_param * t)
            margin = yi * np.dot(xi, self.weights)
            probability = self.sigmoid(margin)

            # Update weights using logistic loss
            self.weights = (1 - eta_t * self.lambda_param) * self.weights + eta_t * (yi * (1 - probability)) * xi

    def predict(self, X):
        linear_output = np.dot(X, self.weights)
        probabilities = self.sigmoid(linear_output)
        return np.where(probabilities >= 0.5, 1, -1)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


# DEF POLINOMIAL EXPANSION
def polynomial_expansion(X, degree=2):
    n_samples, n_features = X.shape
    new_features = list(combinations_with_replacement(range(n_features), degree))
    X_poly = np.empty((n_samples, len(new_features)))

    for i, (f1, f2) in enumerate(new_features):
        X_poly[:, i] = X[:, f1] * X[:, f2]

    return X_poly

## Apply polynomial expansion on the training and test set and combine them with the original
X_train_poly = polynomial_expansion(X_train, degree=2)
X_test_poly = polynomial_expansion(X_test, degree=2)
X_train_expanded = np.hstack((X_train, X_train_poly))
X_test_expanded = np.hstack((X_test, X_test_poly))

#DEF KERNEL
def _gaussian_kernel_matrix(X, Y, gamma):
    sq_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T)
    return np.exp(-sq_dists / (2 * gamma))

def _polynomial_kernel_matrix(X, Y, degree, coef0):
    return (np.dot(X, Y.T) + coef0) ** degree

# DEF PERCEPTRON with GAUSSIAN KERNEL & POLYNOMIAL KERNEL
class KernelPerceptron:
    def __init__(self, kernel='gaussian', gamma=0.5, degree=3, coef0=1, T=1000):
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.T = T
        self.S = []

    def _kernel(self, x1, x2):  # Select the kernel to use
        if self.kernel_type == 'gaussian':
            return self._gaussian_kernel(x1, x2)
        elif self.kernel_type == 'polynomial':
            return self._polynomial_kernel(x1, x2)
        else:
            raise ValueError("Kernel type not supported. Choose 'gaussian' or 'polynomial'.")

    def fit(self, X, y):
        self.support_vectors = []
        self.support_vector_labels = []
        self.S = []

        n_samples = X.shape[0]

        for _ in range(self.T):
            for i in range(n_samples):
                prediction = 0
                for j in range(len(self.S)):
                    prediction += self.support_vector_labels[j] * self._kernel(self.support_vectors[j], X[i])
                prediction = np.sign(prediction)

                # Update the model if the prediction is wrong
                if prediction != y[i]:
                    self.S.append(i)
                    self.support_vectors.append(X[i])
                    self.support_vector_labels.append(y[i])

    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = 0
            for j in range(len(self.S)):
                prediction += self.support_vector_labels[j] * self._kernel(self.support_vectors[j], x)
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


## SVM KERNALIZED with GAUSSIAN KERNEL & POLYNOMIAL KERNEL
class KernelPegasosSVM:
    def __init__(self, lambda_param=0.01, T=1000, kernel='gaussian', gamma=0.5, degree=3, coef0=1):
        self.lambda_param = lambda_param
        self.T = T
        self.kernel_type = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = None
        self.support_vectors = None
        self.support_vector_labels = None

    def _kernel_matrix(self, X, Y):
        """Calculate the kernel matrix between X and Y"""
        if self.kernel_type == 'gaussian':
            return self._gaussian_kernel_matrix(X, Y, self.gamma)
        elif self.kernel_type == 'polynomial':
            return self._polynomial_kernel_matrix(X, Y, self.degree, self.coef0)
        else:
            raise ValueError("Kernel type not supported. Choose 'gaussian' or 'polynomial'.")

    def fit(self, X, y):
        n_samples, _ = X.shape
        # Reset the alpha, support_vectors, and support_vector_labels at the start of each fit call
        self.alpha = np.zeros(n_samples)
        self.support_vectors = X
        self.support_vector_labels = y

        for t in range(1, self.T + 1):
            i_t = np.random.randint(0, n_samples)
            x_i = X[i_t]
            y_i = y[i_t]

            K = self._kernel_matrix(self.support_vectors, np.array([x_i]))

            margin = y_i * (1 / (self.lambda_param * t)) * np.sum(
                self.alpha * self.support_vector_labels * K.flatten()
            )

            # Update of the alpha parameter
            if margin < 1:
                self.alpha[i_t] += 1

    def predict(self, X):
        if len(self.support_vectors) == 0:
            return np.zeros(X.shape[0])

        K = self._kernel_matrix(self.support_vectors, X)
        predictions = np.sign(np.dot(self.alpha * self.support_vector_labels, K))
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def _gaussian_kernel_matrix(self, X, Y, gamma):
        sq_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(Y ** 2, axis=1) - 2 * np.dot(X, Y.T)
        return np.exp(-sq_dists / (2 * gamma))

    def _polynomial_kernel_matrix(self, X, Y, degree, coef0):
        return (np.dot(X, Y.T) + coef0) ** degree




## CODES INITIALIZATION
#PERCEPTRON
param_grid = {'max_iter': [1000, 10000]}

results = []

# Iterate over all combinations of parameters
for max_iter in param_grid['max_iter']:
    perceptron = Perceptron(max_iter=max_iter)
    mean_accuracy = cross_validation_score(perceptron, X_train, y_train, cv=5)

    results.append({
        'max_iter': max_iter,
        'cv_mean_accuracy': mean_accuracy
    })

# Select the best result
best_result = max(results, key=lambda x: x['cv_mean_accuracy'])

# Print the cross-validation results
for result in results:
    print(f"Max Iter: {result['max_iter']}")
    print(f"Mean Cross-Validation Accuracy: {result['cv_mean_accuracy']:.4f}")
    print("-" * 50)

# Print the best result
print(f"Best Max Iter: {best_result['max_iter']}")
print(f"Best Mean Cross-Validation Accuracy: {best_result['cv_mean_accuracy']:.4f}")

# Initialize the Perceptron with the best parameters
best_perceptron = Perceptron(max_iter=best_result['max_iter'])
best_perceptron.fit(X_train, y_train)

# Calculate the misclassification rate
test_accuracy = best_perceptron.score(X_test, y_test)
misclassification_rate = 1 - test_accuracy
print(f"Misclassification Rate on the test set: {misclassification_rate:.4f}")
print("Final weights of the model:", best_perceptron.weights)


#SVM PEGASOS
if __name__ == "__main__":
    param_grid_svm = {
        'lambda_param': [0.001, 0.1, 1.0],
        'T': [1000, 10000, 50000 ],
    }

    results_svm = []

    # Iterate over all combinations of parameters
    for lambda_param, T in product(param_grid_svm['lambda_param'], param_grid_svm['T']):
        svm = PegasosSVM(lambda_param=lambda_param, T=T)
        svm.fit(X_train, y_train)

        # Evaluate with Cross-Validation
        mean_accuracy = cross_validation_score(svm, X_train, y_train, cv=5)

        # Save the results
        results_svm.append({
            'lambda_param': lambda_param,
            'T': T,
            'cv_mean_accuracy': mean_accuracy
        })

    # Select the best result based on cross-validation accuracy
    best_result_svm = max(results_svm, key=lambda x: x['cv_mean_accuracy'])

    # Print the cross-validation results
    for result in results_svm:
        print(f"Lambda: {result['lambda_param']}, T: {result['T']}")
        print(f"Mean Cross-Validation Accuracy: {result['cv_mean_accuracy']:.4f}")
        print("-" * 50)

    # Print the best result
    print(f"Best Lambda: {best_result_svm['lambda_param']}, Best T: {best_result_svm['T']}")
    print(f"Best Mean Cross-Validation Accuracy: {best_result_svm['cv_mean_accuracy']:.4f}")

    # Initialize the Pegasos SVM with the best parameters
    best_svm = PegasosSVM(lambda_param=best_result_svm['lambda_param'], T=best_result_svm['T'])
    best_svm.fit(X_train, y_train)

    # Calculate the misclassification rate on the test set
    final_test_accuracy = best_svm.score(X_test, y_test)
    final_misclassification_rate = 1 - final_test_accuracy
    print(f"Final Misclassification Rate on the test set: {final_misclassification_rate:.4f}")
    print("Final weights of the Pegasos SVM model:", best_svm.weights)



#LINEAR CLASSIFICATION w/ LOG LOSS
if __name__ == "__main__":
    param_grid_logistic = {
        'lambda_param': [0.001, 1.0, 3.0],
        'T': [10000, 50000]
    }

    results_logistic = []

    # Iterate over all combinations of parameters
    for lambda_param, T in product(param_grid_logistic['lambda_param'], param_grid_logistic['T']):
        logistic_classification = LogisticClassification(lambda_param=lambda_param, T=T)

        # Evaluate with Cross-Validation
        mean_accuracy = cross_validation_score(logistic_classification, X_train, y_train, cv=5)

        # Save the results
        results_logistic.append({
            'lambda_param': lambda_param,
            'T': T,
            'cv_mean_accuracy': mean_accuracy
        })

    # Select the best result based on cross-validation accuracy
    best_result = max(results_logistic, key=lambda x: x['cv_mean_accuracy'])

    # Print the cross-validation results for all combinations
    for result in results_logistic:
        print(f"Lambda: {result['lambda_param']}, T: {result['T']}")
        print(f"Mean Cross-Validation Accuracy: {result['cv_mean_accuracy']:.4f}")
        print("-" * 50)

    # Print the best result
    print(f"Best Lambda: {best_result['lambda_param']}, Best T: {best_result['T']}")
    print(f"Best Mean Cross-Validation Accuracy: {best_result['cv_mean_accuracy']:.4f}")

    best_logistic_classification = LogisticClassification(lambda_param=best_result['lambda_param'], T=best_result['T'])
    best_logistic_classification.fit(X_train, y_train)

    # Evaluate on the test set
    test_accuracy = best_logistic_classification.score(X_test, y_test)
    test_misclassification_rate = 1 - test_accuracy
    print(f"Test Misclassification Rate on the test set: {test_misclassification_rate:.4f}")
    print("Final weights of the Pegasos Logistic CLassification model:")
    print(best_logistic_classification.weights)



#PERCEPTRON EXPANDED
if __name__ == "__main__":
    param_grid_perceptron = {
        'max_iter': [1000, 10000],
    }

    results_perceptron = []

    # Iterate over all combinations of parameters
    for max_iter in param_grid_perceptron['max_iter']:
        perceptron_expanded = Perceptron(max_iter=max_iter)

        # Evaluate with Cross-Validation
        mean_accuracy_expanded = cross_validation_score(perceptron_expanded, X_train_expanded, y_train, cv=5)

        results_perceptron.append({
            'max_iter': max_iter,
            'cv_mean_accuracy': mean_accuracy_expanded
        })

    # Select the best result based on cross-validation accuracy
    best_result = max(results_perceptron, key=lambda x: x['cv_mean_accuracy'])

    # Print the cross-validation results for all combinations
    for result in results_perceptron:
        print(f"Max Iterations: {result['max_iter']}")
        print(f"Mean Cross-Validation Accuracy: {result['cv_mean_accuracy']:.4f}")
        print("-" * 50)

    print(f"Best Max Iterations: {best_result['max_iter']}")
    print(f"Best Mean Cross-Validation Accuracy: {best_result['cv_mean_accuracy']:.4f}")

    # Initialize Perceptron with the best parameters
    best_perceptron_expanded = Perceptron(max_iter=best_result['max_iter'])
    best_perceptron_expanded.fit(X_train_expanded, y_train)

    # Evaluate on the test set
    test_accuracy_expanded = best_perceptron_expanded.score(X_test_expanded, y_test)
    test_misclassification_rate_expanded = 1 - test_accuracy_expanded
    print(f"Test Misclassification Rate on the test set with polynomial features: {test_misclassification_rate_expanded:.4f}")
    print("Final weights of the model with polynomial features:")
    print(best_perceptron_expanded.weights[:X_train_expanded.shape[1]])

#SVM EXPANDED
if __name__ == "__main__":
    param_grid = {
        'lambda_param': [0.001, 0.1, 2.0],
        'max_iter': [10000, 50000]
    }

    results = []

    # Iterate over all combinations of parameters
    for lambda_param in param_grid['lambda_param']:
        for max_iter in param_grid['max_iter']:
            svm_poly = PegasosSVM(lambda_param=lambda_param, T=max_iter)

            # Evaluate with Cross-Validation
            mean_accuracy_poly = cross_validation_score(svm_poly, X_train_expanded, y_train, cv=5)

            # Save the results
            results.append({
                'lambda_param': lambda_param,
                'max_iter': max_iter,
                'cv_mean_accuracy': mean_accuracy_poly
            })

    # Select the best result based on cross-validation accuracy
    best_result = max(results, key=lambda x: x['cv_mean_accuracy'])

    # Print the cross-validation results for all combinations
    for result in results:
        print(f"Lambda: {result['lambda_param']}, Max Iter: {result['max_iter']}")
        print(f"CV Mean Accuracy: {result['cv_mean_accuracy']:.4f}")
        print("-" * 50)

    print(f"Best Lambda: {best_result['lambda_param']}, Best Max Iter: {best_result['max_iter']}")
    print(f"Best CV Mean Accuracy: {best_result['cv_mean_accuracy']:.4f}")

    # Initialize the Pegasos SVM model with the best parameters
    best_svm_poly = PegasosSVM(lambda_param=best_result['lambda_param'], T=best_result['max_iter'])
    best_svm_poly.fit(X_train_expanded, y_train)

    # Evaluate on the test set with polynomial features
    test_accuracy_poly = best_svm_poly.score(X_test_expanded, y_test)
    test_misclassification_rate_poly = 1 - test_accuracy_poly
    print(f"Test Misclassification Rate on the test set with polynomial features: {test_misclassification_rate_poly:.4f}")
    print("Final weights of the Pegasos SVM model with polynomial features:")
    print(best_svm_poly.weights)


# LOG CLASSIFICATION EXPANDED
if __name__ == "__main__":
    # Define the parameter grid
    param_grid = {
        'lambda_param': [0.001, 1.0, 3.0],
        'max_iter': [10000, 50000]
    }

    # List to store the results
    results = []

    # Iterate over all combinations of parameters
    for lambda_param in param_grid['lambda_param']:
        for max_iter in param_grid['max_iter']:
            # Initialize the Pegasos Logistic Regression model with polynomial features
            logistic_classification_poly = LogisticClassification(lambda_param=lambda_param, T=max_iter)

            # Evaluate with Cross-Validation
            mean_accuracy_poly = cross_validation_score(logistic_classification_poly, X_train_expanded, y_train, cv=5)

            # Save the results
            results.append({
                'lambda_param': lambda_param,
                'max_iter': max_iter,
                'cv_mean_accuracy': mean_accuracy_poly
            })

    # Select the best result based on cross-validation accuracy
    best_result = max(results, key=lambda x: x['cv_mean_accuracy'])

    # Print the cross-validation results for all combinations
    for result in results:
        print(f"Lambda: {result['lambda_param']}, Max Iter: {result['max_iter']}")
        print(f"CV Mean Accuracy: {result['cv_mean_accuracy']:.4f}")
        print("-" * 50)

    # Print the best result
    print(f"Best Lambda: {best_result['lambda_param']}, Best Max Iter: {best_result['max_iter']}")
    print(f"Best CV Mean Accuracy: {best_result['cv_mean_accuracy']:.4f}")

    # Initialize the Pegasos Logistic Regression model with the best parameters
    best_logistic_classification_poly = LogisticClassification(lambda_param=best_result['lambda_param'], T=best_result['max_iter'])
    best_logistic_classification_poly.fit(X_train_expanded, y_train)

    # Evaluate on the test set with polynomial features
    test_accuracy_poly = best_logistic_classification_poly.score(X_test_expanded, y_test)
    test_misclassification_rate_poly = 1 - test_accuracy_poly
    print(f"Test Misclassification Rate on the test set with polynomial features: {test_misclassification_rate_poly:.4f}")

    # Print the final weights of the model with polynomial features
    print("Final weights of the Pegasos Logistic Classification model with polynomial features:")
    print(best_logistic_classification_poly.weights)


# PERCEPTRON GAUSSIAN KERNEL
if __name__ == "__main__":
    # Define the parameter grid
    param_grid = {
        'gamma': [0.01, 0.1, 1.0],
        'T': [10]
    }

    # List to store the results
    results = []

    # Iterate over all combinations of parameters
    for gamma in param_grid['gamma']:
        for T in param_grid['T']:
            # Initialize the Kernelized Perceptron with Gaussian Kernel
            perceptron_gaussian = KernelPerceptron(kernel='gaussian', gamma=gamma, T=T)

            # Evaluate with Cross-Validation
            mean_accuracy = cross_validation_score(perceptron_gaussian, X_train, y_train, cv=5)

            # Save the results
            results.append({
                'gamma': gamma,
                'T': T,
                'cv_mean_accuracy': mean_accuracy
            })

    # Select the best result based on cross-validation accuracy
    best_result = max(results, key=lambda x: x['cv_mean_accuracy'])

    # Print the cross-validation results for all combinations
    for result in results:
        print(f"Gamma: {result['gamma']}, T: {result['T']}")
        print(f"Mean Cross-Validation Accuracy: {result['cv_mean_accuracy']:.4f}")
        print("-" * 50)

    # Print the best result
    print(f"Best Gamma: {best_result['gamma']}, Best T: {best_result['T']}")
    print(f"Best CV Mean Accuracy: {best_result['cv_mean_accuracy']:.4f}")

    # Initialize the Kernelized Perceptron with the best parameters
    best_perceptron_gaussian = KernelPerceptron(kernel='gaussian', gamma=best_result['gamma'], T=best_result['T'])
    best_perceptron_gaussian.fit(X_train, y_train)

    # Evaluate on the test set
    test_accuracy = best_perceptron_gaussian.score(X_test, y_test)
    test_misclassification_rate = 1 - test_accuracy
    print(f"Test Misclassification Rate on the test set: {test_misclassification_rate:.4f}")


# PERCEPTRON POLYNOMIAL KERNEL
if __name__ == "__main__":
    # Define the parameter grid for the Polynomial Kernel
    param_grid_polynomial = {
        'degree': [2, 3,4],  # Adjust the degrees according to what you want to explore
        'T': [10]
    }

    # List to store the results
    results_polynomial = []

    # Iterate over all combinations of parameters for the Polynomial Kernel
    for degree in param_grid_polynomial['degree']:
        for T in param_grid_polynomial['T']:
            # Initialize the Kernelized Perceptron with Polynomial Kernel
            perceptron_polynomial = KernelPerceptron(kernel='polynomial', degree=degree, T=T)

            # Evaluate with Cross-Validation
            mean_accuracy_polynomial = cross_validation_score(perceptron_polynomial, X_train, y_train, cv=5)

            # Save the results
            results_polynomial.append({
                'degree': degree,
                'T': T,
                'cv_mean_accuracy': mean_accuracy_polynomial
            })

    # Select the best result based on cross-validation accuracy
    best_result_polynomial = max(results_polynomial, key=lambda x: x['cv_mean_accuracy'])

    # Print the cross-validation results for all combinations
    for result in results_polynomial:
        print(f"Degree: {result['degree']}, T: {result['T']}")
        print(f"Mean Cross-Validation Accuracy: {result['cv_mean_accuracy']:.4f}")
        print("-" * 50)

    # Print the best result
    print(f"Best Degree: {best_result_polynomial['degree']}, Best T: {best_result_polynomial['T']}")
    print(f"Best CV Mean Accuracy: {best_result_polynomial['cv_mean_accuracy']:.4f}")

    # Initialize the Kernelized Perceptron with the best parameters
    best_perceptron_polynomial = KernelPerceptron(kernel='polynomial', degree=best_result_polynomial['degree'], T=best_result_polynomial['T'])
    best_perceptron_polynomial.fit(X_train, y_train)

    # Evaluate on the test set
    test_accuracy_polynomial = best_perceptron_polynomial.score(X_test, y_test)
    test_misclassification_rate_polynomial = 1 - test_accuracy_polynomial
    print(f"Test Misclassification Rate on the test set: {test_misclassification_rate_polynomial:.4f}")


# SVM GAUSSIAN KERNEL
if __name__ == "__main__":
    param_grid_gaussian = {
        'lambda_param': [0.005, 0.01, 2.0],
        'T': [1000, 10000, 50000],
        'gamma': [0.1, 1.0]
    }

    results_gaussian = []

    for lambda_param in param_grid_gaussian['lambda_param']:
        for T in param_grid_gaussian['T']:
            for gamma in param_grid_gaussian['gamma']:
                svm_gaussian = KernelPegasosSVM(lambda_param=lambda_param, T=T, kernel='gaussian', gamma=gamma)
                mean_accuracy_gaussian = cross_validation_score(svm_gaussian, X_train, y_train, cv=5)

                results_gaussian.append({
                    'lambda_param': lambda_param,
                    'T': T,
                    'gamma': gamma,
                    'cv_mean_accuracy': mean_accuracy_gaussian
                })

    best_result_gaussian = max(results_gaussian, key=lambda x: x['cv_mean_accuracy'])

    for result in results_gaussian:
        print(f"Lambda: {result['lambda_param']}, T: {result['T']}, Gamma: {result['gamma']}")
        print(f"Mean Cross-Validation Accuracy: {result['cv_mean_accuracy']:.4f}")
        print("-" * 50)

    print(f"Best Lambda: {best_result_gaussian['lambda_param']}, Best T: {best_result_gaussian['T']}, Best Gamma: {best_result_gaussian['gamma']}")
    print(f"Best CV Mean Accuracy: {best_result_gaussian['cv_mean_accuracy']:.4f}")

    best_svm_gaussian = KernelPegasosSVM(lambda_param=best_result_gaussian['lambda_param'], T=best_result_gaussian['T'],
                                         kernel='gaussian', gamma=best_result_gaussian['gamma'])
    best_svm_gaussian.fit(X_train, y_train)

    test_accuracy_gaussian = best_svm_gaussian.score(X_test, y_test)
    test_misclassification_rate_gaussian = 1 - test_accuracy_gaussian
    print(f"Test Misclassification Rate on the test set: {test_misclassification_rate_gaussian:.4f}")


#SVM POLYNOMIAL KERNEL
if __name__ == "__main__":
    param_grid_polynomial = {
        'lambda_param': [0.005, 0.01, 0.1],
        'T': [1000, 10000, 50000],
        'degree': [2, 3, 4]
    }

    results_polynomial = []

    for lambda_param in param_grid_polynomial['lambda_param']:
        for T in param_grid_polynomial['T']:
            for degree in param_grid_polynomial['degree']:
                svm_polynomial = KernelPegasosSVM(lambda_param=lambda_param, T=T, kernel='polynomial', degree=degree)
                mean_accuracy_polynomial = cross_validation_score(svm_polynomial, X_train, y_train, cv=5)

                results_polynomial.append({
                    'lambda_param': lambda_param,
                    'T': T,
                    'degree': degree,
                    'cv_mean_accuracy': mean_accuracy_polynomial
                })

    best_result_polynomial = max(results_polynomial, key=lambda x: x['cv_mean_accuracy'])

    for result in results_polynomial:
        print(f"Lambda: {result['lambda_param']}, T: {result['T']}, Degree: {result['degree']}")
        print(f"CV Mean Accuracy: {result['cv_mean_accuracy']:.4f}")
        print("-" * 50)

    print(f"Best Lambda: {best_result_polynomial['lambda_param']}, Best Degree: {best_result_polynomial['degree']}, Best T: {best_result_polynomial['T']}")
    print(f"Best CV Mean Accuracy: {best_result_polynomial['cv_mean_accuracy']:.4f}")

    best_svm_polynomial = KernelPegasosSVM(lambda_param=best_result_polynomial['lambda_param'], T=best_result_polynomial['T'],
                                           kernel='polynomial', degree=best_result_polynomial['degree'])
    best_svm_polynomial.fit(X_train, y_train)

    test_accuracy_polynomial = best_svm_polynomial.score(X_test, y_test)
    test_misclassification_rate_polynomial = 1 - test_accuracy_polynomial
    print(f"Test Misclassification Rate on the test set: {test_misclassification_rate_polynomial:.4f}")