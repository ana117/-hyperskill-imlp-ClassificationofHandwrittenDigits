import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# load MNIST dataset
(features, target), *_ = tf.keras.datasets.mnist.load_data()

# reshape to total_image x total_pixel
total_pixels = features.shape[1] * features.shape[2]
features = features.reshape(features.shape[0], total_pixels)

# split into train and test sets
NUM_OF_ROWS = 6000
TEST_SIZE = 0.3
RAND_SEED = 40
X_train, X_test, y_train, y_test = train_test_split(features[:NUM_OF_ROWS],
                                                    target[:NUM_OF_ROWS],
                                                    test_size=TEST_SIZE,
                                                    random_state=RAND_SEED)

best_model = "none"
best_score = 0
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    global best_model, best_score
    model.fit(features_train, target_train)
    prediction = model.predict(features_test)
    score = accuracy_score(target_test, prediction)
    print(f'Model: {model}\nAccuracy: {score:.3f}\n')

    # update model with the best accuracy
    if score > best_score:
        model_name = str(model)
        best_score = score
        best_model = model_name[:model_name.find("(")]


fit_predict_eval(KNeighborsClassifier(), X_train, X_test, y_train, y_test)
fit_predict_eval(DecisionTreeClassifier(random_state=RAND_SEED), X_train, X_test, y_train, y_test)
fit_predict_eval(LogisticRegression(random_state=RAND_SEED), X_train, X_test, y_train, y_test)
fit_predict_eval(RandomForestClassifier(random_state=RAND_SEED), X_train, X_test, y_train, y_test)

print(f"The answer to the question: {best_model} - {best_score:.3f}")
