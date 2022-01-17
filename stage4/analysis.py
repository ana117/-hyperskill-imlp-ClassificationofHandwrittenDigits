import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer

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

# normalize data
transformer = Normalizer()
X_train_norm = transformer.transform(X_train)
X_test_norm = transformer.transform(X_test)

scores = []
def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    global scores
    model.fit(features_train, target_train)
    prediction = model.predict(features_test)
    score = accuracy_score(target_test, prediction)
    print(f'Model: {model}\nAccuracy: {score:.3f}\n')

    model_name = str(model)
    model_name = model_name[:model_name.find("(")]
    scores.append((round(score, 3), model_name))


# fit_predict_eval(KNeighborsClassifier(), X_train, X_test, y_train, y_test)
# fit_predict_eval(DecisionTreeClassifier(random_state=RAND_SEED), X_train, X_test, y_train, y_test)
# fit_predict_eval(LogisticRegression(random_state=RAND_SEED), X_train, X_test, y_train, y_test)
# fit_predict_eval(RandomForestClassifier(random_state=RAND_SEED), X_train, X_test, y_train, y_test)

# normalized data
fit_predict_eval(KNeighborsClassifier(), X_train_norm, X_test_norm, y_train, y_test)
fit_predict_eval(DecisionTreeClassifier(random_state=RAND_SEED), X_train_norm, X_test_norm, y_train, y_test)
fit_predict_eval(LogisticRegression(random_state=RAND_SEED), X_train_norm, X_test_norm, y_train, y_test)
fit_predict_eval(RandomForestClassifier(random_state=RAND_SEED), X_train_norm, X_test_norm, y_train, y_test)

scores.sort(reverse=True)
print("The answer to the 1st question: yes\n")  # accuracy increased
print(f"The answer to the 2nd question: {scores[0][1]}-{scores[0][0]:.3f}, {scores[1][1]}-{scores[1][0]:.3f}")
