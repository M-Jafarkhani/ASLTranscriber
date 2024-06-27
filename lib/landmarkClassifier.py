import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from lib.util import LABELS, calculate_distance, scale_rows
import tensorflow as tf


class LandmarkClassifier:
    classifer = None
    model = None

    def __init__(self):
        pass

    def classify(self):
        df = pd.read_csv(r'distance/data.csv', index_col=False)
        X = df.drop(['label'], axis=1).copy()
        y = df.pop('label').copy().astype('int')
        scaled_data = scale_rows(X.values)
        X = pd.DataFrame(scaled_data, columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.50, random_state=9)
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print("Score = {:.2f}%".format(score * 100))
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(classifier, r'models/ASL.pkl', compress=True)

    def load_classifier(self):
        self.classifer = joblib.load(r'models/ASL.pkl')

    def new_predict(self, landmarks):
        if self.model == None:
            self.model = tf.keras.models.load_model(
                r'models/keypoint_classifier.hdf5', compile=False)
            self.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        distances = calculate_distance(landmarks)
        predict_result = self.model.predict(np.array([distances]))
        return LABELS[np.argmax(np.squeeze(predict_result))]

    def predict(self, landmarks):
        if self.classifer == None:
            self.load_classifier()
        distances = calculate_distance(landmarks)
        scaled_data = scale_rows(np.array([distances]))
        scaled_data = np.expand_dims(scaled_data[0], axis=0)
        prediction = self.classifer.predict_proba(scaled_data)[0]
        max_index = np.argmax(prediction)
        prob = prediction[max_index]
        if prob < 0.5:
            return ''
        sign = LABELS[max_index]
        return f'{sign} (%{int(prob*100)})'
