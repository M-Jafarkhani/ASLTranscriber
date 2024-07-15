import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
from lib.featuresExtractor import FeaturesExtractor
from lib.util import LABELS
import tensorflow as tf


class LandmarkDetector:
    classifier_model = None
    cnn_model = None
    featuresExtractor = None

    def __init__(self):
        self.featuresExtractor = FeaturesExtractor()
        self.features_folder = 'features'
        self.features_file = 'data'
        self.models_folder = 'models'
        self.classifier_model_file = 'ASL_RF'
        self.network_model_file = 'ASL_CNN'

    def classify(self):
        df = pd.read_csv(
            f'{self.features_folder}/{self.features_file}.csv', index_col=False)
        X = df.drop(['label'], axis=1).copy()
        y = df.pop('label').copy().astype('int')
        X = pd.DataFrame(X, columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.75, random_state=9)
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print("Score = {:.2f}%".format(score * 100))
        if not os.path.exists(f'{self.models_folder}'):
            os.makedirs(f'{self.models_folder}')
        joblib.dump(
            classifier, f'{self.models_folder}/{self.classifier_model_file}.pkl', compress=True)

    def load_classifier(self):
        self.classifier_model = joblib.load(
            f'{self.models_folder}/{self.classifier_model_file}.pkl')

    def load_cnn(self):
        self.cnn_model = tf.keras.models.load_model(
            f'{self.models_folder}/{self.network_model_file}.hdf5', compile=False)
        self.cnn_model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def predict_with_cnn(self, handedness, landmarks):
        if self.cnn_model == None:
            self.load_cnn()
        features = self.featuresExtractor.get_features(handedness, landmarks)
        features_df = pd.DataFrame([features])
        features_df = features_df.drop(columns=['label'])
        features_list = features_df.iloc[0].tolist()
        predict_result = self.cnn_model.predict(np.array([features_list]))
        return LABELS[np.argmax(np.squeeze(predict_result))]

    def predict_with_rf(self, handedness, landmarks):
        if self.classifier_model == None:
            self.load_classifier()
        features = self.featuresExtractor.get_features(handedness, landmarks)
        features_df = pd.DataFrame([features])
        features_df = features_df.drop(columns=['label'])
        features_list = features_df.iloc[0].tolist()
        features_list = np.expand_dims(features_list, axis=0)
        prediction = self.classifier_model.predict_proba(features_list)[0]
        max_index = np.argmax(prediction)
        prob = prediction[max_index]
        if prob < 0.2:
            return ''
        sign = LABELS[max_index]
        return f'{sign} (%{int(prob*100)})'
