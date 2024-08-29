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
    """
    A Python class for training a model from extarcted features, and make predictions.

    ...

    Attributes
    ----------
    classifier_model : any
        Holds pre-trained model for Random Forest Classifier (RFC) approach

    dnn_model : any
        Holds pre-trained model for Deep Neural Network (DNN) approach

    featuresExtractor : FeaturesExtractor
        An instance of FeaturesExtractor class to compute features and extaract them on single image/frame

    features_folder : str
        Refers to the folder where extarcted features are saved, initialized to 'features'

    features_file : str
        Refers to the file where extarcted features are saved, initialized to 'data'

    models_folder : str
        Refers to the folder where different pre-trained models are saved, initialized to 'models'

    classifier_model_file : str
        Refers to the file where RFC apprach model is saved, initialized to 'ASL_RF'

    network_model_file : str
        Refers to the file where DNN apprach model is saved, initialized to 'ASL_DNN'

    Methods
    -------
    classify() -> None:
        The main classification process for RFC appraoch.

    load_classifier() -> None:
        Loads the pre-trained model using RFC approach.    

    load_dnn() -> None:
        Loads the pre-trained model using DNN approach.      

    predict_with_dnn(handedness: str, landmarks: any) -> str
        Makes prediction using DNN approach

    predict_with_rf(handedness: str, landmarks: any) -> str
        Makes prediction using RFC approach
    """
    classifier_model = None
    dnn_model = None
    featuresExtractor = None

    def __init__(self) -> None:
        """
        Inits the attributes, including setting features_folder to 'features',
        features_file to 'data', models_folder to 'models', classifier_model_file to 'ASL_RF'
        and network_model_file to 'ASL_DNN'

        Parameters
        ----------
        None
        """
        self.featuresExtractor = FeaturesExtractor()
        self.features_folder = 'features'
        self.features_file = 'data'
        self.models_folder = 'models'
        self.classifier_model_file = 'ASL_RF'
        self.network_model_file = 'ASL_DNN'

    def classify(self) -> None:
        """
        The main classification process for RFC appraoch.

        Parameters
        ----------
        None
        """
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

    def load_classifier(self) -> None:
        """
        Inits the classifier_model object, loading from pre-trained pkl file
        
        Parameters
        ----------
        None
        """
        self.classifier_model = joblib.load(
            f'{self.models_folder}/{self.classifier_model_file}.pkl')

    def load_dnn(self) -> None:
        """
        Inits the dnn_model object, loading from pre-trained hdf5 file
        
        Parameters
        ----------
        None
        """
        self.dnn_model = tf.keras.models.load_model(
            f'{self.models_folder}/{self.network_model_file}.hdf5', compile=False)
        self.dnn_model.compile(
            optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def predict_with_dnn(self, handedness: str, landmarks: any) -> str:
        """
        Predicts the ASL alphabet or digit from input features data, using DNN apprach. 

        Parameters
        ----------
        handedness: str
            'R' for right-hand or 'L' for left-hand.

        landmarks: any
            List of 21 landmarks, with x, y and z coordinates.

        Returns
        -------
        str
            A label, from ASL letter or digits of ASL, which probably resembles the input data.
        """
        if self.dnn_model == None:
            self.load_dnn()
        features = self.featuresExtractor.get_features(handedness, landmarks)
        features_df = pd.DataFrame([features])
        features_df = features_df.drop(columns=['label'])
        features_list = features_df.iloc[0].tolist()
        predict_result = self.dnn_model.predict(np.array([features_list]))
        return LABELS[np.argmax(np.squeeze(predict_result))]

    def predict_with_rf(self, handedness: str, landmarks: any) -> str:
        """
        Predicts the ASL alphabet or digit from input features data, using RFC apprach. 

        Parameters
        ----------
        handedness: str
            'R' for right-hand or 'L' for left-hand.

        landmarks: any
            List of 21 landmarks, with x, y and z coordinates.

        Returns
        -------
        str
            A label, from ASL letter or digits of ASL, which probably resembles the input data.
        """
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
