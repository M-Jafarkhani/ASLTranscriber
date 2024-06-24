from lib.datasetAnnotator import DatasetAnnotator
from lib.distanceExtractor import DistanceExtractor
from lib.handLandmarker import HandLandmarker
from lib.landmarkClassifier import LandmarkClassifier
from lib.signTranscriber import SignTranscriber


def main():
    # datasetAnnotator = DatasetAnnotator()
    # datasetAnnotator.annotate()

    # distanceExtractor = DistanceExtractor()
    # distanceExtractor.extract()

    # classifier = LandmarkClassifier()
    # classifier.classify()

    transcriber = SignTranscriber()
    transcriber.transcribe()

if __name__ == "__main__":
    main()
