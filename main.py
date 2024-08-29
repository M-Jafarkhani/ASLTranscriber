from lib.datasetAnnotator import DatasetAnnotator
from lib.featuresExtractor import FeaturesExtractor
from lib.landmarkDetector import LandmarkDetector
from lib.signTranscriber import SignTranscriber


def main() -> None:
    """
    The main function to orchestrate the workflow.

    Parameters
    ----------
    None      

    Returns
    -------
    None
    """        
    # datasetAnnotator = DatasetAnnotator()
    # datasetAnnotator.annotate()

    # featuresExtractor = FeaturesExtractor()
    # featuresExtractor.extract()

    # detector = LandmarkDetector()
    # detector.classify()

    transcriber = SignTranscriber()
    transcriber.transcribe()

if __name__ == "__main__":
    main()
