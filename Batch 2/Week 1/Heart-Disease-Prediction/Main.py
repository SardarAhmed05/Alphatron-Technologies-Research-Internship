"""
Master Machine Learning Pipeline Orchestrator (OOP)
"""

from Step_1_DataPreparation import DataPreparation
from Step_2_EDA import ExploratoryDataAnalysis
from Step_3_Data_Transformation import DataTransformation
from Step_4_FeatureSelection import FeatureSelector
from Step_5_Train_Test_DataSplit import DataSplitter
from Step_6_Model_Training import ModelTrainer
from Step_7_ModelTesting_Performceevaluation import ModelEvaluator
from Step_8_DataSampling import DataSampler
from Step_9_Cross_Validation import CrossValidator
from Step_10_HyperparameterOptimization import HyperparameterOptimizer
from Step_11_Model_Comparison import ModelComparator


class HeartDiseasePipeline:
    """
    Master pipeline controller that orchestrates all 11 modular steps
    using clean Object-Oriented dependency instantiation.
    """

    def __init__(self, raw_data_path: str = "heart_disease_uci.csv"):
        self.raw_data_path = raw_data_path
        self.step1 = DataPreparation(input_path=self.raw_data_path)
        self.step2 = ExploratoryDataAnalysis()
        self.step3 = DataTransformation()
        self.step4 = FeatureSelector()
        self.step5 = DataSplitter()
        self.step6 = ModelTrainer()
        self.step7 = ModelEvaluator()
        self.step8 = DataSampler()
        self.step9 = CrossValidator()
        self.step10 = HyperparameterOptimizer()
        self.step11 = ModelComparator()

    def run(self):
        """Executes all 11 pipeline steps sequentially."""
        print("\n" + "#"*65)
        print("   HEART DISEASE PREDICTION - END-TO-END 11-STEP ML PIPELINE (OOP)")
        print("#"*65)

        self.step1.run()
        self.step2.run()
        self.step3.run()
        self.step4.run()
        self.step5.run()
        self.step6.run()
        self.step7.run()
        self.step8.run()
        self.step9.run()
        self.step10.run()
        self.step11.run()

        print("\n" + "#"*65)
        print("   ALL 11 PIPELINE STEPS EXECUTED AND COMPLETED SUCCESSFULLY!")
        print("#"*65 + "\n")


if __name__ == "__main__":
    pipeline = HeartDiseasePipeline()
    pipeline.run()
