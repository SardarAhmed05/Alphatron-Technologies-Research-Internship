"""
Master Machine Learning Pipeline Orchestrator (Main.py)
======================================================
Executes all 11 steps of the Heart Disease Machine Learning workflow sequentially.

Author: Sardar Ahmed
Internship: Alphatron Technologies Research Internship (Week 1)
"""

from Step_1_DataPreparation import run_data_preparation
from Step_2_EDA import run_eda
from Step_3_Data_Transformation import run_data_transformation
from Step_4_FeatureSelection import run_feature_selection
from Step_5_Train_Test_DataSplit import run_data_split
from Step_6_Model_Training import run_model_training
from Step_7_ModelTesting_Performceevaluation import run_model_evaluation
from Step_8_DataSampling import run_data_sampling
from Step_9_Cross_Validation import run_cross_validation
from Step_10_HyperparameterOptimization import run_hyperparameter_optimization
from Step_11_Model_Comparison import run_model_comparison

def main():
    print("\n" + "#"*65)
    print("   HEART DISEASE PREDICTION - END-TO-END 11-STEP ML PIPELINE")
    print("#"*65)

    run_data_preparation()
    run_eda()
    run_data_transformation()
    run_feature_selection()
    run_data_split()
    run_model_training()
    run_model_evaluation()
    run_data_sampling()
    run_cross_validation()
    run_hyperparameter_optimization()
    run_model_comparison()

    print("\n" + "#"*65)
    print("   ALL 11 PIPELINE STEPS EXECUTED AND COMPLETED SUCCESSFULLY!")
    print("#"*65 + "\n")

if __name__ == "__main__":
    main()
