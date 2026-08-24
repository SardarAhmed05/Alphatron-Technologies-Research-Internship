"""
Master Deep Learning Pipeline Orchestrator (OOP)
Batch 2 - Week 2: Deep Learning Architectures (ANN, CNN, RNN, LSTM)
"""

import time
from typing import Dict, Any

from Step_1_ANN import ArtificialNeuralNetwork
from Step_2_CNN import ConvolutionalNeuralNetwork
from Step_3_RNN import RecurrentNeuralNetwork
from Step_4_LSTM import LSTMNeuralNetwork
from Step_5_Model_Comparison import DLModelComparator


class DeepLearningPipeline:
    """
    Master pipeline orchestrator that coordinates training, evaluation,
    and comparative analysis across all four foundational Deep Learning architectures.
    """

    def __init__(
        self,
        models_dir: str = "Models",
        visualizations_dir: str = "Visualizations",
        comparison_dir: str = "ModelComparison"
    ):
        self.models_dir = models_dir
        self.visualizations_dir = visualizations_dir
        self.comparison_dir = comparison_dir

        # Instantiate modular components
        self.ann_step = ArtificialNeuralNetwork(
            models_dir=self.models_dir,
            visualizations_dir=self.visualizations_dir
        )
        self.cnn_step = ConvolutionalNeuralNetwork(
            models_dir=self.models_dir,
            visualizations_dir=self.visualizations_dir
        )
        self.rnn_step = RecurrentNeuralNetwork(
            models_dir=self.models_dir,
            visualizations_dir=self.visualizations_dir
        )
        self.lstm_step = LSTMNeuralNetwork(
            models_dir=self.models_dir,
            visualizations_dir=self.visualizations_dir
        )
        self.comparator_step = DLModelComparator(
            models_dir=self.models_dir,
            output_dir=self.comparison_dir,
            visualizations_dir=self.visualizations_dir
        )

        self.pipeline_results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Executes all 5 pipeline steps sequentially."""
        total_start_time = time.time()

        print("\n" + "#" * 75)
        print("   DEEP LEARNING ARCHITECTURES - END-TO-END 5-STEP PIPELINE (OOP)")
        print("   Batch 2 - Week 2 | Alphatron Technologies Research Internship")
        print("#" * 75)

        # Step 1: ANN
        ann_res = self.ann_step.run()
        self.comparator_step.add_benchmark_result(ann_res)
        self.pipeline_results["ANN"] = ann_res

        # Step 2: CNN
        cnn_res = self.cnn_step.run()
        self.comparator_step.add_benchmark_result(cnn_res)
        self.pipeline_results["CNN"] = cnn_res

        # Step 3: SimpleRNN
        rnn_res = self.rnn_step.run()
        self.comparator_step.add_benchmark_result(rnn_res)
        self.pipeline_results["RNN"] = rnn_res

        # Step 4: LSTM
        lstm_res = self.lstm_step.run()
        self.comparator_step.add_benchmark_result(lstm_res)
        self.pipeline_results["LSTM"] = lstm_res

        # Step 5: Model Comparison & Benchmarking
        summary_df = self.comparator_step.run()
        self.pipeline_results["Summary_DF"] = summary_df

        elapsed_total = time.time() - total_start_time
        print("\n" + "#" * 75)
        print(f"   ALL 5 DEEP LEARNING PIPELINE STEPS COMPLETED IN {elapsed_total:.2f}s!")
        print("   Serialized Models: Models/")
        print("   Plots & Charts:    Visualizations/ & ModelComparison/")
        print("   Benchmark CSV:     ModelComparison/deep_learning_benchmark_results.csv")
        print("#" * 75 + "\n")

        return self.pipeline_results


if __name__ == "__main__":
    pipeline = DeepLearningPipeline()
    pipeline.run()
