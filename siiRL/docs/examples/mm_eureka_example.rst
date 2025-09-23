MM-Eureka Example with GRPO
===========================

Introduction
------------

This guide details how to fine-tune a multi-modal Large Language Model using the **Group Relative Policy Optimization (GRPO)** algorithm on the **MM-Eureka** dataset. MM-Eureka is a challenging dataset designed to test mathematical reasoning that requires interpreting both text and images.

**Paper:** https://arxiv.org/pdf/2503.07365.

**Dataset:** https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset

The goal is to enhance a model's ability to perform complex reasoning by processing visual and textual information simultaneously. We use GRPO, an advanced RL algorithm, to optimize the model's policy.

Dataset Overview
----------------

MM-Eureka problems consist of a text-based question paired with one or more images. The model must understand the content of the image to solve the problem correctly.

**An example from MM-Eureka:**

**Prompt:**
   .. image:: https://github.com/sii-research/siiRL/raw/main/docs/_static/cube.jpg
      :width: 50%

   Question: A cube loses one vertex after a 'corner' is removed. This geometric shape is ___ (fill in the number).

**Answer:**
   3

Step 1: Data Preprocessing
--------------------------

The raw MM-Eureka dataset, typically in `.jsonl` format, must be converted to Parquet. This involves not only structuring the text but also processing the associated images.

The script `examples/data_preprocess/mm_eureka.py` handles this. It performs the following actions:
- Parses each line of the input JSONL file.

- Reads the image file specified in `image_urls` and embeds its byte content directly into the Parquet file.

- Formats the user prompts to include instructions for the desired output structure (`<think>...</think><answer>...</answer>`).

- Splits the data into training and testing sets.

Run the script with your dataset file:

.. code:: bash

   cd examples/data_preprocess
   python3 mm_eureka.py --jsonl_file /path/to/your/mm_eureka_data.jsonl --output_dir ~/data/mm_eureka/

Step 2: Defining the Reward Score
---------------------------------

A custom reward function is crucial for multi-modal reasoning. For MM-Eureka, we use a composite score defined in `siirl/utils/reward_score/mm_eureka.py`. This function evaluates two aspects of the model's response:

1.  **Accuracy Reward**: This is the primary component. It parses the mathematical expression from the model's output (often in LaTeX) and compares it against the ground truth using the `math_verify` utility. This provides a robust check for mathematical correctness.
2.  **Format Reward**: A smaller, secondary reward is given if the model correctly follows the required `<think>...</think><answer>...</answer>` structure. This encourages the model to generate well-formed, interpretable reasoning chains.

The final reward is a weighted sum of these two components (e.g., `0.9 * accuracy_reward + 0.1 * format_reward`), balancing correctness with style.

Step 3: Download the Pre-trained Model
--------------------------------------

For this multi-modal task, we use a powerful vision-language model like `Qwen2.5-VL-7B-Instruct`. Ensure the model is available locally for the training script.

- **Recommended: Download via CLI:**

  .. code:: bash

     # For Hugging Face
     huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ~/data/models/Qwen2.5-VL-7B-Instruct
     
     # For ModelScope
     modelscope download Qwen/Qwen2.5-VL-7B-Instruct --local_dir ~/data/models/Qwen2.5-VL-7B-Instruct

- **Automatic Download:** Alternatively, specify the model identifier directly in the run script's `actor_rollout_ref.model.path` field.

Step 4: Perform GRPO Training
-----------------------------

With the data and model prepared, you can launch the training job using the GRPO algorithm.

**Training Script**

The script `examples/grpo_trainer/run_qwen2_5_vl-7b.sh` provides a complete configuration for this task. It sets up the environment, Ray cluster, and all necessary hyperparameters for GRPO training on the MM-Eureka dataset. Adapt the `HOME` path and other variables as needed for your environment.

.. literalinclude:: ../../examples/grpo_trainer/run_qwen2_5_vl-7b.sh
   :language: bash
   :caption: examples/grpo_trainer/run_qwen2_5_vl-7b.sh 