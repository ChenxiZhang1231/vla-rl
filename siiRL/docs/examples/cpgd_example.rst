DeepScaleR Example with CPGD
==============================

Introduction
------------

This example demonstrates how to fine-tune a Large Language Model for advanced mathematical reasoning on the **DeepScaleR** dataset using **Clipped Policy Gradient Optimization with Policy Drift (CPGD)**, a novel reinforcement learning algorithm designed for enhanced training stability.

**Paper:** `CPGD: Toward Stable Rule-based Reinforcement Learning for Language Models <https://arxiv.org/abs/2505.12504>`__

**Dataset:** https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset

While algorithms like PPO and GRPO are powerful, they can sometimes suffer from instability due to their reliance on importance-sampling ratios in the loss function. CPGD is proposed to mitigate these issues by providing a more stable policy update mechanism, making it a robust choice for complex reasoning tasks.

CPGD Algorithm Overview
-----------------------

CPGD enhances training stability by making two key modifications to the standard policy gradient approach:

1.  **Clipped Policy Gradient Objective**: Instead of directly using the policy ratio in the loss (which can cause high variance), CPGD uses a policy gradient objective. It then applies a clipping mechanism to the *logarithm* of the policy ratio. This prevents excessive policy updates when the ratio becomes too large, effectively keeping the optimization within a trusted region.
2.  **Policy Drift Regularization**: CPGD introduces a *policy drift* term, which is a KL divergence penalty between the current policy and the old policy from the start of the training iteration. This acts as a dynamic regularizer, pulling the policy back if it strays too far, too quickly, thus preventing training collapse.

Together, these features allow CPGD to achieve consistent performance improvements while avoiding the instability often seen in other RL algorithms.

Step 1: Prepare the Dataset
---------------------------

The data preparation process is identical to other examples using this dataset. First, preprocess the DeepScaleR dataset into the required Parquet format.

.. code:: bash

   cd examples/data_preprocess
   python3 deepscaler.py --local_dir ~/data/deepscaler

This command downloads, processes, and saves the training and testing sets in the `~/data/deepscaler` directory.

Step 2: Download the Pre-trained Model
--------------------------------------

You need a base model to start the CPGD training. In this example, we use `Qwen2.5-7B-Instruct`.

- **Recommended: Download via CLI:** Use a tool like `huggingface-cli` to download the model to a local directory.

  .. code:: bash

     huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ~/data/models/Qwen2.5-7B-Instruct

- **Automatic Download:** You can also specify the model name directly in the `actor_rollout_ref.model.path` field of the run script, and the framework will download it automatically.

Step 3: Perform CPGD Training
-----------------------------

With the data and model ready, you can now launch the training job using the CPGD algorithm.

**Reward Function**

For this task, we use the same rule-based reward function as in the PPO/GRPO examples. The framework's default reward mechanism performs an exact match on the final answer within the `\\boxed{...}` block. A correct answer receives a positive reward, and an incorrect one receives zero.

**Training Script**

Below is a complete training script from `examples/cpgd_trainer/run_qwen2_5-7b.sh`. It is configured to use the CPGD algorithm (`algorithm.adv_estimator=cpgd`). Note the presence of CPGD-specific parameters like `actor_rollout_ref.actor.policy_drift_coeff` and `algorithm.weight_factor_in_cpgd`.

.. literalinclude:: ../../examples/cpgd_trainer/run_qwen2_5-7b.sh
   :language: bash
   :caption: examples/cpgd_trainer/run_qwen2_5-7b.sh
