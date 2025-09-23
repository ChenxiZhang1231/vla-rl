Prepare Data for Post-Training
========================================

Before starting the post-training job, we need to prepare the data for policy training. The data should be preprocessed and stored in Parquet format, which facilitates efficient distributed data loading and processing.

We provide several data preprocessing scripts for popular datasets under the ``examples/data_preprocess/`` directory, such as ``gsm8k.py``, ``math_dataset.py``, and ``deepscaler.py``. To support a new custom dataset, you will need to create a similar script.

This document uses the ``DeepScaleR`` dataset as an example to detail the data preparation process and its specifications.

General Data Preprocessing Workflow
-----------------------------------

A typical data preprocessing script involves the following steps:

1.  **Load Raw Data**: Use a library like Hugging Face's ``datasets`` to load the original dataset from the Hub or local files.
2.  **Define Processing Logic**: Implement a core mapping function (which we often name ``make_map_fn``) to convert each sample from the original dataset into the specific format required by our framework.
3.  **Apply Transformation and Save**: Use the ``datasets.map()`` method to apply this function to the entire dataset. Then, save the processed result in Parquet format locally, with an option to upload it to a distributed file system like HDFS.

Here is a simplified framework of the process:

.. code:: python

   import argparse
   import os
   import datasets
   from siirl.utils.extras.hdfs_io import copy, makedirs

   def make_map_fn(split_name):
       # ... Define your data processing logic here ...
       def process_fn(example, idx):
           # ... Transform each data sample ...
           return transformed_data
       return process_fn

   if __name__ == '__main__':
       parser = argparse.ArgumentParser()
       # ... Define arguments ...
       args = parser.parse_args()

       # 1. Load data
       raw_dataset = datasets.load_dataset(...)
       
       # 2. Apply transformation
       processed_dataset = raw_dataset.map(function=make_map_fn('train'), with_indices=True)

       # 3. Save as Parquet
       local_dir = args.local_dir
       processed_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))

       # (Optional) Upload to HDFS
       if args.hdfs_dir:
           makedirs(args.hdfs_dir)
           copy(src=local_dir, dst=args.hdfs_dir)


DeepScaleR Dataset Processing in Practice
-------------------------------------------

Let's take ``examples/data_preprocess/deepscaler.py`` as a concrete example to demonstrate how to process the ``agentica-org/DeepScaleR-Preview-Dataset``.

The core task is to implement the ``make_map_fn`` function, which maps original fields (like ``problem``, ``answer``, and ``solution``) to the standard format required by the training framework.

.. code:: python

   data_source = "agentica-org/DeepScaleR-Preview-Dataset"
   instruction_following = 'Let\'s think step by step and output the final within \\boxed{}.'

   def make_map_fn(split_name):

       def process_fn(example, idx):
           question_raw = example.pop("problem") 
           answer_raw = example.pop("answer") 

           question = question_raw + " " + instruction_following 
           solution = example.pop("solution") 
           data = {
               "data_source": data_source,
               "prompt": [
                   {
                   "role": "user",
                       "content": question,
                   }
               ],
               "ability": "math",
               "reward_model": {"style": "rule", "ground_truth": answer_raw},
               "extra_info": {
                   "split": split_name,
                   "index": idx,
                   "answer": solution, 
                   "question": question_raw, 
               },
           }
           
           return data

       return process_fn

Data Format Specification
-------------------------

To ensure the framework can correctly parse and utilize the data, each sample processed by ``make_map_fn`` must contain the following five key fields:

1.  ``data_source``: A string indicating the source or name of the dataset. This field is used to dynamically select the corresponding reward function during training.
    - Example: ``"agentica-org/DeepScaleR-Preview-Dataset"``

2.  ``prompt``: A list used to construct the model's input, formatted to be compatible with Hugging Face's Chat Template. The data loader will automatically apply the template and tokenize the input.
    - Example: ``[{"role": "user", "content": "What is 2+2? Let's think step by step..."}]``

3.  ``ability``: A string defining the domain or capability of the current task, such as ``"math"``, ``"coding"``, or ``"general"``.

4.  ``reward_model``: A dictionary containing information needed to calculate the reward. Currently, the ``ground_truth`` field is primarily used during evaluation.
    - **Note**: The ``ground_truth`` you provide must align with the logic of the corresponding reward function you implement. For a math problem, it might be the standard answer; for code generation, it could be a set of unit tests.
    - Example: ``{"style": "rule", "ground_truth": "\\boxed{4}"}``

5.  ``extra_info``: A dictionary for storing additional metadata, such as the original dataset split (train/test) or sample index. This information is not used directly in training but is useful for debugging and data traceability.

By following these specifications, you can prepare your dataset to be used smoothly within the SiiRL post-training pipeline.