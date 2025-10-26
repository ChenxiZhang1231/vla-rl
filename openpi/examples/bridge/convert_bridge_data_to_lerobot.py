# lrm
import shutil
import sys
sys.path.append('/inspire/ssd/project/robotsimulation/public/users/zhangjiahui/vla-rl-dev/lerobot/src')
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
from pathlib import Path
import numpy as np

REPO_NAME = "/inspire/ssd/project/robotsimulation/public/data/bridge/bridge_orig_lerobot_new"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "bridge_orig",
]  # For simplicity we will combine multiple Libero datasets into one training dataset

data_dir = "/inspire/ssd/project/robotsimulation/public/data/bridge"
def main(*, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = Path(REPO_NAME)
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="widowx",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            steps_iter = episode["steps"].as_numpy_iterator()
            try:
                curr = next(steps_iter)
            except StopIteration:
                continue
            
            for nxt in steps_iter:
                curr_img   = curr["observation"]["image_0"]          # (256,256,3), uint8
                curr_state = curr["observation"]["state"].astype(np.float32)  # (7,)
                next_state = nxt["observation"]["state"].astype(np.float32)   # (7,)
                delta = next_state - curr_state                                     # (7,)
                delta[-1] = curr["action"][-1]

                lang = curr["language_instruction"]
                if isinstance(lang, (bytes, bytearray)):
                    lang = lang.decode()

                dataset.add_frame(
                    {
                        "image":   curr_img,
                        "state":   curr_state,
                        "actions": delta,           # 用 delta 代替原 action
                    },
                    task=lang,
                )
                curr = nxt
                
                
            last_img   = curr["observation"]["image_0"]
            last_state = curr["observation"]["state"].astype(np.float32)
            last_lang  = curr["language_instruction"]
            if isinstance(last_lang, (bytes, bytearray)):
                last_lang = last_lang.decode()

            dataset.add_frame(
                {
                    "image":   last_img,
                    "state":   last_state,
                    "actions": np.zeros_like(last_state, dtype=np.float32),
                },
                task=last_lang,
            )

            dataset.save_episode()
            
            #     dataset.add_frame(
            #         {
            #             "image": step["observation"]["image_0"],
            #             "state": step["observation"]["state"],
            #             "actions": step["action"],
            #             # "task": step["language_instruction"].decode(),
            #         },
            #         task=step["language_instruction"].decode()
            #     )
            # dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    # if push_to_hub:
    #     dataset.push_to_hub(
    #         tags=["libero", "panda", "rlds"],
    #         private=False,
    #         push_videos=True,
    #         license="apache-2.0",
    #     )


if __name__ == "__main__":
    tyro.cli(main)
