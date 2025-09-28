import tensorflow_datasets as tfds
# import tyro
import os
import imageio
from tqdm import tqdm

def main(
    dataset_name: str, 
    data_dir: str, 
    output_dir: str = "extracted_videos",
    num_trajs: int = 10
):
    ds = tfds.load(dataset_name, split='train', data_dir=data_dir)
    print(f"Successfully loaded dataset: {dataset_name}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Videos will be saved to: {output_dir}")

    for i, episode in enumerate(tqdm(ds.take(num_trajs), total=num_trajs, desc="Exporting videos")):
        head_frames = []
        
        for step in episode['steps']:
            head_rgb_image = step['observation']['image_camera_head'].numpy()
            head_frames.append(head_rgb_image)
            instruction = step['language_instruction'].numpy().decode('utf-8')

        video_path = os.path.join(output_dir, f"traj_{i}_head_rgb.mp4")
        try:
            imageio.mimsave(video_path, head_frames, fps=15)
            print(f"Saved video for episode {i} to {video_path} with instruction: '{instruction}'")
        except Exception as e:
            print(f"Error saving video for episode {i}: {e}")

if __name__ == '__main__':
    # tyro.cli(main)
    main(
        dataset_name='sample_r1_lite',
        data_dir='/inspire/ssd/project/robotsimulation/public/data/Galaxea-Open-World-Dataset/rlds'
    )
