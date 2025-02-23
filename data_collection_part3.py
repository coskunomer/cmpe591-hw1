import numpy as np
import torch
from homework1 import Hw1Env

env = Hw1Env(render_mode="offscreen")

train_N = 1000  # Training data size
test_N = 200    # Test data size

def generate_data(N, save_dir):
    positions = torch.zeros(N, 2, dtype=torch.float)
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs_before = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    imgs_after = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    
    for i in range(N):
        print(f"Generating {save_dir} data point: {i+1}/{N}")
        env.reset()
        obj_pos_before, pixels_before = env.state()  # Before action
        action_id = np.random.randint(4)
        env.step(action_id)
        obj_pos_after, pixels_after = env.state()  # After action
        
        positions[i] = torch.tensor(obj_pos_after)
        actions[i] = action_id
        imgs_before[i] = pixels_before
        imgs_after[i] = pixels_after

    torch.save(positions, f"{save_dir}/position.pt")
    torch.save(actions, f"{save_dir}/actions.pt")
    torch.save(imgs_before, f"{save_dir}/imgs_before.pt")
    torch.save(imgs_after, f"{save_dir}/imgs_after.pt")
    print(f"Saved {save_dir} data.")

generate_data(train_N, "training_data_part3")
generate_data(test_N, "test_data_part3")
