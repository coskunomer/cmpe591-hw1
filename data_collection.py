import numpy as np
import torch
from homework1 import Hw1Env

env = Hw1Env(render_mode="offscreen")

train_N = 1000  # Training data size
test_N = 200    # Test data size

def generate_data(N, save_dir):
    positions = torch.zeros(N, 2, dtype=torch.float)
    actions = torch.zeros(N, dtype=torch.uint8)
    imgs = torch.zeros(N, 3, 128, 128, dtype=torch.uint8)
    
    for i in range(N):
        print(f"Generating {save_dir} data point: {i+1}/{N}")
        action_id = np.random.randint(4)
        env.step(action_id)
        obj_pos, pixels = env.state()
        positions[i] = torch.tensor(obj_pos)
        actions[i] = action_id
        imgs[i] = pixels
        env.reset()
    
    torch.save(positions, f"{save_dir}/position.pt")
    torch.save(actions, f"{save_dir}/actions.pt")
    torch.save(imgs, f"{save_dir}/imgs.pt")
    print(f"Saved {save_dir} data.")

# training data
generate_data(train_N, "training_data")

# test data
generate_data(test_N, "test_data")
