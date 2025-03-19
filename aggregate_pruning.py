import fire
from safetensors import safe_open
import numpy as np
import glob
import os


def get_log_alpha_heads(filename):
    try:
        with safe_open(filename, framework="pt", device="cpu") as f:
            ret = np.empty((12, 12))

            for key in f.keys():
                if "log_alpha" in key:
                    idx = int(
                        key.split(".")[2]
                    )  # 'gpt_neox.layers.1.attention.log_alpha_heads
                    ret[idx] = f.get_tensor(key).numpy()
            return ret
    except Exception as e:
        print(f"Error reading safetensors file: {e}")
        return None


def main(super_dir):
    np.random.seed(0)
    dirs = glob.glob(super_dir + "/*/")
    print(super_dir)
    num_masks = len(dirs)

    alpha_head_stack = np.zeros((12, 12, num_masks))
    for i, dir in enumerate(dirs):
        log_alpha_heads = get_log_alpha_heads(
            os.path.join(dir, "checkpoint-5000/", "model.safetensors")
        )
        if log_alpha_heads is not None:
            alpha_head_stack[:, :, i] = log_alpha_heads

    aggregated_params = np.min(alpha_head_stack, axis=2)

    heads_remaining = np.sum(aggregated_params > 0)
    print(f"% remaining heads: {heads_remaining / 144}")

    # save aggregated params
    output_path = os.path.join(super_dir, "aggregated_params.npy")
    np.save(output_path, aggregated_params)

    # flip the params
    output_path_flipped = os.path.join(super_dir, "aggregated_params_flipped.npy")
    np.save(output_path_flipped, -aggregated_params)

    p = heads_remaining / 144
    for i in range(5):
        random_params = np.random.choice([1, -1], (12, 12), [p, 1 - p]) * 5
        output_name = os.path.join(super_dir, f"random_params_{i}.npy")
        np.save(output_name, random_params)


if __name__ == "__main__":
    fire.Fire({"main": main})
