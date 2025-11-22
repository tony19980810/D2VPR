import torch

def remove_prefix_in_pth(input_path, output_path, prefix="module."):
    state_dict = torch.load(input_path, map_location="cpu")['model_state_dict']

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
        else:
            new_k = k
        new_state_dict[new_k] = v

    torch.save({"model_state_dict": new_state_dict}, output_path)
    print(f"Saved modified state_dict to {output_path}")


remove_prefix_in_pth("./best_model.pth", "./best_model1.pth", prefix="module.")
