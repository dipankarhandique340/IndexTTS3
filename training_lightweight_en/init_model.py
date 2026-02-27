import os
import sys
import torch
import yaml
from omegaconf import OmegaConf

# Add parent dir to path so we can import indextts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.modules.commons import MyModel

def main():
    config_path = "config_light.yaml"
    if not os.path.exists(config_path):
        print(f"Config {config_path} not found.")
        return

    config = OmegaConf.load(config_path)

    print("Initializing lightweight GPT model...")
    # Initialize GPT
    gpt = UnifiedVoice(**config.gpt)
    
    gpt_out = "gpt_light_init.pth"
    torch.save(gpt.state_dict(), gpt_out)
    print(f"Saved initial GPT checkpoint to: {gpt_out} ({(os.path.getsize(gpt_out)/1e6):.2f} MB)")

    print("Initializing lightweight S2Mel model...")
    # Initialize S2Mel
    s2mel = MyModel(config.s2mel)
    
    s2mel_out = "s2mel_light_init.pth"
    torch.save(s2mel.state_dict(), s2mel_out)
    print(f"Saved initial S2Mel checkpoint to: {s2mel_out} ({(os.path.getsize(s2mel_out)/1e6):.2f} MB)")

    print("\nSuccess! You can now use these .pth files as your '--base-checkpoint' when training.")

if __name__ == "__main__":
    main()
