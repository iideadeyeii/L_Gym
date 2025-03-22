import os
import sys
import time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

# Create directories if they don't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

# Download function with progress monitoring
def download_with_progress(repo_id, filename, local_dir):
    print(f"Starting download of {filename} from {repo_id}...")
    start_time = time.time()
    
    # Set resume_download=True to resume interrupted downloads
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        resume_download=True,
        force_download=False,
        token=None,  # Add your token here if needed for private repos
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Download completed in {duration:.1f} seconds: {file_path}")
    return file_path

# Download function for specific model
def download_model(model_name):
    if model_name == "bdsqlsz/flux1-dev2pro-single":
        # Create directories
        unet_folder = "models/unet/bdsqlsz/flux1-dev2pro-single"
        vae_folder = "models/vae"
        clip_folder = "models/clip"
        
        ensure_dir(unet_folder)
        ensure_dir(vae_folder)
        ensure_dir(clip_folder)
        
        # Download model file
        print("Downloading model files...")
        
        # Download UNET model
        print("Downloading UNET model...")
        unet_path = os.path.join(unet_folder, "flux1-dev2pro.safetensors")
        if not os.path.exists(unet_path):
            print(f"Downloading flux1-dev2pro.safetensors from bdsqlsz/flux1-dev2pro-single")
            download_with_progress(
                repo_id="bdsqlsz/flux1-dev2pro-single", 
                filename="flux1-dev2pro.safetensors",
                local_dir=unet_folder
            )
        else:
            print(f"UNET model already exists at {unet_path}")
        
        # Download VAE
        print("Downloading VAE model...")
        vae_path = os.path.join(vae_folder, "ae.sft")
        if not os.path.exists(vae_path):
            print(f"Downloading ae.sft from cocktailpeanut/xulf-dev")
            download_with_progress(
                repo_id="cocktailpeanut/xulf-dev", 
                filename="ae.sft",
                local_dir=vae_folder
            )
        else:
            print(f"VAE model already exists at {vae_path}")
        
        # Download CLIP
        print("Downloading CLIP models...")
        clip_l_path = os.path.join(clip_folder, "clip_l.safetensors")
        if not os.path.exists(clip_l_path):
            print(f"Downloading clip_l.safetensors from comfyanonymous/flux_text_encoders")
            download_with_progress(
                repo_id="comfyanonymous/flux_text_encoders", 
                filename="clip_l.safetensors",
                local_dir=clip_folder
            )
        else:
            print(f"CLIP model already exists at {clip_l_path}")
        
        # Download T5XXL
        t5xxl_path = os.path.join(clip_folder, "t5xxl_fp16.safetensors")
        if not os.path.exists(t5xxl_path):
            print(f"Downloading t5xxl_fp16.safetensors from comfyanonymous/flux_text_encoders")
            download_with_progress(
                repo_id="comfyanonymous/flux_text_encoders", 
                filename="t5xxl_fp16.safetensors",
                local_dir=clip_folder
            )
        else:
            print(f"T5XXL model already exists at {t5xxl_path}")
    else:
        print(f"Model {model_name} not recognized")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "bdsqlsz/flux1-dev2pro-single"
    
    print(f"Downloading model: {model_name}")
    download_model(model_name)
    print("Download complete!")