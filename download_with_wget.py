import os
import sys
import subprocess

# Create directories if they don't exist
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

# Download file using wget
def download_with_wget(url, output_path):
    if os.path.exists(output_path):
        print(f"File already exists: {output_path}")
        return
        
    print(f"Downloading {url} to {output_path}")
    cmd = ["wget", "-c", "--progress=bar:force", url, "-O", output_path]
    subprocess.run(cmd)
    
    if os.path.exists(output_path):
        filesize = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
        print(f"Download complete: {output_path} ({filesize:.2f} MB)")
    else:
        print(f"Error: Failed to download {url}")

def download_model(model_name="bdsqlsz/flux1-dev2pro-single"):
    # Create directories
    unet_folder = "models/unet/bdsqlsz/flux1-dev2pro-single"
    vae_folder = "models/vae"
    clip_folder = "models/clip"
    
    ensure_dir(unet_folder)
    ensure_dir(vae_folder)
    ensure_dir(clip_folder)
    
    # Download model files
    print("Downloading model files...")
    
    # Download UNET model
    print("Downloading UNET model...")
    unet_path = os.path.join(unet_folder, "flux1-dev2pro.safetensors")
    unet_url = "https://huggingface.co/bdsqlsz/flux1-dev2pro-single/resolve/main/flux1-dev2pro.safetensors"
    download_with_wget(unet_url, unet_path)
    
    # Download VAE
    print("Downloading VAE model...")
    vae_path = os.path.join(vae_folder, "ae.sft")
    vae_url = "https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/ae.sft"
    download_with_wget(vae_url, vae_path)
    
    # Download CLIP
    print("Downloading CLIP model...")
    clip_path = os.path.join(clip_folder, "clip_l.safetensors")
    clip_url = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
    download_with_wget(clip_url, clip_path)
    
    # Download T5XXL
    print("Downloading T5XXL model...")
    t5_path = os.path.join(clip_folder, "t5xxl_fp16.safetensors")
    t5_url = "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
    download_with_wget(t5_url, t5_path)
    
    print("All downloads complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        download_model(model_name)
    else:
        download_model()