import os
import torch
import numpy as np
from PIL import Image
from IPython.display import display
import torchvision.utils as vutils
from tqdm.notebook import tqdm


from DiT import DiT_S_2, DiT_S_4, DiT_S_8



# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# CUDA optimization
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Define diffusion parameters
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=2e-2):
    """Linear beta schedule as used in your training script"""
    return torch.linspace(beta_start, beta_end, timesteps)

class CIFAR10Sampler:
    def __init__(self, model, timesteps=1000, device=device):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Define beta schedule and derived quantities
        self.betas = linear_beta_schedule(timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # For sampling
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]])
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    @torch.no_grad()
    def sample_timestep(self, x, t, y, cfg_scale=3.0):
        """Sample from the model at a specific timestep"""
        batch_size = x.shape[0]
        
        # Double batch for classifier-free guidance
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        
        # Create conditional and unconditional inputs
        y_uncond = torch.ones_like(y) * 10  # Use 10 as unconditional class (outside CIFAR-10 range)
        y_in = torch.cat([y, y_uncond], dim=0)
        
        # Get model prediction
        noise_pred = self.model(x_in, t_in, y_in)
        
        # Extract noise predictions (first 3 channels are noise prediction)
        noise_pred = noise_pred[:, :3]
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        
        # Apply classifier-free guidance
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        # Get alpha values for this timestep
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]
        
        # Reshape for broadcasting
        alpha = alpha.view(-1, 1, 1, 1)
        alpha_cumprod = alpha_cumprod.view(-1, 1, 1, 1)
        beta = beta.view(-1, 1, 1, 1)
        
        # Generate the previous sample
        if t[0] > 0:  # Not the final step
            noise = torch.randn_like(x)
            variance = self.posterior_variance[t].view(-1, 1, 1, 1)
            
            # Calculate mean
            x_prev = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            )
            
            # Add noise scaled by variance
            x_prev = x_prev + torch.sqrt(variance) * noise
        else:  # Final step (t=0)
            x_prev = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            )
        
        return x_prev

    @torch.no_grad()
    def sample(self, shape, class_labels, cfg_scale=3.0):
        """Generate samples using the reverse diffusion process"""
        b = shape[0]
        
        # Start from random noise
        x = torch.randn(shape, device=self.device)
        
        # Sampling loop
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling", total=self.timesteps):
            # Full batch with the same timestep
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            
            # Sample x_{t-1} from x_t
            x = self.sample_timestep(x, t, class_labels, cfg_scale)
        
        # Normalize to [0, 1] for saving
        x = (x + 1) / 2.0
        return x

# Function to load the model from checkpoint
def load_model(checkpoint_path, model_variant='S_2'):
    """Load your trained DiT model"""
    # Select model architecture based on variant
    if model_variant == 'S_2':
        model = DiT_S_2(input_size=32, in_channels=3, num_classes=10, learn_sigma=True)
    elif model_variant == 'S_4':
        model = DiT_S_4(input_size=32, in_channels=3, num_classes=10, learn_sigma=True)
    elif model_variant == 'S_8':
        model = DiT_S_8(input_size=32, in_channels=3, num_classes=10, learn_sigma=True)
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Use EMA model if available (typically gives better results)
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("Loaded EMA model weights")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")
    
    return model.to(device).eval()

# Main generation function - run this in your notebook
def generate_cifar10_images(
    checkpoint_path,
    output_dir='generated_images',
    model_variant='S_2',
    num_images=16,
    class_labels=None,  # Specify class labels or None to generate one per class
    cfg_scale=3.0,
    seed=42,
    timesteps=1000
):
    """Generate CIFAR-10 images using your trained DiT model"""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model = load_model(checkpoint_path, model_variant)
    print(f"Model loaded successfully: {model_variant}")
    
    # Create sampler
    sampler = CIFAR10Sampler(model, timesteps=timesteps, device=device)
    
    # Set up class labels
    if class_labels is None:
        # Generate one image per CIFAR-10 class
        class_labels = list(range(10))
        num_images = max(num_images, 10)  # Ensure we generate at least one per class
    
    # Make sure we have enough class labels for all images
    if len(class_labels) < num_images:
        # Repeat labels if needed
        class_labels = (class_labels * (num_images // len(class_labels) + 1))[:num_images]
    else:
        # Truncate if we have too many
        class_labels = class_labels[:num_images]
    
    # Convert to tensor
    class_labels = torch.tensor(class_labels, device=device)
    
    # Generate images
    print(f"Generating {num_images} images with class labels: {class_labels.cpu().numpy()}")
    
    # Define shape: [batch_size, channels, height, width]
    shape = (num_images, 3, 32, 32)
    
    # Sample images
    samples = sampler.sample(shape, class_labels, cfg_scale=cfg_scale)
    
    # Save individual images
    for i in range(num_images):
        img_path = f"{output_dir}/class_{class_labels[i].item()}_seed_{seed}_img_{i}.png"
        vutils.save_image(samples[i], img_path)
    
    # Create grid of all images
    grid_size = int(np.ceil(np.sqrt(num_images)))
    grid = vutils.make_grid(samples, nrow=grid_size, padding=2, normalize=False)
    grid_path = f"{output_dir}/grid_seed_{seed}.png"
    vutils.save_image(grid, grid_path)
    
    # Display grid
    try:
        grid_img = Image.open(grid_path)
        display(grid_img)
    except:
        print(f"Grid saved to {grid_path}")
    
    print(f"Generated {num_images} images in directory: {output_dir}")
    return grid_path

if __name__ == "__main__":
# Path to your checkpoint file from training
    checkpoint_path = 'results/checkpoints/dit_best_model.pt' 

    # Generate images
    generate_cifar10_images(
        checkpoint_path=checkpoint_path,
        output_dir='generated_samples',
        model_variant='S_2',  # Use S_2, S_4, or S_8 depending on which you trained
        num_images=16,
        class_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Generate images from all classes
        cfg_scale=3.0,  # Higher values = stronger class conditioning
        seed=42,
        timesteps=1000  # Can be reduced for faster sampling (e.g., 250)
    )


    # CIFAR-10 classes for reference
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]