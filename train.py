import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict
from time import time

from DiT import DiT_S_2, DiT_S_4, DiT_S_8  
from diffusers.models import AutoencoderKL
# Import the diffusion module
from diffusion import create_diffusion

# CUDA optimization - essential for training efficiency
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


#################################################################################
#                             EMA and Utility Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                             Dataset Preparation                               #
#################################################################################

def prepare_dataset(args):
    """
    Prepare the CIFAR-10 dataset with proper augmentations
    """
    # Data augmentation and normalization
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val
    )
    
    # Store the original transform to re-apply later
    train_transform = trainset.transform
    
    # Split training data into train and validation sets
    if args.val_size > 0:
        val_size = int(args.val_size * len(trainset))
        train_size = len(trainset) - val_size
        trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])
        
        # Apply appropriate transforms to each subset
        # For the validation set, we want to use the validation transform
        if hasattr(valset, 'dataset'):
            valset.dataset.transform = transform_val
    else:
        valset = testset  # If no validation set is needed, use the test set
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    valloader = torch.utils.data.DataLoader(
        valset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return trainloader, valloader, train_transform


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DiT Training on CIFAR-10')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--model-variant', type=str, default='S_2', choices=['S_2', 'S_4', 'S_8'], 
                        help='DiT model variant (S_2, S_4, S_8)')
    parser.add_argument('--timestep-respacing', type=str, default="", help='Timestep respacing string for diffusion')
    parser.add_argument('--val-size', type=float, default=0.1, help='Fraction of training data to use for validation')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save-interval', type=int, default=5, help='Checkpoint saving interval (epochs)')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--use-vae', action='store_true', help='Use VAE to encode images to latent space')
    parser.add_argument('--vae', type=str, choices=['ema', 'mse'], default='ema', help='VAE model to use (ema or mse)')
    parser.add_argument('--latent-size', type=int, default=4, help='Size of latent space (original / 8)')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device and optimize for VRAM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for CUDA
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = f"{args.results_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model based on variant
    if args.model_variant == 'S_2':
        model_fn = DiT_S_2
    elif args.model_variant == 'S_4':
        model_fn = DiT_S_4
    elif args.model_variant == 'S_8':
        model_fn = DiT_S_8
    
    # Initialize VAE if requested
    vae = None
    in_channels = 3
    input_size = 32
    
    if args.use_vae:
        print(f"Using VAE model: {args.vae}")
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        # Freeze VAE parameters
        for param in vae.parameters():
            param.requires_grad = False
        vae.eval()  # Keep VAE in eval mode
        
        # When using VAE, DiT works in the latent space
        in_channels = 4  # VAE latent channels
        input_size = args.latent_size  # VAE reduces spatial dimensions by 8x
    
    # Create model
    model = model_fn(
        input_size=input_size,
        in_channels=in_channels,  # 3 for RGB or 4 for VAE latents
        num_classes=10,  # CIFAR-10 has 10 classes
        learn_sigma=True
    ).to(device)
    
    # Create EMA model
    ema_model = deepcopy(model).to(device)
    requires_grad(ema_model, False)  # EMA model should not require gradients
    
    # Create diffusion process using the imported function
    diffusion = create_diffusion(timestep_respacing=args.timestep_respacing)  # default: 1000 steps, linear noise schedule
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DiT parameters: {total_params:,}")
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Prepare dataset
    trainloader, valloader, train_transform = prepare_dataset(args)
    print(f"Training on {len(trainloader.dataset)} samples, validating on {len(valloader.dataset)} samples")
    
    # Resume training if checkpoint is provided
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Loaded checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    # Variables for monitoring/logging
    running_loss = 0
    log_steps = 0
    start_time = time()
    
    print(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        # Training phase
        model.train()  # Important! This enables dropout for classifier-free guidance
        epoch_loss = 0.0
        
        # Progress bar for training
        train_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # If using VAE, encode images to latent space
            if args.use_vae:
                with torch.no_grad():
                    images = vae.encode(images).latent_dist.sample().mul_(0.18215)
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device).long()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass and loss calculation with mixed precision
            with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                # Use the diffusion model's training_losses function
                model_kwargs = dict(y=labels)
                loss_dict = diffusion.training_losses(model, images, t, model_kwargs)
                loss = loss_dict["loss"].mean()
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping to improve stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA model
            update_ema(ema_model, model)
            
            # Log loss
            running_loss += loss.item()
            log_steps += 1
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                "loss": loss.item(), 
                "lr": optimizer.param_groups[0]['lr'],
                "step": global_step
            })
            
            # Log metrics periodically
            if global_step % args.log_interval == 0:
                # Measure training speed
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                # Log metrics
                avg_loss = running_loss / log_steps
                print(f"(step={global_step:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                
                # Reset monitoring variables
                running_loss = 0
                log_steps = 0
                start_time = time()
            
            # Free up GPU memory if needed
            if torch.cuda.is_available() and args.batch_size > 32:
                torch.cuda.empty_cache()
        
        # Calculate average epoch loss
        epoch_loss /= len(trainloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Train Loss: {epoch_loss:.6f}")
        
        # Validation phase
        model.eval()  # Important! This disables randomized embedding dropout
        val_loss = 0.0
        
        # Progress bar for validation
        val_pbar = tqdm(valloader, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]")
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                # If using VAE, encode images to latent space
                if args.use_vae:
                    images = vae.encode(images).latent_dist.sample().mul_(0.18215)
                
                # Sample random timesteps
                t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=device).long()
                
                # Get loss with mixed precision
                with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu'):
                    model_kwargs = dict(y=labels)
                    loss_dict = diffusion.training_losses(model, images, t, model_kwargs)
                    loss = loss_dict["loss"].mean()
                
                val_loss += loss.item()
                
                # Update progress bar
                val_pbar.set_postfix({"val_loss": loss.item()})
        
        val_loss /= len(valloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Validation Loss: {val_loss:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Save model checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'args': args
            }
            checkpoint_path = f"{checkpoint_dir}/dit_checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"Model saved at epoch {epoch+1} to {checkpoint_path}")
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'args': args
            }
            best_model_path = f"{checkpoint_dir}/dit_best_model.pt"
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved with validation loss: {val_loss:.6f}")
    
    # Final message
    print("Training complete!")


if __name__ == "__main__":
    main()