import time
from typing import Tuple

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True

from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video
from utils import sigmoid_beta_schedule
from einops import rearrange
from torch import autocast
import os, io, asyncio, struct
from PIL import Image
import frontendmanager
import numpy as np

assert torch.cuda.is_available()
device = "cuda:0"
# Sampling params
model_path = "oasis500m.pt"
vae_path = "vit-l-20.pt"
B = 1
max_noise_level = 1000
ddim_noise_steps = 16
noise_abs_max = 20
enable_torch_compile_model = True
enable_torch_compile_vae = True
# Adjustable context window size
context_window_size = 4  # Adjust this value as needed
n_prompt_frames = 4
offset = 0
scaling_factor = 0.07843137255
# Get input video (first frame as prompt)
video_id = os.environ["STARTING_IMAGE_NAME"]
stabilization_level = 15
screen_width = 1024  # Adjust as needed
screen_height = 1024  # Adjust as needed

# Define ACTION_KEYS
ACTION_KEYS = [
    "inventory",
    "ESC",
    "hotbar.1",
    "hotbar.2",
    "hotbar.3",
    "hotbar.4",
    "hotbar.5",
    "hotbar.6",
    "hotbar.7",
    "hotbar.8",
    "hotbar.9",
    "forward",
    "back",
    "left",
    "right",
    "cameraX",
    "cameraY",
    "jump",
    "sneak",
    "sprint",
    "swapHands",
    "attack",
    "use",
    "pickItem",
    "drop",
]


def clamp_mouse_input(mouse_input: Tuple[int, int]) -> Tuple[float, float]:
    """
    Clamps and normalizes mouse input coordinates.

    Args:
        mouse_input (Tuple[int, int]): A tuple containing mouse x and y coordinates.

    Returns:
        Tuple[float, float]: A tuple containing the clamped and normalized x and y values.

    Raises:
        AssertionError: If the normalized values are out of the expected range.
    """
    max_val = 20
    bin_size = 0.5
    num_buckets = int(max_val / bin_size)  # 40

    x, y = mouse_input

    # Normalize the inputs
    normalized_x = (x - num_buckets) / num_buckets
    normalized_y = (y - num_buckets) / num_buckets

    # Clamp the values to be within [-1, 1]
    clamped_x = max(-1.0, min(1.0, normalized_x))
    clamped_y = max(-1.0, min(1.0, normalized_y))

    # Optional: Assert to ensure values are within the expected range
    assert -1.0 - 1e-3 <= clamped_x <= 1.0 + 1e-3, f"Normalized x must be in [-1, 1], got {clamped_x}"
    assert -1.0 - 1e-3 <= clamped_y <= 1.0 + 1e-3, f"Normalized y must be in [-1, 1], got {clamped_y}"

    return (clamped_x, clamped_y)

server_thread = frontendmanager.start()
server_eloop = frontendmanager.get_event_loop()

latest_grabbed_inputs = []

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
keymap = dotdict({k: k for k in ["K_e","K_ESCAPE","K_1","K_2","K_3","K_4","K_5","K_6","K_7","K_8","K_9","K_w","K_a","K_s","K_d","K_SPACE","K_LSHIFT","K_RSHIFT","K_LCTRL","K_RCTRL","K_q"]})

default_actmap = {
    "inventory": 0,
    "ESC": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
    "forward": 0,
    "back": 0,
    "left": 0,
    "right": 0,
    "camera": (0, 0),
    "jump": 0,
    "sneak": 0,
    "sprint": 0,
    "swapHands": 0,
    "attack": 0,
    "use": 0,
    "pickItem": 0,
    "drop": 0
}

# Helper functions to capture live actions
def get_current_action(mouse_rel):
    action = {}
    if len(latest_grabbed_inputs) < 1:
        return default_actmap
    action = {}
    inputs = latest_grabbed_inputs.pop(0)
    mouse_rel = inputs["mouse_movement"]
    keys = inputs["keys"]
    mouse_buttons = inputs["mouse_buttons"]
    # Map keys to actions
    action["inventory"] = 1 if keys[keymap.K_e] else 0
    action["ESC"] = 1 if keys[keymap.K_ESCAPE] else 0
    action["hotbar.1"] = 1 if keys[keymap.K_1] else 0
    action["hotbar.2"] = 1 if keys[keymap.K_2] else 0
    action["hotbar.3"] = 1 if keys[keymap.K_3] else 0
    action["hotbar.4"] = 1 if keys[keymap.K_4] else 0
    action["hotbar.5"] = 1 if keys[keymap.K_5] else 0
    action["hotbar.6"] = 1 if keys[keymap.K_6] else 0
    action["hotbar.7"] = 1 if keys[keymap.K_7] else 0
    action["hotbar.8"] = 1 if keys[keymap.K_8] else 0
    action["hotbar.9"] = 1 if keys[keymap.K_9] else 0
    action["forward"] = 2 if keys[keymap.K_w] else 0
    action["back"] = 2 if keys[keymap.K_s] else 0
    action["left"] = 2 if keys[keymap.K_a] else 0
    action["right"] = 2 if keys[keymap.K_d] else 0
    action["camera"] = (mouse_rel[1] / 4, mouse_rel[0] / 4)  # tuple (x, y)
    action["jump"] = 1 if keys[keymap.K_SPACE] else 0
    action["sneak"] = 1 if keys[keymap.K_LSHIFT] or keys[keymap.K_RSHIFT] else 0
    action["sprint"] = 1 if keys[keymap.K_LCTRL] or keys[keymap.K_RCTRL] else 0
    action["swapHands"] = 0  # Map to a key if needed
    action["attack"] = 1 if mouse_buttons[0] else 0  # Left mouse button
    action["use"] = 1 if mouse_buttons[2] else 0     # Right mouse button
    action["pickItem"] = 0  # Map to a key if needed
    action["drop"] = 1 if keys[keymap.K_q] else 0
    return action


def action_to_tensor(action):
    actions_one_hot = torch.zeros(len(ACTION_KEYS), device=device)
    for j, action_key in enumerate(ACTION_KEYS):
        if action_key.startswith("camera"):
            if action_key == "cameraX":
                value = action["camera"][0]
            elif action_key == "cameraY":
                value = action["camera"][1]
            else:
                raise ValueError(f"Unknown camera action key: {action_key}")
            # Normalize value to be in [-1, 1]
            max_val = 20
            bin_size = 0.5
            num_buckets = int(max_val / bin_size)
            value = (value) / num_buckets
            value = max(min(value, 1.0), -1.0)
        else:
            value = action.get(action_key, 0)
            value = float(value)
        actions_one_hot[j] = value
    return actions_one_hot

# Load DiT checkpoint
ckpt = torch.load(model_path)
model = DiT_models["DiT-S/2"]()
model.load_state_dict(ckpt, strict=False)
model = model.to(device).half().eval()

# Load VAE checkpoint
vae_ckpt = torch.load(vae_path)
vae = VAE_models["vit-l-20-shallow-encoder"]()
vae.load_state_dict(vae_ckpt)
vae = vae.to(device).half().eval()

noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1).to(device)
ctx_max_noise_idx = ddim_noise_steps // 10 * 3

if enable_torch_compile_model:
    # Optional compilation for performance
    model = torch.compile(model, mode='reduce-overhead')
if enable_torch_compile_vae:
    vae = torch.compile(vae, mode='reduce-overhead')


# mp4_path = '/home/mix/Playground/ComfyUI/output/game_00001.mp4'

mp4_path = f"sample_data/{video_id}.mp4"
video = read_video(mp4_path, pts_unit="sec")[0].float() / 255

video = video[offset:]

# Initialize action list
def reset():
    global x
    global actions_list
    x = encode(video, vae)
    # Initialize with initial action (assumed zero action)
    actions_list = []
    initial_action = torch.zeros(len(ACTION_KEYS), device=device).unsqueeze(0)
    for i in range(n_prompt_frames - 1):
        actions_list.append(initial_action)


@torch.inference_mode
def sample(x, actions_tensor, ddim_noise_steps, stabilization_level, alphas_cumprod, noise_range, noise_abs_max, model):
    """
    Sample function with constant alpha_next and stabilization_level implemented.

    Args:
        x (torch.Tensor): Current latent tensor of shape [B, T, C, H, W].
        actions_tensor (torch.Tensor): Actions tensor of shape [B, T, num_actions].
        ddim_noise_steps (int): Number of DDIM noise steps.
        stabilization_level (int): Level to stabilize the initial frames.
        alphas_cumprod (torch.Tensor): Cumulative product of alphas for each timestep.
        noise_range (torch.Tensor): Noise schedule tensor.
        noise_abs_max (float): Maximum absolute noise value.
        model (torch.nn.Module): The diffusion model.

    Returns:
        torch.Tensor: Updated latent tensor after sampling.
    """
    B, context_length, C, H, W = x.shape

    for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
        # Set up noise values
        t_ctx = torch.full((B, context_length - 1), stabilization_level - 1, dtype=torch.long, device=device)
        t = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
        t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
        t_next = torch.where(t_next < 0, t, t_next)
        t = torch.cat([t_ctx, t], dim=1)
        t_next = torch.cat([t_ctx, t_next], dim=1)

        # Get model predictions
        with autocast("cuda", dtype=torch.half):
            v = model(x, t, actions_tensor)

        # Compute x_start and x_noise
        x_start = alphas_cumprod[t].sqrt() * x - (1 - alphas_cumprod[t]).sqrt() * v
        x_noise = ((1 / alphas_cumprod[t]).sqrt() * x - x_start) / (1 / alphas_cumprod[t] - 1).sqrt()

        # Compute alpha_next with constant values for context frames
        alpha_next = alphas_cumprod[t_next].clone()
        alpha_next[:, :-1] = torch.ones_like(alpha_next[:, :-1])

        # Ensure the last frame has alpha_next set to 1 if it's the first noise step
        if noise_idx == 1:
            alpha_next[:, -1:] = torch.ones_like(alpha_next[:, -1:])

        # Compute the predicted x
        x_pred = alpha_next.sqrt() * x_start + x_noise * (1 - alpha_next).sqrt()

        # Update only the last frame in the latent tensor
        x[:, -1:] = x_pred[:, -1:]

        # Optionally clamp the noise to maintain stability
        x[:, -1:] = torch.clamp(x[:, -1:], -noise_abs_max, noise_abs_max)

    return x


@torch.inference_mode
def encode(video, vae):
    x = video[:n_prompt_frames].unsqueeze(0).to(device)
    # VAE encoding
    x = rearrange(x, "b t h w c -> (b t) c h w").half()
    H, W = x.shape[-2:]
    with torch.no_grad():
        x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H//vae.patch_size, w=W//vae.patch_size)
    return x

@torch.inference_mode
def decode(x, vae):
    # VAE decoding of the last frame
    x_last = x[:, -1:]
    x_last = rearrange(x_last, "b t c h w -> (b t) (h w) c").half()
    with torch.no_grad():
        x_decoded = (vae.decode(x_last / scaling_factor) + 1) / 2
    x_decoded = rearrange(x_decoded, "(b t) c h w -> b t h w c", b=1, t=1)
    x_decoded = torch.clamp(x_decoded, 0, 1)
    x_decoded = (x_decoded * 255).byte().cpu().numpy()
    frame = x_decoded[0, 0]
    pili = Image.fromarray(frame)
    buffer = io.BytesIO()
    pili.save(buffer, format="WEBP", quality=75)
    return buffer.getvalue()

reset()

# Get alphas
betas = sigmoid_beta_schedule(max_noise_level).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

# Initialize variables for FPS measurement
frame_times = []  # List to store timestamps of recent frames
fps = 0.0

# Initialize variable for toggling FPS display
show_fps = True

# Main loop
running = True
mouse_captured = False  # Initially not captured

reset_context = False
while running:
    current_time = time.time()
    if not reset_context:
        # Capture current action
        action = get_current_action()
        actions_curr = action_to_tensor(action).unsqueeze(0)  # Shape [1, num_actions]
        actions_list.append(actions_curr)

        # Generate a random latent for the new frame
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x = torch.cat([x, chunk], dim=1)

        # Implement sliding window for context frames and actions
        if x.shape[1] > context_window_size:
            x = x[:, -context_window_size:]
            actions_list = actions_list[-context_window_size:]
        # Prepare actions tensor
        actions_tensor = torch.stack(actions_list, dim=1)  # Shape [1, context_length, num_actions]
    else:
        reset_context = False
    
    x = sample(x, actions_tensor, ddim_noise_steps, stabilization_level, alphas_cumprod, noise_range, noise_abs_max, model)

    frame = decode(x, vae)

    # --- FPS Counter ---
    # Update frame times
    frame_times.append(current_time)
    # Remove frame times older than 1 second
    while frame_times and frame_times[0] < current_time - 1:
        frame_times.pop(0)
    # Calculate FPS
    fps = len(frame_times)

    asyncio.run_coroutine_threadsafe( frontendmanager.send_news( struct.pack("<H", fps) + frame ), server_eloop )