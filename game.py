import time
from typing import Tuple

import torch
import torch._dynamo
import websockets.extensions
import websockets.extensions.permessage_deflate
torch._dynamo.config.suppress_errors = True

from dit import DiT_models
from vae import VAE_models
from torchvision.io import read_video
from utils import sigmoid_beta_schedule
from einops import rearrange
from torch import autocast
from safetensors.torch import load_file

from aiohttp import web
import threading
import numpy as np
import json
import asyncio
import ffmpeg
import os
import frontend
import websockets

assert torch.cuda.is_available()
device = "cuda:0"

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

latest_grabbed_inputs = []

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
keymap = dotdict({k: k for k in ["K_e","K_ESCAPE","K_1","K_2","K_3","K_4","K_5","K_6","K_7","K_8","K_9","K_w","K_a","K_s","K_d","K_SPACE","K_LSHIFT","K_RSHIFT","K_LCTRL","K_RCTRL","K_q"]})

def get_current_action(mouse_rel):
    action = {}
    inputs = latest_grabbed_inputs.pop(0)
    keys = inputs.keys
    mouse_buttons = inputs.mbuttons
    clamped_input = clamp_mouse_input(mouse_rel)
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


async def new_client(websocket, path):
    print(f"New client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            await message_received(websocket, message)
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")

async def message_received(client, message):
    global latest_grabbed_inputs
    latest_grabbed_inputs.append(json.loads(message)["input"])

start_server = websockets.serve(new_client, "127.0.0.1", 17890, extensions=[
    websockets.extensions.permessage_deflate.ServerPerMessageDeflateFactory(
        server_max_window_bits=13,
        client_max_window_bits=13,
        compress_settings={"memLevel": 7}
    )
])

async def main_server():
    async with start_server:
        await asyncio.Future()
def server():
    asyncio.run(main_server())

websocket_thread = threading.Thread(target=server)
websocket_thread.daemon = True
websocket_thread.start()
frontend_thread = frontend.start()

# Load DiT checkpoint
model = DiT_models["DiT-S/2"]()
model_ckpt = load_file("oasis500m.safetensors")
model.load_state_dict(model_ckpt, strict=False)
model = model.to(device).half().eval()

print("loading vae")

# Load VAE checkpoint
vae = VAE_models["vit-l-20-shallow-encoder"]()
vae_ckpt = load_file("vit-l-20.safetensors")
vae.load_state_dict(vae_ckpt)
vae = vae.to(device).half().eval()


# Sampling params
B = 1
max_noise_level = 1000
ddim_noise_steps = 16
noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1).to(device)
noise_abs_max = 20
ctx_max_noise_idx = ddim_noise_steps // 10 * 3
enable_torch_compile_model = True
enable_torch_compile_vae = True

if enable_torch_compile_model:
    # Optional compilation for performance
    model = torch.compile(model, mode='reduce-overhead')
if enable_torch_compile_vae:
    vae = torch.compile(vae, mode='reduce-overhead')

context_window_size = 4

video_id = os.environ["STARTING_IMAGE_NAME"]

mp4_path = f"sample_data/{video_id}.mp4"
video = read_video(mp4_path, pts_unit="sec")[0].float() / 255

offset = 0
video = video[offset:]
n_prompt_frames = 4
scaling_factor = 0.07843137255
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
def sample(x, actions_tensor, ddim_noise_steps, ctx_max_noise_idx, model):
    # Prepare time steps
    context_length = x.shape[1]
    for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
        # Set up noise values
        ctx_noise_idx = min(noise_idx, ctx_max_noise_idx)
        t_ctx = torch.full((B, context_length - 1), noise_range[ctx_noise_idx], dtype=torch.long, device=device)
        t_last = torch.full((B, 1), noise_range[noise_idx], dtype=torch.long, device=device)
        t = torch.cat([t_ctx, t_last], dim=1)
        t_next = torch.cat([t_ctx, torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)],
                           dim=1)
        t_next = torch.where(t_next < 0, t, t_next)

        # Add noise to context frames (except the last frame)
        x_curr = x.clone()
        if context_length > 1:
            ctx_noise = torch.randn_like(x_curr[:, :-1])
            ctx_noise = torch.clamp(ctx_noise, -noise_abs_max, +noise_abs_max)
            x_curr[:, :-1] = alphas_cumprod[t[:, :-1]].sqrt() * x_curr[:, :-1] + \
                             (1 - alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise

        # Get model predictions
        with autocast("cuda", dtype=torch.half):
            v = model(x_curr, t, actions_tensor)

        x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
        x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) / \
                  (1 / alphas_cumprod[t] - 1).sqrt()

        # Get frame prediction
        x_pred = alphas_cumprod[t_next].sqrt() * x_start + x_noise * (1 - alphas_cumprod[t_next]).sqrt()
        x[:, -1:] = x_pred[:, -1:]
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
    return frame


reset()

# Get alphas
betas = sigmoid_beta_schedule(max_noise_level).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

# Initialize variables for FPS measurement
frame_times = []  # List to store timestamps of recent frames
fps = 0.0

# Initialize variables for displaying adjustment info
adjustment_message = ""
adjustment_display_time = 0  # Time when the message should stop displaying

# Initialize variable for toggling FPS display
show_fps = True

# Main loop
running = True
mouse_captured = False  # Initially not captured

reset_context = False
while running:
    current_time = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == keymap.KEYDOWN:
            if event.key == keymap.K_F2:
                if mouse_captured:
                    # Release the mouse
                    pygame.mouse.set_visible(True)
                    pygame.event.set_grab(False)
                    mouse_captured = False
                    print("Mouse released.")
                else:
                    # Capture the mouse
                    pygame.mouse.set_visible(False)
                    pygame.event.set_grab(True)
                    mouse_captured = True
                    pygame.mouse.set_pos(center_pos)  # Reset to center
                    pygame.mouse.get_rel()  # Reset relative movement
                    print("Mouse captured.")

            elif event.key == keymap.K_F3:
                # Toggle FPS display
                show_fps = not show_fps
                print(f"FPS display toggled to {'ON' if show_fps else 'OFF'}.")
            elif event.key == keymap.K_F4:
                # Reset Context
                reset()
                reset_context = True

            # Handle '+' and '-' key presses to adjust ddim_noise_steps
            elif event.key in [keymap.K_PLUS, keymap.K_EQUALS]:
                ddim_noise_steps += 1
                if ddim_noise_steps > 100:  # Set an upper limit if desired
                    ddim_noise_steps = 100
                # Update noise_range and ctx_max_noise_idx
                noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1).to(device)
                ctx_max_noise_idx = ddim_noise_steps // 10 * 3
                adjustment_message = f"ddim_noise_steps: {ddim_noise_steps}"
                adjustment_display_time = current_time + 2  # Display for 2 seconds
                print(adjustment_message)

            elif event.key in [keymap.K_MINUS, keymap.K_UNDERSCORE]:
                ddim_noise_steps -= 1
                if ddim_noise_steps < 1:
                    ddim_noise_steps = 1
                # Update noise_range and ctx_max_noise_idx
                noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1).to(device)
                ctx_max_noise_idx = ddim_noise_steps // 10 * 3
                adjustment_message = f"ddim_noise_steps: {ddim_noise_steps}"
                adjustment_display_time = current_time + 2  # Display for 2 seconds
                print(adjustment_message)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if not mouse_captured:
                # Capture the mouse on mouse click if it's not already captured
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
                mouse_captured = True
                pygame.mouse.set_pos(center_pos)  # Reset to center
                pygame.mouse.get_rel()  # Reset relative movement
                print("Mouse captured on click.")

    if mouse_captured:
        # Get relative mouse movement
        rel = pygame.mouse.get_rel()
        relative_mouse_movement = rel

        # Reset mouse position to the center
        pygame.mouse.set_pos(center_pos)
    else:
        relative_mouse_movement = (0, 0)
    if not reset_context:
        # Capture current action
        action = get_current_action(relative_mouse_movement)
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
    x = sample(x, actions_tensor, ddim_noise_steps, ctx_max_noise_idx, model)

    frame = decode(x, vae)

    # Convert to surface and display
    frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
    frame_surface = pygame.transform.scale(frame_surface, (screen_width, screen_height))
    screen.blit(frame_surface, (0, 0))

    # --- FPS Counter ---
    # Update frame times
    frame_times.append(current_time)
    # Remove frame times older than 1 second
    while frame_times and frame_times[0] < current_time - 1:
        frame_times.pop(0)
    # Calculate FPS
    fps = len(frame_times)

    if show_fps:
        fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))  # White color
        fps_rect = fps_text.get_rect(topright=(screen_width - 10, 10))  # 10 pixels padding from top-right
        screen.blit(fps_text, fps_rect)
    # -------------------

    # --- Adjustment Info Display ---
    if adjustment_message and current_time < adjustment_display_time:
        adjustment_text = font.render(adjustment_message, True, (255, 255, 0))  # Yellow color
        adjustment_rect = adjustment_text.get_rect(center=(screen_width // 2, 30))  # Top center
        screen.blit(adjustment_text, adjustment_rect)
    elif current_time >= adjustment_display_time:
        adjustment_message = ""  # Clear the message
    # ---------------------------------

    pygame.display.flip()

    # Control frame rate
    clock.tick(35)  # Adjust FPS as needed

pygame.quit()
