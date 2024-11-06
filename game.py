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
import ctypes
import struct

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

def get_current_action():
    if len(latest_grabbed_inputs) < 1:
        return default_actmap
    action = {}
    inputs = latest_grabbed_inputs.pop(0)
    mouse_rel = inputs["mouse_movement"]
    keys = inputs["keys"]
    mouse_buttons = inputs["mouse_buttons"]
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

clients_connected = set()
lock = threading.Lock()

def message_received(message):
    with lock:
        global latest_grabbed_inputs
        latest_grabbed_inputs.append(json.loads(message)["input"])

async def frontend_handler(request):
    return web.FileResponse(f"frontend_static/frontend.html")

async def ws_handler(request):
    ws = web.WebSocketResponse()
    ws.received_fframe = False
    global clients_connected
    await ws.prepare(request)
    with lock:
        clients_connected.add(ws)
    async for msg in ws:
        if msg.type == web.WSMsgType.TEXT:
            message_received(msg.data)
    with lock:
        clients_connected.remove(ws)
    return ws

async def send_news(data):
    for ws in clients_connected:
            if not ws.closed and ws.received_fframe == False:
                await ws.send_bytes(data)
                ws.received_fframe = True

async def send_deltas(data):
    for ws in clients_connected:
            if not ws.closed and ws.received_fframe == True:
                await ws.send_bytes(data)

server_eloop: asyncio.AbstractEventLoop = None

def start_server():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    global server_eloop
    server_eloop = loop

    app = web.Application()
    app.router.add_get("/", frontend_handler)
    app.router.add_get("/stream", ws_handler)
    app.add_routes([web.static("/s", "frontend_static")])
    handler = app.make_handler()

    server = loop.create_server(handler, host="127.0.0.1", port=17890)
    loop.run_until_complete(server)
    loop.run_forever()

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

print("load DiT")

# Load DiT checkpoint
model = DiT_models["DiT-S/2"]()
model_ckpt = load_file("oasis500m.safetensors")
model.load_state_dict(model_ckpt, strict=False)
model = model.to(device).half().eval()

print("loading ViT (VAE)")

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
video = read_video(mp4_path, pts_unit="sec")[0].float().div_(255)
print(video.shape)
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
    H, W, C = frame.shape
    alpha_channel = 255 * np.ones((H, W, 1), dtype=np.uint8)
    frame_rgba = np.concatenate((frame, alpha_channel), axis=2)
    frame_rgba_1d = frame_rgba.flatten()
    return (frame_rgba_1d, W, H)


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

last_ft = 0

send_delta = True
prev_frame = None

reset_context = False
while running:
    current_time = time.time()
    if last_ft == 0:
        last_ft = current_time-0.1 # so u dont accidentally divide by 0
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

    x = sample(x, actions_tensor, ddim_noise_steps, ctx_max_noise_idx, model)

    frame = decode(x, vae)

    # --- FPS Counter ---
    fps = int( 1 / (current_time - last_ft) )
    # -------------------

    last_ft = current_time

    print(f"FPS is {fps}, current frame pixel count is {len(frame[0]) / 4}")

    # format: fps[short], width[short], isDelta[bool_as_byte], payload[???]
    if prev_frame:
        delta = np.bitwise_xor(prev_frame, frame)
        asyncio.run_coroutine_threadsafe( send_deltas( struct.pack("<HH?", fps, frame[1], True) + bytes(delta) ), server_eloop )
        asyncio.run_coroutine_threadsafe( send_news( struct.pack("<HH?", fps, frame[1], False) + bytes(frame[0]) ), server_eloop )
    else:
        asyncio.run_coroutine_threadsafe( send_news( struct.pack("<HH?", fps, frame[1], False) + bytes(frame[0]) ), server_eloop )

    send_delta = False
    prev_frame = frame