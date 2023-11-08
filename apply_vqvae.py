from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from torchvision.io import read_video, read_video_timestamps
import sys
sys.path.append(r'C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\VideoGPT-master')
from videogpt import download, VQVAE, load_videogpt
from videogpt.data import preprocess

# Load the VQ-VAE
filepath = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\VideoGPT-master\lightning_logs\version_2\checkpoints\epoch=43-step=136135.ckpt"
device = torch.device('cuda')
vqvae = VQVAE.load_from_checkpoint(filepath).to(device)

# Load the video
# `resolution` must be divisible by the encoder image stride
# `sequence_length` must be divisible by the encoder temporal stride
resolution, sequence_length = vqvae.args.resolution, 16

video_filename = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\gesture_sorted_data_resized\test\EFL\102_18_0_9_1_chair.mp4"
pts = read_video_timestamps(video_filename, pts_unit='sec')[0]
video = read_video(video_filename, pts_unit='sec', start_pts=pts[0], end_pts=pts[sequence_length - 1])[0]
video = preprocess(video, resolution, sequence_length).unsqueeze(0).to(device)
print(video.shape)

# Apply the VQ-VAE to get an encoding and reconstruct the video
with torch.no_grad():
    encodings = vqvae.encode(video)
    video_recon = vqvae.decode(encodings)
    video_recon = torch.clamp(video_recon, -0.5, 0.5)

# Visualize the reconstruction
videos = torch.cat((video, video_recon), dim=-1)
videos = videos[0].permute(1, 2, 3, 0)  # CTHW -> THWC
videos = ((videos + 0.5) * 255).cpu().numpy().astype('uint8')

fig = plt.figure()
plt.title('real (left), reconstruction (right)')
plt.axis('off')

for this_frame in videos:
    plt.imshow(this_frame)
    plt.show()
