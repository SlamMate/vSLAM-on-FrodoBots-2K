import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from networks import LiteMono, DepthDecoder
from layers import disp_to_depth
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Video depth prediction using Lite-Mono.')
    parser.add_argument('--video_path', type=str, required=True, help='path to the input video')
    parser.add_argument('--output_path', type=str, required=True, help='path to the output video')
    parser.add_argument('--load_weights_folder', type=str, required=True, help='path to pretrained model weights')
    parser.add_argument('--model', type=str, default='lite-mono', choices=["lite-mono", "lite-mono-small", "lite-mono-tiny", "lite-mono-8m"], help='model type')
    parser.add_argument('--no_cuda', action='store_true', help='if set, disables CUDA')
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    print("-> Loading model from ", args.load_weights_folder)
    encoder_path = os.path.join(args.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(args.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path, map_location=device)
    decoder_dict = torch.load(decoder_path, map_location=device)

    feed_height = 320
    feed_width = 1024

    encoder = LiteMono(model=args.model, height=feed_height, width=feed_width)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    encoder.to(device)
    encoder.eval()

    depth_decoder = DepthDecoder(encoder.num_ch_enc, scales=range(3))
    depth_decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in depth_decoder.state_dict()})
    depth_decoder.to(device)
    depth_decoder.eval()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is a common codec for mp4 files
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))

    transform = transforms.ToTensor()

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to model input size
            input_image = pil.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transform(input_image).unsqueeze(0).to(device)

            # Model prediction
            features = encoder(input_image)
            outputs = depth_decoder(features)
            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (height, width), mode="bilinear", align_corners=False)
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            colormapped_im = cv2.cvtColor(colormapped_im, cv2.COLOR_RGB2BGR)
            
            # Write frame to output video
            out.write(colormapped_im)

    cap.release()
    out.release()
    print('-> Done!')

if __name__ == '__main__':
    main()

