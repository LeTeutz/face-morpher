import argparse
import warnings

warnings.filterwarnings("ignore", message="A NumPy version >=1.17.3 and <1.25.0")
warnings.filterwarnings("ignore", message="Default grid_sample")

import os
from UI import generate_UI


# The path renaming is needed to avoid an error in which Python cannot detect the stylegan2-ada-pytorch modules
def rename_path():
    style_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'stylegan2-ada-pytorch'))
    style_path_new = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'stylegan2adapytorch'))
    if os.path.exists(style_path) and not os.path.exists(style_path_new):
        os.rename(style_path, style_path_new)
    
    spiga_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'SPIGA'))
    spiga_path_new = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'spiga'))
    os.rename(spiga_path, spiga_path_new)

if "__main__" == __name__:
    parser = argparse.ArgumentParser(description='Arguments for the project.')
    parser.add_argument('--no_frames', type=int, default=10,
                        help='Number of frames for the morphing animation')
    parser.add_argument('--image_size', type=int, default=512, 
                        help='The size of the (squared) images that the application will use')
    parser.add_argument('--landmarks_path', type=str, default='landmarks.txt',
                        help='Path to the landmarks file, if you want to use already known/computed landmarks')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second in the final videos')
    parser.add_argument('--synth_steps', type=int, default=250,
                        help='Number of training steps for the final projection')
    parser.add_argument('--output_folder', type=str, default='output', 
                        help='The folder where the output files will be saved')
    parser.add_argument('--debug', type=bool, default=False,
                        help="Debug mode")
    parser.add_argument('--projection_frames', type=int, default=10, 
                        help='Number of intermidiary frames between 2 projected frames in the final video.')

    args = parser.parse_args()

    rename_path()
    generate_UI(args)