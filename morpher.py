import subprocess
import cv2
import sys
import os
import numpy as np
from multiprocessing import Pool

from tqdm import tqdm

style_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'stylegan2adapytorch'))
sys.path.append(style_path)

from models.stylegan2adapytorch.projector import run_projection

NO_FRAMES = 10
PROJECTION_INTERMEDIATE_FRAMES = 10
FPS = 30
SYNTH_STEPS = 250
PROJECTION = True
DEBUG = False
IS_FACE = True
IMAGE_SIZE = (512, 512)

q = 3
output_folder = "output"


def interpolate_landmark_positions(landmarks1, landmarks2, t):
    """
    This function interpolates between two sets of landmarks based on a given alpha.
    It calculates the new position of each landmark as a weighted average of the positions in the two sets.

    Args:
        landmarks1 (list): The first set of landmarks. Each landmark is a tuple of (x, y) coordinates.
        landmarks2 (list): The second set of landmarks. Each landmark is a tuple of (x, y) coordinates.
        alpha (float): The weight for the interpolation. A value of 0 will return landmarks1, a value of 1 will return landmarks2.

    Returns:
        list: The interpolated landmarks.
    """
    
    interpolated_landmarks = []

    if len(landmarks1) != len(landmarks2):
        raise ValueError("Both images must have the same number of landmarks.")

    for i in range(len(landmarks1)):
        x_A, y_A = landmarks1[i]
        x_B, y_B = landmarks2[i]
        x_interp = (1 - t) * x_A + t * x_B
        y_interp = (1 - t) * y_A + t * y_B
        interpolated_landmarks.append((x_interp, y_interp))

    return interpolated_landmarks


def bilinear_interpolation(image, x, y):
    """
    This function performs bilinear interpolation for a given x, y coordinate on an image.
    It first calculates the integer and fractional parts of the coordinates.
    Then it gets the values of the four pixels surrounding the coordinate.
    Finally, it calculates the interpolated value by taking a weighted average of these pixels.

    Args:
        image (np.array): The image on which to perform the interpolation.
        x (float): The x-coordinate for the interpolation.
        y (float): The y-coordinate for the interpolation.

    Returns:
        float: The interpolated pixel value.
    """

    # clamp the coordinates to the image size
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    
    if x >= image.shape[1] - 1:
        x = image.shape[1] - 2
    if y >= image.shape[0] - 1:
        y = image.shape[0] - 2
    
    x1 = int(x)
    x2 = x1 + 1
    y1 = int(y)
    y2 = y1 + 1
    dx = x - x1
    dy = y - y1

    R1, G1, B1 = image[y1, x1] 
    R2, G2, B2 = image[y1, x2]
    R3, G3, B3 = image[y2, x1]
    R4, G4, B4 = image[y2, x2]

    # the bilinear interpolation formula
    R_interp = (1 - dx) * (1 - dy) * R1 + dx * (1 - dy) * R2 + (1 - dx) * dy * R3 + dx * dy * R4
    G_interp = (1 - dx) * (1 - dy) * G1 + dx * (1 - dy) * G2 + (1 - dx) * dy * G3 + dx * dy * G4
    B_interp = (1 - dx) * (1 - dy) * B1 + dx * (1 - dy) * B2 + (1 - dx) * dy * B3 + dx * dy * B4

    return (R_interp, G_interp, B_interp)


def interpolate_color(image1, image2, x1, y1, x2, y2, t):
    """
    This function interpolates the colors of two images at given coordinates based on a given t value.
    It first performs bilinear interpolation on both images to get the colors at the coordinates.
    Then it calculates the final color as a weighted average of the two colors.

    Args:
        image1 (np.array): The first image.
        image2 (np.array): The second image.
        x1 (float): The x-coordinate for the interpolation on the first image.
        y1 (float): The y-coordinate for the interpolation on the first image.
        x2 (float): The x-coordinate for the interpolation on the second image.
        y2 (float): The y-coordinate for the interpolation on the second image.
        t (float): The weight for the interpolation. 

    Returns:
        tuple: The interpolated color.
    """
    color_A = bilinear_interpolation(image1, x1, y1)
    color_B = bilinear_interpolation(image2, x2, y2)

    final_color = (
        (1 - t) * color_A[0] + t * color_B[0],
        (1 - t) * color_A[1] + t * color_B[1],
        (1 - t) * color_A[2] + t * color_B[2]
    )

    return final_color


def relative_shepard_interpolation(vectors, landmarks, x, y, q):
    """
    This function performs relative Shepard interpolation for a given x, y coordinate based on a set of vectors and landmarks.
    It calculates the weights for each vector based on its corresponding landmark's distance to the coordinate.
    Then it calculates the interpolated coordinate as a weighted average of the vectors, adjusted by the original coordinate.

    Args:
        vectors (list): The vectors for the interpolation. Each vector is a tuple of (x, y) values.
        landmarks (list): The landmarks corresponding to the vectors. Each landmark is a tuple of (x, y) coordinates.
        x (float): The x-coordinate for the interpolation.
        y (float): The y-coordinate for the interpolation.
        q (float): The power parameter for the Shepard interpolation.

    Returns:
        tuple: The interpolated (x, y) coordinate.
    """
    weighted_sum_x = 0.0
    weighted_sum_y = 0.0
    total_weight = 0.0

    for i in range(len(landmarks)):
        x_i, y_i = landmarks[i]
        distance = ((x_i - x) ** 2 + (y_i - y) ** 2) ** (1 / 2)
        # if the distance is 0, the weight will be infinite, so we can just return the landmark
        if distance == 0:
            return vectors[i][0] + x, vectors[i][1] + y
        weight_i = 1.0 / (distance ** q)
        weighted_sum_x += weight_i * vectors[i][0]
        weighted_sum_y += weight_i * vectors[i][1]
        total_weight += weight_i

    x_interp = weighted_sum_x / total_weight + x
    y_interp = weighted_sum_y / total_weight + y

    return x_interp, y_interp


def create_morphed_frame(image1, image2, landmarks1, landmarks2, frame, total_frames, q):
    """
    This function creates a morphed frame by interpolating between two images and their landmarks.
    It first calculates the vectors between the corresponding landmarks in the two images.
    Then it creates an empty image and fills each pixel by performing relative Shepard interpolation on the vectors and landmarks.

    Args:
        image1 (np.array): The first image.
        image2 (np.array): The second image.
        landmarks1 (list): The landmarks for the first image. Each landmark is a tuple of (x, y) coordinates.
        landmarks2 (list): The landmarks for the second image. Each landmark is a tuple of (x, y) coordinates.
        frame (float): The index of the frame to generate.
        total_frames (float): The total number of frames to generate. 
        q (float): The power parameter for the Shepard interpolation.

    Returns:
        np.array: The morphed frame.
    """
    # threading might induce compiling of the function that makes the global varibles to not respond to updates in the main thread
    # that's why we need to pass the global variables as arguments to the function, will be fixed in the future
    t = frame / (total_frames - 1)

    # compute the frame's interpolated landmarks and the offset vectors
    interpolated_landmarks = interpolate_landmark_positions(landmarks1, landmarks2, t)
    morphed_frame = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)

    vectors1 = [(landmarks1[i][0] - interpolated_landmarks[i][0], landmarks1[i][1] - interpolated_landmarks[i][1]) for i in range(len(landmarks1))]
    vectors2 = [(landmarks2[i][0] - interpolated_landmarks[i][0], landmarks2[i][1] - interpolated_landmarks[i][1]) for i in range(len(landmarks2))]

    for x in range(image1.shape[1]):    
        for y in range(image1.shape[0]):
            # find the pixel's position in the position space determined by the shepard interpolation on landmarks
            x_prime_1, y_prime_1 = relative_shepard_interpolation(vectors1, landmarks1, x, y, q)
            x_prime_2, y_prime_2 = relative_shepard_interpolation(vectors2, landmarks2, x, y, q)
            color = interpolate_color(image1, image2, x_prime_1, y_prime_1, x_prime_2, y_prime_2, t)
            morphed_frame[y, x] = color

    print(f"Frame {round(t * (total_frames - 1))} processed")

    return morphed_frame


def remove_background(image, landmarks):
    """
    This function removes the background from an image based on the convex hull of the landmarks.
    It first calculates the convex hull of the landmarks, then creates a mask and draws the hull on it.
    The mask is then applied to the image to remove the background.

    Args:
        image (np.array): The original image.
        landmarks (list): A list of tuples representing the x and y coordinates of the landmarks.

    Returns:
        np.array: The image with the background removed.
    """
    landmarks = np.array(landmarks, dtype=np.int32)
    hull = cv2.convexHull(landmarks)

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], (255, 255, 255))

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, mask)

    return image


def project_frame(i):
    """
    This function projects a frame onto the latent space of a StyleGAN model.
    It first loads the frame and preprocesses it, then uses the StyleGAN encoder to project the frame.

    Args:
        i (int): The index of the frame to project.

    Returns:
        np.array: The latent vector representing the frame in the StyleGAN latent space.
    """
    global output_folder, SYNTH_STEPS

    # The actual implementation will depend on your specific code.
    subfolder = os.path.join(output_folder, f"frame_{i}")
    frame_path = os.path.join(subfolder, f"frame_{i}.jpg")
    run_projection(frame_path, subfolder, seed=303, num_steps=SYNTH_STEPS)
    print(f"Frame {i} projected!")
    return cv2.imread(os.path.join(subfolder, f"proj00.png"))


def generate_morphing_animation(image1, image2, landmarks1, landmarks2, remove_bckg, args):    
    """
    This function generates a morphing animation between two images based on their landmarks.
    It first removes the background from the images if requested.
    Then it creates a series of morphed frames by interpolating between the images and their landmarks.
    Finally, it saves the frames as a video.

    Args:
        image1 (np.array): The first image.
        image2 (np.array): The second image.
        landmarks1 (list): The landmarks for the first image. Each landmark is a tuple of (x, y) coordinates.
        landmarks2 (list): The landmarks for the second image. Each landmark is a tuple of (x, y) coordinates.
        remove_bckg (bool): Whether to remove the background from the images.
        args (argparse.Namespace): The command-line arguments.

    Returns:
        None
    """
    # setup
    global NO_FRAMES, FPS, SYNTH_STEPS, PROJECTION, q, output_folder, DEBUG, IS_FACE, PROJECTION_INTERMEDIATE_FRAMES, IMAGE_SIZE
    NO_FRAMES, FPS, SYNTH_STEPS, PROJECTION, q, DEBUG, IS_FACE, PROJECTION_INTERMEDIATE_FRAMES = \
        args.no_frames, args.fps, args.synth_steps, args.gan_projection, args.q, args.debug, args.is_face, args.projection_frames
    
    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)
    IMAGE_SIZE = (image1.shape[0], image1.shape[1])    

    print("Morphing animation generation started!")
    
    # see comment in create_morphed_frame, error with global variables, that's why we introduce a redundant variable
    total_frames = NO_FRAMES

    # morphing animation generation with multiprocessing
    arguments = [(image1, image2, landmarks1, landmarks2, frame, total_frames, q) for frame in range(total_frames)]

    with Pool(10) as pool:
        morphed_frames = list(tqdm(pool.starmap(create_morphed_frame, arguments), total=total_frames))

    for frame in range(total_frames):
        subfolder = os.path.join(output_folder, f"frame_{frame}")
        os.makedirs(subfolder, exist_ok=True)
        cv2.imwrite(os.path.join(subfolder, f"frame_{frame}.jpg"), morphed_frames[frame])
        sys.stdout.write(f"\r{frame / (NO_FRAMES - 1) * 100:.2f}% complete")

    # if the user wants to remove the background, we do it here    
    if remove_bckg:
        for i, frame in enumerate(morphed_frames):
            t = i / (total_frames - 1)
            morphed_frames[i] = remove_background(frame, interpolate_landmark_positions(landmarks1, landmarks2, t))

    # debug code
    if DEBUG:
        # landmark interpolation debug
        os.makedirs(os.path.join(output_folder, "interpolation"), exist_ok=True)
        for i, frame in enumerate(morphed_frames):
            landmarks_in = interpolate_landmark_positions(landmarks1, landmarks2, i / (total_frames - 1))
            frame_copy = frame.copy()
            for j in range(len(landmarks_in)):
                x, y = landmarks_in[j]
                x, y = int(x), int(y)
                xx, yy = landmarks2[j]
                xx, yy = int(xx), int(yy)
                cv2.arrowedLine(frame_copy, (x, y), (xx, yy), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(output_folder, "interpolation", f"debug_frame_{i}.jpg"), frame_copy)

        # background removal debug
        os.makedirs(os.path.join(output_folder, "background"), exist_ok=True)
        for i, frame in enumerate(morphed_frames):
            t = i / (total_frames - 1)
            dframe = remove_background(frame, interpolate_landmark_positions(landmarks1, landmarks2, t))
            cv2.imwrite(os.path.join(output_folder, "background", f"nobackground_frame_{i}.jpg"), dframe)

    # save the frames as a video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_folder, "morphing_animation.mp4"), fourcc, FPS / PROJECTION_INTERMEDIATE_FRAMES, (image1.shape[0], image1.shape[1]))

    for frame in range(total_frames):
        out.write(morphed_frames[frame])   
    
    out.release()

    # projection animation generation, only if the user wants it and the images are faces, projection for other things not yet supported
    if PROJECTION and IS_FACE:
        projected_frames = []

        print("Projection animation generation started!")
 
        projected_frames = [project_frame(i) for i in range(total_frames)]

        final_frames = interpolate_final_frames(projected_frames, PROJECTION_INTERMEDIATE_FRAMES)

        # save the frames as a video using OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(output_folder, "projection_animationnn.mp4"), fourcc, FPS, (projected_frames[0].shape[0], projected_frames[0].shape[1]))

        for frame in range(len(final_frames)):
            out.write(final_frames[frame])
        
        out.release()


def interpolate_final_frames(frames, number):
    """
    This function interpolates between a series of frames to create a smooth animation.
    It generates a number of intermediate frames between each pair of consecutive frames in the input list.

    Args:
        frames (list): The original frames. Each frame is a numpy array representing an image.
        number (int): The number of intermediate frames to generate between each pair of original frames.

    Returns:
        list: The list of original and interpolated frames.
    """
    interpolated_frames = []
    for i in range(len(frames) - 1):
        interpolated_frames.append(frames[i])
        for j in range(number):
            t = j / (number - 1)
            interpolated_frames.append(cv2.addWeighted(frames[i], 1 - t, frames[i + 1], t, 0))
    interpolated_frames.append(frames[-1])
    return interpolated_frames


def crop_and_resize_image(frame_path):
    """
    This function crops and resizes an image.
    It first reads the image from the given path, then calculates the size of the square crop based on the smaller dimension of the image.
    The image is then cropped to a square and resized to 1024x1024 pixels.
    Finally, the modified image is saved back to the original path.

    Args:
        frame_path (str): The path to the image file.

    Returns:
        None
    """
    image = cv2.imread(frame_path)
    height, width = image.shape[:2]
    size = min(height, width)

    startx = width//2 - size//2
    starty = height//2 - size//2

    image = image[starty:starty+size, startx:startx+size]
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_AREA)

    cv2.imwrite(frame_path, image)


# Credits: StyleGAN2-ada-pytorch repository
# https://github.com/NVlabs/stylegan2-ada-pytorch
# arxiv.org/abs/2006.06676
def run_projection(frame_path, outdir, seed=303, num_steps=1):
    """
    This function projects an image onto the latent space of a StyleGAN model.
    It first loads the image from the given path, then uses the StyleGAN encoder to project the image.
    The resulting latent vector is saved to the specified output directory.

    Args:
        frame_path (str): The path to the image file.
        outdir (str): The directory where the output latent vector should be saved.
        seed (int, optional): The random seed for the StyleGAN encoder. Defaults to 303.
        num_steps (int, optional): The number of steps for the projection. Defaults to 1.

    Returns:
        None
    """
    # we resize the image to 1024x1024 because the StyleGAN model performs better with this size (it's the size it was trained on)
    crop_and_resize_image(frame_path)
    subprocess.run([
        "python", 
        "models/stylegan2adapytorch/projector.py", 
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl", 
        "--target=" + frame_path, 
        "--outdir=" + outdir,
        "--save-video=" + "False", 
        "--seed=" + str(seed), 
        "--num-steps=" + str(num_steps)	
    ])
    subprocess.run([
        "python", 
        "models/stylegan2adapytorch/generate.py", 
        "--outdir=" + outdir, 
        "--projected-w=" + os.path.join(outdir, "projected_w.npz"), 
        "--network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    ])
    # resize back to the original size
    image = cv2.imread(os.path.join(outdir, "proj00.png"))
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(outdir, "proj00.png"), image)