from tkinter import BooleanVar, Tk, Label, Button, Frame, StringVar, ttk, filedialog
import cv2
from PIL import ImageTk, Image
import numpy as np

from landmarks import automatic_facial_landmarks, densify_landmarks
from morpher import generate_morphing_animation

# Global list to store the ImageTk.PhotoImage objects
photo_images = []

landmarks1, landmarks2 = [], []
colors_landmarks = [[], []]
dragging_landmark = None
selection_range = 8

TKINTER_WINDOW_SIZE = 256
LANDMARKS_PATH = "landmarks.txt"
NO_FRAMES = 10
FPS = 30
SYNTH_STEPS = 250
PROJECTION = True
DEBUG = False


def select_image():
    """
    This function opens a file dialog and allows the user to select an image file.
    It supports jpg, jpeg, and png file formats.

    Returns:
        str: The file path of the selected image.
    """
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    return file_path


def resize_and_crop(image_path, size):
    """
    This function resizes and crops an image to a square shape with the given size.
    It first creates a new square image with the maximum dimension of the original image.
    Then it calculates the padding sizes and pastes the original image onto the center of the new image.
    The border is filled with the color of the closest pixel.

    Args:
        image_path (str): The file path of the image to be resized and cropped.
        size (tuple): The desired size of the output image.

    Returns:
        Image: The resized and cropped image.
    """
    img = Image.open(image_path)
    max_dim = max(img.size)
    new_img = Image.new("RGB", (max_dim, max_dim))

    # Calculate the padding sizes
    left = (max_dim - img.size[0]) // 2
    top = (max_dim - img.size[1]) // 2

    # Paste the image onto the center of the new image
    new_img.paste(img, (left, top))

    # Fill the border with the color of the closest pixel
    for x in range(left):
        for y in range(new_img.height):
            new_img.putpixel((x, y), new_img.getpixel((left, y)))
    for x in range(left + img.width, new_img.width):
        for y in range(new_img.height):
            new_img.putpixel((x, y), new_img.getpixel((left + img.width - 1, y)))
    for y in range(top):
        for x in range(new_img.width):
            new_img.putpixel((x, y), new_img.getpixel((x, top)))
    for y in range(top + img.height, new_img.height):
        for x in range(new_img.width):
            new_img.putpixel((x, y), new_img.getpixel((x, top + img.height - 1)))

    new_img = new_img.resize((size, size))
    return new_img


def display_image_with_landmarks(image, landmarks, label, index):
    """
    This function displays an image with landmarks on a Tkinter label.
    It first creates a copy of the image, then draws circles at the landmark positions.
    The image is then converted from BGR to RGB and displayed on the label.

    Args:
        image (np.array): The original image.
        landmarks (list): A list of tuples representing the x and y coordinates of the landmarks.
        label (Tkinter.Label): The label on which the image is displayed.
        index (int): The index of the current image.

    """
    image_copy = image.copy()

    # size of the circles of the landmarks depends on the size of the image
    scale_factor = max(1, TKINTER_WINDOW_SIZE // 128)

    for i, (x, y) in enumerate(landmarks):
        cv2.circle(image_copy, (int(x), int(y)), scale_factor, colors_landmarks[index][i], -1)
    
    image_cv = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    image_with_landmarks = Image.fromarray(image_cv)
    photo_images.append(ImageTk.PhotoImage(image_with_landmarks))
    # TKinter trickery
    label.config(image=photo_images[-1])


def handle_landmark(event, image, landmarks, label, selected_button, index):
    """
    This function handles the event of a landmark being clicked or dragged in the image.
    It first checks if the "Add" button is selected. If it is, a new landmark is added at the cursor position.
    If the "Move" button is selected, it checks if a landmark is being dragged.
    If a landmark is being dragged, it updates the position of the landmark to the cursor position.

    Args:
        event (Tkinter.Event): The event that triggered the function.
        image (np.array): The original image.
        landmarks (list): A list of tuples representing the x and y coordinates of the landmarks.
        label (Tkinter.Label): The label on which the image is displayed.
        selected_button (Tkinter.StringVar): The currently selected button ("Add" or "Move").
        index (int): The index of the current image.

    """  
    global dragging_landmark
    x, y = event.x, event.y

    if selected_button.get() == "Add":
        landmarks.append((x, y))
        # if there is a color for the corresponding landmark already, use it
        # otherwise generate a new random color and add it to the list
        if len(colors_landmarks[index]) < len(colors_landmarks[1 - index]):
            colors_landmarks[index].append(colors_landmarks[1 - index][len(colors_landmarks[index])])
        else:
            colors_landmarks[index].extend(generate_random_colors(1))
            
        display_image_with_landmarks(image, landmarks, label, index)
    elif selected_button.get() == "Remove":
        for landmark in landmarks:
            lx, ly = landmark
            # if the cursor is close enough to a landmark, remove it
            if abs(lx - x) < selection_range and abs(ly - y) < selection_range:
                idx = landmarks.index(landmark)
                landmarks.remove(landmark)
                colors_landmarks[index].pop(idx)
                display_image_with_landmarks(image, landmarks, label, index)
                break
    elif selected_button.get() == "Edit":
        for i, landmark in enumerate(landmarks):
            lx, ly = landmark
            # if the cursor is close enough to a landmark, start dragging it
            if abs(lx - x) < selection_range and abs(ly - y) < selection_range:
                dragging_landmark = i
                break


def move_landmark(event, image, landmarks, label, selected_button, index):
    """
    This function updates the position of a landmark when it is being dragged.
    It first checks if a landmark is being dragged.
    If a landmark is being dragged, it updates the position of the landmark to the cursor position.

    Args:
        event (Tkinter.Event): The event that triggered the function.
        landmarks (list): A list of tuples representing the x and y coordinates of the landmarks.
        index (int): The index of the current image.

    """
    global dragging_landmark
    if selected_button.get() != "Edit":
        return
    if dragging_landmark is not None:
        landmarks[dragging_landmark] = (event.x, event.y)
        display_image_with_landmarks(image, landmarks, label, index)


def end_drag():
    """
    Resets the global variable 'dragging_landmark' to None, indicating that no landmark is currently being dragged.
    """
    global dragging_landmark
    dragging_landmark = None


#------------------------ UI ------------------------#

def generate_UI(args):
    """
    This function generates the user interface for the application.
    It first sets global variables based on the arguments passed to the function.
    Then it creates the root window and several Tkinter variables.

    Args:
        args (argparse.Namespace): The arguments passed to the script.

    """
    global TKINTER_WINDOW_SIZE, LANDMARKS_PATH, NO_FRAMES, FPS, SYNTH_STEPS, PROJECTION, DEBUG
    TKINTER_WINDOW_SIZE, LANDMARKS_PATH, NO_FRAMES, FPS, SYNTH_STEPS, PROJECTION, DEBUG = args.image_size, args.landmarks_path, args.no_frames, args.fps, args.synth_steps, args.projection_frames, args.debug

    root = Tk()
    selected_button = StringVar()
    remove_bg = BooleanVar()
    no_sift_landmarks = StringVar()
    is_face = BooleanVar(value=True)
    gan_projection = BooleanVar()
    q_value = StringVar(value="3")
    
    # ---- Image initialization ----

    input_image1_path = select_image()
    input_image2_path = select_image()

    image1_pil = resize_and_crop(input_image1_path, TKINTER_WINDOW_SIZE)
    image2_pil = resize_and_crop(input_image2_path, TKINTER_WINDOW_SIZE)

    image1_cv = cv2.cvtColor(np.array(image1_pil), cv2.COLOR_RGB2BGR)
    image2_cv = cv2.cvtColor(np.array(image2_pil), cv2.COLOR_RGB2BGR)

    image1 = ImageTk.PhotoImage(image1_pil)
    image2 = ImageTk.PhotoImage(image2_pil)

    label1 = Label(root, image=image1)
    label1.grid(row=0, column=0)
    label2 = Label(root, image=image2)
    label2.grid(row=0, column=1)

    label1.bind("<Button-1>", lambda event: handle_landmark(event, image1_cv, landmarks1, label1, selected_button, index=0))
    label1.bind("<B1-Motion>", lambda event: move_landmark(event, image1_cv, landmarks1, label1, selected_button, index=0))
    label1.bind("<ButtonRelease-1>", lambda event: end_drag())

    label2.bind("<Button-1>", lambda event: handle_landmark(event, image2_cv, landmarks2, label2, selected_button, index=1))
    label2.bind("<B1-Motion>", lambda event: move_landmark(event, image2_cv, landmarks2, label2, selected_button, index=1))
    label2.bind("<ButtonRelease-1>", lambda event: end_drag())

    # ---- Buttons functions ----

    def detect_landmarks_button():
        button_detection.config(state='disabled')
        generate_automatic_landmarks(image1_cv, image2_cv, label1, label2)
        button_densify.config(state='normal')
    
    def densify_landmarks_button():
        landmarks_densification(image1_cv, image2_cv, label1, label2, iterations=1, threshold=0)

    def get_number_landmarks():
        try:
            return int(no_sift_landmarks.get())
        except:
            return None

    def detect_SIFT_landmarks():
        button_sift.config(state='disabled')
        generate_sift_landmarks(image1_cv, image2_cv, label1, label2, get_number_landmarks())
        button_densify.config(state='normal')

    def read_landmarks():
        global landmarks1, landmarks2
        landmarks1, landmarks2 = read_landmarks_from_file(LANDMARKS_PATH)
        colors = generate_random_colors(len(landmarks1))
        colors_landmarks[0].extend(colors)
        colors_landmarks[1].extend(colors)
        display_image_with_landmarks(image1_cv, landmarks1, label1, index=0)
        display_image_with_landmarks(image2_cv, landmarks2, label2, index=1)

    def write_landmarks():
        f = open(LANDMARKS_PATH, 'w')
        f.write(str(TKINTER_WINDOW_SIZE) + '\n')
        for i in range(len(landmarks1)):
            f.write(str(int(landmarks1[i][0])) + ',' +
                    str(int(landmarks1[i][1])) + ',' + 
                    str(int(landmarks2[i][0])) + ',' + 
                    str(int(landmarks2[i][1])) + '\n')
        f.close()

    def reset_landmarks():
        global landmarks1, landmarks2
        landmarks1, landmarks2 = [], []
        colors_landmarks[0], colors_landmarks[1] = [], []
        display_image_with_landmarks(image1_cv, landmarks1, label1, index=0)
        display_image_with_landmarks(image2_cv, landmarks2, label2, index=1)
        button_detection.config(state='normal')
        button_sift.config(state='normal')

    def start_process():
        face1 = image1_cv
        face2 = image2_cv

        # make q have the default value of 3
        try:
            args.q = int(q_value.get())
        except ValueError:
            args.q = 3
        args.is_face = is_face.get()
        args.gan_projection = gan_projection.get()

        if remove_bg.get():
            cv2.imshow("face1", face1)
            cv2.imshow("face2", face2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        generate_morphing_animation(
            face1,
            face2,
            landmarks1, 
            landmarks2, 
            remove_bg.get(),
            args)
        
        root.destroy()

    # ---- Buttons definition ----

    button_frame = Frame(root)
    button_frame.grid(row=1, column=0, columnspan=2)

    button_detection = ttk.Button(button_frame, text="Detect Landmarks", command=detect_landmarks_button)
    button_detection.grid(row=0, column=0)

    button_densify = ttk.Button(button_frame, text="Densify Landmarks", command=densify_landmarks_button)
    button_densify.grid(row=0, column=1)
    button_densify.config(state='disabled')

    button_add = ttk.Radiobutton(button_frame, text="Add Landmarks", variable=selected_button, value="Add")
    button_add.grid(row=0, column=2)

    button_edit = ttk.Radiobutton(button_frame, text="Edit Landmarks", variable=selected_button, value="Edit")
    button_edit.grid(row=0, column=3)

    button_remove = ttk.Radiobutton(button_frame, text="Remove Landmarks", variable=selected_button, value="Remove")
    button_remove.grid(row=0, column=4)

    button_rm_bg = ttk.Checkbutton(button_frame, text="Remove Background", variable=remove_bg)
    button_rm_bg.grid(row=0, column=5)

    button_sift = ttk.Button(button_frame, text="SIFT Landmarks", command=detect_SIFT_landmarks)
    button_sift.grid(row=0, column=6)

    no_sift_landmarks_entry = ttk.Entry(button_frame, textvariable=no_sift_landmarks)
    no_sift_landmarks_entry.grid(row=0, column=7)

    button_read = ttk.Button(button_frame, text="Read Landmarks", command=read_landmarks)
    button_read.grid(row=1, column=0)

    button_write = ttk.Button(button_frame, text="Write Landmarks", command=write_landmarks)
    button_write.grid(row=1, column=1)

    button_reset = ttk.Button(button_frame, text="Reset Landmarks", command=reset_landmarks)
    button_reset.grid(row=1, column=2)

    label_q = ttk.Label(button_frame, text="Enter q value:")
    label_q.grid(row=1, column=3)

    entry_q = ttk.Entry(button_frame, textvariable=q_value)
    entry_q.grid(row=1, column=4)

    checkbox_is_face = ttk.Checkbutton(button_frame, text="Is it a face?", variable=is_face)
    checkbox_is_face.grid(row=1, column=5)

    checkbox_projection = ttk.Checkbutton(button_frame, text="GAN projection?", variable=gan_projection)
    checkbox_projection.grid(row=1, column=6)

    buttonMain = Button(button_frame, text="Create morphing animation", command=start_process)
    buttonMain.grid(row=1, column=7, columnspan=7)

    root.mainloop()



def read_landmarks_from_file(path):
    """
    This function reads landmarks from a file and returns them as a list of tuples.
    Each tuple represents the x and y coordinates of a landmark.

    Args:
        file_path (str): The path to the file containing the landmarks.

    Returns:
        list: A list of tuples representing the landmarks.
    """
    f = open(path, 'r')
    scale = int(f.readline())

    lmrks1 = []
    lmrks2 = []
    for line in f.readlines():
        line = line.split(',')
        lmrks1.append((int(line[0]), int(line[1])))
        lmrks2.append((int(line[2]), int(line[3])))
    f.close()

    # scale the landmarks to the currrent window size
    lmrks1 = [(int(x * TKINTER_WINDOW_SIZE / scale), int(y * TKINTER_WINDOW_SIZE / scale)) for x, y in lmrks1]
    lmrks2 = [(int(x * TKINTER_WINDOW_SIZE / scale), int(y * TKINTER_WINDOW_SIZE / scale)) for x, y in lmrks2]

    return lmrks1, lmrks2


def cut_background(image, landmarks):
    """
    This function applies a mask to an image based on the convex hull of the landmarks.
    It first calculates the convex hull of the landmarks, then creates a mask and draws the hull on it.
    The mask is then applied to the image.

    Args:
        image (np.array): The original image.
        landmarks (list): A list of tuples representing the x and y coordinates of the landmarks.

    Returns:
        np.array: The image with the mask applied.
    """
    landmarks = np.array(landmarks, dtype=np.int32)
    hull = cv2.convexHull(landmarks)
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    cv2.fillPoly(mask, [hull], (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image = cv2.bitwise_and(image, mask)
    return image


def generate_random_colors(no_colors):
    """
    This function generates a list of random RGB colors.
    Each color is a tuple of three integers between 0 and 255.

    Args:
        no_colors (int): The number of colors to generate.

    Returns:
        list: A list of tuples representing the RGB colors.
    """
    colors = []
    for _ in range(no_colors):
        colors.append((np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
    return colors


def generate_automatic_landmarks(image1, image2, label1, label2):
    """
    This function generates automatic facial landmarks for two images.
    It first calls the `automatic_facial_landmarks` function to get the landmarks.
    Then it generates random colors for the landmarks and adds them to the `colors_landmarks` list.

    Args:
        image1 (np.array): The first image.
        image2 (np.array): The second image.
        label1 (Tkinter.Label): The label on which the first image is displayed.
        label2 (Tkinter.Label): The label on which the second image is displayed.
    """
    global landmarks1, landmarks2
    
    lmrks1, lmrks2 = automatic_facial_landmarks(image1, image2)

    landmarks1, landmarks2 = lmrks1, lmrks2

    # generate random colors for the landmarks
    colors = generate_random_colors(len(landmarks1))
    colors_landmarks[0].extend(colors)
    colors_landmarks[1].extend(colors)
    
    display_image_with_landmarks(image1, lmrks1, label1, index=0)
    display_image_with_landmarks(image2, lmrks2, label2, index=1)


def landmarks_densification(image1, image2, label1, label2, iterations=1, threshold=0):
    """
    This function densifies the landmarks for two images.
    It first calls the `densify_landmarks` function to get the densified landmarks.
    Then it generates random colors for the new landmarks and adds them to the `colors_landmarks` list.

    Args:
        image1 (np.array): The first image.
        image2 (np.array): The second image.
        label1 (Tkinter.Label): The label on which the first image is displayed.
        label2 (Tkinter.Label): The label on which the second image is displayed.
        iterations (int, optional): The number of iterations for the densification process. Defaults to 1.
        threshold (int, optional): The threshold for the densification process. Defaults to 0.
    """
    global landmarks1, landmarks2

    lmrks1, lmrks2 = densify_landmarks(landmarks1, landmarks2, iterations, threshold)
    
    colors = generate_random_colors(len(lmrks1) - len(landmarks1))
    colors_landmarks[0].extend(colors)
    colors_landmarks[1].extend(colors)
    landmarks1, landmarks2 = lmrks1, lmrks2

    display_image_with_landmarks(image1, lmrks1, label1, index=0)
    display_image_with_landmarks(image2, lmrks2, label2, index=1)


def generate_sift_landmarks(image1, image2, label1, label2, no_landmarks=None):
    """
    This function generates SIFT (Scale-Invariant Feature Transform) landmarks for two images.
    It first detects keypoints and computes descriptors for both images using SIFT.
    Then it matches the descriptors using Brute-Force Matcher.
    The matched keypoints are used as the landmarks.

    Args:
        image1 (np.array): The first image.
        image2 (np.array): The second image.
        label1 (Tkinter.Label): The label on which the first image is displayed.
        label2 (Tkinter.Label): The label on which the second image is displayed.
        no_landmarks (int, optional): The number of landmarks to generate. If None, all matches are used. Defaults to None.
    """
    global landmarks1, landmarks2

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # we want to sort the matches by distance and only keep the first no_landmarks
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # if no_landmarks is not None, only keep the first no_landmarks, otherwise keep all matches
    if no_landmarks is not None:
        matches = matches[:no_landmarks]

    # landmarks need to be corresponding, so we need to keep the indices of the keypoints
    lmrks1 = [kp1[match.queryIdx].pt for match in matches]
    lmrks2 = [kp2[match.trainIdx].pt for match in matches]

    landmarks1, landmarks2 = lmrks1, lmrks2

    colors = generate_random_colors(len(landmarks1))
    colors_landmarks[0].extend(colors)
    colors_landmarks[1].extend(colors)
    
    display_image_with_landmarks(image1, lmrks1, label1, index=0)
    display_image_with_landmarks(image2, lmrks2, label2, index=1)


# function not in use, but kept for future reference
def show_output_animation():
    """
    This function displays an animation of the output images.
    It first reads the images from the output path, then creates a figure and displays the images in a loop.

    Args:
        output_path (str): The path to the directory containing the output images.
    """
    cap = cv2.VideoCapture('morphing_animationnn.mp4')

    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Morphing Animation', frame)
            if cv2.waitKey(100) & 0xFF == ord('q') & 0xFF == 27:
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    cv2.destroyAllWindows()

    if True: 
        cap = cv2.VideoCapture('projection_animationnn.mp4')

        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Projection Animation', frame)
                if cv2.waitKey(50) & 0xFF == ord('q') & 0xFF == 27:
                    break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()
        cv2.destroyAllWindows()