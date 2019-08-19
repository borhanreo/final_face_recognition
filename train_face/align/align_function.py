from PIL import Image
from align.detector import detect_faces
from align.align_trans import get_reference_facial_points, warp_and_crop_face
import numpy as np
import os
from tqdm import tqdm
import argparse
def get_crop_image(dest_root,crop_size,path,image_name):
    if not os.path.isdir(dest_root):
        os.mkdir(dest_root)
    scale = crop_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale
    print("Processing",path)
    img = Image.open(path)
    try:  # Handle exception
        _, landmarks = detect_faces(img)
        if len(landmarks) == 0:  # If the landmarks cannot be detected, the img will be discarded
            print("{} is discarded due to non-detected landmarks!")
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
        img_warped = Image.fromarray(warped_face)
        if image_name.split('.')[-1].lower() not in ['jpg', 'jpeg']:  # not from jpg
            image_name = '.'.join(image_name.split('.')[:-1]) + '.jpg'
        img_warped.save(dest_root, image_name)
        return dest_root + image_name
    except Exception:
        print("{} is discarded due to exception!")
        return "Nooo"






