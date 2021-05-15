"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os

IMG_EXTENSIONS = [
    ".jpg",
    ".JPG",
    ".jpeg",
    ".JPEG",
    ".png",
    ".PNG",
    ".ppm",
    ".PPM",
    ".bmp",
    ".BMP",
    ".tiff",
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_tensor(filename):
    return filename.endswith(".pt")


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def make_latents_dataset(dir):
    tensors = []
    assert os.path.isdir(dir), "%s is not a valid directory" % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_tensor(fname):
                path = os.path.join(root, fname)
                tensors.append(path)
    return tensors
