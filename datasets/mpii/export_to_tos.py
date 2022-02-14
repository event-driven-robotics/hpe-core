
import argparse
import cv2
import pathlib

from tqdm import tqdm

import datasets.utils.mat_files as mat_utils
import datasets.mpii.utils.parsing as mpii_parse

from datasets.utils.events_representation import EROSSynthetic


def export_to_tos(data_ann: dict, image_folder: pathlib.Path, output_folder: pathlib.Path, gaussian_blur_k_size: int,
                  gaussian_blur_sigma: int, canny_low_th: int, canny_high_th: int, canny_aperture: int,
                  canny_l2_grad: bool, salt_pepper_low_th: int, salt_pepper_high_th: int) -> None:

    """Export MPII images to TOS-like frames.

    The function converts MPII frames to a TOS-like representation by applying Gaussian blur, a Canny edge detector
    and salt and pepper noise to the original RGB frames (see class EROSSynthetic in datasets/utils/events_representation.py
    for implementation details).

    Parameters
    ----------
    data_ann: dict
        Dictionary containing MPII annotation data
    image_folder: pathlib.Path
        Path to the images folder
    output_folder: pathlib.Path
        Path to the output folder where the TOS-like frames and the json file will be saved
    gaussian_blur_k_size: int
        kernel size used for pre-Canny edge detection Gaussian blurring
    gaussian_blur_sigma: int
        sigma used for pre-Canny edge detection Gaussian blurring
    canny_low_th: int
        min value for Canny's hysteresis thresholding
    canny_high_th: int
        max value for Canny's hysteresis thresholding
    canny_aperture: int
        aperture size for Canny's Sobel operator
    canny_l2_grad: bool
        flag indicating if L2 norm has to be used for Canny's gradient computation
    salt_pepper_low_th: int
        salt and pepper min value for post-Canny edge noise addition
    salt_pepper_high_th: int
        salt and pepper max value for post-Canny edge noise addition
    """

    iterator = mpii_parse.MPIIIterator(data_ann, image_folder)

    tos = EROSSynthetic(gaussian_blur_k_size, gaussian_blur_sigma, canny_low_th, canny_high_th,
                        canny_aperture, canny_l2_grad, salt_pepper_low_th, salt_pepper_high_th)

    for fi, (img, poses_ann, img_name) in enumerate(tqdm(iterator)):

        if img is None:
            print(f'image {img_name} does not exist')
            continue

        frame = tos.get_frame(img)
        file_path = output_folder / img_name
        if not cv2.imwrite(str(file_path), frame):
            print(f'could not save image to {str(file_path)}')
            continue


def main(args):

    data_ann_path = pathlib.Path(args.a)
    data_ann_path = pathlib.Path(data_ann_path.resolve())
    data_ann = mat_utils.loadmat(str(data_ann_path))

    img_folder_path = pathlib.Path(args.i)
    img_folder_path = pathlib.Path(img_folder_path.resolve())

    output_folder_path = pathlib.Path(args.o)
    output_folder_path = pathlib.Path(output_folder_path.resolve())
    output_folder_path.mkdir(parents=True, exist_ok=True)

    export_to_tos(data_ann, img_folder_path, output_folder_path,
                  gaussian_blur_k_size=args.gk, gaussian_blur_sigma=args.gs,
                  canny_low_th=args.cl, canny_high_th=args.ch, canny_aperture=args.ca, canny_l2_grad=args.cg,
                  salt_pepper_low_th=args.sl, salt_pepper_high_th=args.sh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ################
    # general params
    ################

    parser.add_argument('-a', help='path to annotations file')
    parser.add_argument('-i', help='path to images folder')
    parser.add_argument('-o', help='path to output folder')

    ######################
    # synthetic TOS params
    ######################

    # canny edge detector params
    parser.add_argument('-gk', help='gaussian blur kernel size', type=int, default=5)
    parser.add_argument('-gs', help='gaussian blur sigma', type=int, default=0)
    parser.add_argument('-cl', help='canny edge low threshold', type=int, default=0)
    parser.add_argument('-ch', help='canny edge high threshold', type=int, default=1000)
    parser.add_argument('-ca', help='canny edge aperture', type=int, default=5)
    parser.add_argument('-cg', help='canny edge l2 gradient flag', dest='cg', action='store_true')
    parser.set_defaults(cg=False)

    # salt and pepper params
    parser.add_argument('-sl', help='salt and pepper low threshold', type=int, default=5)
    parser.add_argument('-sh', help='salt and pepper high threshold', type=int, default=250)

    args = parser.parse_args()

    main(args)
