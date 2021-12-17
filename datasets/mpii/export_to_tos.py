import argparse
import cv2
import pathlib

import datasets.utils.mat_files as mat_utils
import datasets.mpii.utils.parsing as mpii_parse

from datasets.utils.events_representation import TOSSynthetic


def export_to_tos(data_ann, image_folder, output_folder, gaussian_blur_k_size, gaussian_blur_sigma, canny_low_th,
                  canny_high_th, canny_aperture, canny_l2_grad, salt_pepper_low_th, salt_pepper_high_th):

    iterator = mpii_parse.MPIIIterator(data_ann, image_folder)

    tos = TOSSynthetic(gaussian_blur_k_size, gaussian_blur_sigma, canny_low_th, canny_high_th,
                       canny_aperture, canny_l2_grad, salt_pepper_low_th, salt_pepper_high_th)

    for fi, (img, poses_ann, img_name) in enumerate(iterator):
        frame = tos.get_frame(img)
        file_path = output_folder / img_name
        if not cv2.imwrite(str(file_path), frame):
            print(f'could not save image to {str(file_path)}')

        # TODO: export 2d poses


def main(args):

    data_ann_path = pathlib.Path(args.a)
    data_ann_path = pathlib.Path(data_ann_path.resolve())
    data_ann = mat_utils.loadmat(str(data_ann_path))

    img_folder_path = pathlib.Path(args.i)
    img_folder_path = pathlib.Path(img_folder_path.resolve())

    output_folder_path = pathlib.Path(args.o)
    output_folder_path = pathlib.Path(output_folder_path.resolve())
    output_folder_path.mkdir(parents=True, exist_ok=False)

    export_to_tos(data_ann, img_folder_path, output_folder_path,
                  gaussian_blur_k_size=args.gk, gaussian_blur_sigma=args.gs,
                  canny_low_th=args.cl, canny_high_th=args.ch, canny_aperture=args.ca, canny_l2_grad=args.cg,
                  salt_pepper_low_th=args.sl, salt_pepper_high_th=args.sh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', help='path to annotations file')
    parser.add_argument('-i', help='path to images folder')
    parser.add_argument('-o', help='path to output folder')

    # synthetic TOS params

    # canny edge detector params
    parser.add_argument('-gk', help='gaussian blur kernel size', type=int, default=5)
    parser.add_argument('-gs', help='gaussian blur sigma', type=int, default=0)
    parser.add_argument('-cl', help='canny edge low threshold', type=int, default=0)
    parser.add_argument('-ch', help='canny edge high threshold', type=int, default=1000)
    parser.add_argument('-ca', help='canny edge aperture', type=int, default=5)
    parser.add_argument('-cg', help='canny edge l2 gradient flag', dest='cg', action='store_true')
    parser.set_defaults(cg=False)

    # salt and pepper params
    parser.add_argument('-sl', help='salt and pepper low threshold', type=int, default=30)
    parser.add_argument('-sh', help='salt and pepper high threshold', type=int, default=225)

    args = parser.parse_args()

    main(args)
