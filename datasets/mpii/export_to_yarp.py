
import argparse
import pathlib


def export_to_yarp(input_folder, annotations_path, output_folder):
    print('export_to_yarp is not implemented for MPII dataset')
    pass


def main(args):

    # annotations file path
    annotations_path = pathlib.Path(args.a)
    annotations_path = pathlib.Path(annotations_path.resolve())

    # input image folder
    input_folder = pathlib.Path(args.i)
    input_folder = pathlib.Path(input_folder.resolve())

    # output folder
    output_folder = pathlib.Path(args.o)
    output_folder = pathlib.Path(output_folder.resolve())

    export_to_yarp(input_folder, annotations_path, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to input images folder')
    parser.add_argument('-a', help='path to input annotations folder')
    parser.add_argument('-o', help='path to output folder')
    args = parser.parse_args()

    main(args)
