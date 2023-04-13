import tensorflow as tf
import tensorflow_hub as hub

import numpy as np
import argparse
import cv2, csv, time, os
from funcs import movenet, _keypoints_and_edges_for_display, draw_prediction_on_image

# from datasets.h36m.utils.parsing import movenet_to_hpecore

from pycore.moveenet.visualization.visualization import add_skeleton, movenet_to_hpecore
from pycore.moveenet.utils.utils import ensure_loc
from datasets.utils.export import str2bool

def get_movenet_13(pose_17):
    assert pose_17.shape == (17,2)
    pose_13 = pose_17[4:,:]
    pose_13[0,:] = pose_17[0,:]
    return pose_13

def write_results(path, row):
    # Write a data point into a csvfile
    with open(path, 'a') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerow(row)


def create_row(ts, skt, delay=0.0):
    # Function to create a row to be written into a csv file.
    row = []
    ts = float(ts)
    row.extend([ts, delay])
    row.extend(skt)
    return row

def main(args):
    model_name = "movenet_lightning" #@param ["movenet_lightning", "movenet_thunder", "movenet_lightning_f16.tflite", "movenet_thunder_f16.tflite", "movenet_lightning_int8.tflite", "movenet_thunder_int8.tflite"]

    if "movenet_lightning" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        input_size = 192
    elif "movenet_thunder" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)

    # if args.dev:
    #     video_path = "/home/ggoyal/data/h36m_cropped/cropped_video/cam2_S9_Directions.mp4"
    #     write_csv = "/home/ggoyal/data/results/tester/cam2_S9_Directions/movenet_rgb.csv"
    # else:
    video_path=args.input
    write_csv = args.output
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    filename = os.path.basename(video_path).split('.')[0]
    # Read until video is completed
    fi = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            height, width, _ = frame.shape
            image = tf.convert_to_tensor(frame, dtype=tf.uint8)
            # Resize and pad the image to keep the aspect ratio and fit the expected size.
            input_image = tf.expand_dims(image, axis=0)
            input_image = tf.image.resize(input_image, [input_size, input_size])
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)/1000

            start_sample = time.time()
            keypoints_with_scores = movenet(input_image, module)
            (keypoint_locs, keypoint_edges,
             edge_colors) = _keypoints_and_edges_for_display(
                keypoints_with_scores, height, width,0)
            keypoint_locs_13 = get_movenet_13(keypoint_locs)
            kps_hpecore = movenet_to_hpecore(keypoint_locs_13)
            kps_pre_hpecore = np.reshape(kps_hpecore, [-1])
            if args.dev or args.save_image:
                for i in range(len(keypoint_locs_13[:])):
                    frame = add_skeleton(frame, kps_pre_hpecore, (0, 0, 255), True, normalised=False)
                    # pre = keypoint_locs_13
                    # x = int(pre[i,0])
                    # y = int(pre[i,1])
                    # cv2.circle(frame, (x, y), 3, (0, 0, 255), 3)  # predicted keypoints in red
                if args.dev:
                    cv2.imshow('Frame', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                if args.save_image:
                    ensure_loc(os.path.dirname(args.save_image))
                    cv2.imwrite(os.path.join(args.save_image, f'{filename}_{fi:04d}.jpg'), frame)

            if args.output:

                row = create_row(ts, kps_pre_hpecore, delay=time.time() - start_sample)
                ensure_loc(os.path.dirname(write_csv))
                write_results(write_csv, row)
            fi+=1

        # Break the loop
        else:
            break
        if args.stop:
            if fi > args.stop:
                break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='Path to sample', default='', type=str)
    parser.add_argument('-output', help='Path to csv file', default=None, type=str)
    parser.add_argument('-save_image', help='Path to image folder', default=None, type=str)
    parser.add_argument("-dev", type=str2bool, nargs='?', const=True, default=False, help="Set for visualisation mode.")
    parser.add_argument("-stop", type=str, default=None, help="early stop.")

    args = parser.parse_args()
    main(args)