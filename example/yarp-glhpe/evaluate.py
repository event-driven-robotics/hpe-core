import glob

import sys
from tqdm import tqdm
import numpy as np
import cv2
import json
import os
from time import time

HOME_DIR = os.path.expanduser('~')
sys.path.append(os.path.join(HOME_DIR, 'code', 'gl_hpe'))
sys.path.append(os.path.join(HOME_DIR, 'code', 'gl_hpe', 'venv', 'src', 'event-library'))
sys.path.append(os.path.join(HOME_DIR, 'code', 'gl_hpe', 'venv', 'src', 'pose3d-utils'))
sys.path.append(os.path.join(HOME_DIR, 'code', 'hpe-core', 'example', 'movenet'))
import experimenting
import torch
from experimenting.utils.visualization import plot_skeleton_2d, plot_skeleton_3d, plot_skeleton_2d_lined
from experimenting.utils.skeleton_helpers import Skeleton
from experimenting.utils import utilities
from experimenting import utils
from lib.task.task_tools import write_output
from lib.utils.utils import ensure_loc

dev = False

class GlHpeModule():

    def __init__(self):
        self.counter = 0
        self.dataset = 'DHP19'
        self.image_w = 346  # Size of image read.
        self.image_h = 260  #
        self.datadir = os.path.join(HOME_DIR, 'data')
        self.inputdir = os.path.join(self.datadir, "DHP19/training/eF")
        self.inputfilelist = os.path.join(self.datadir, "DHP19/training/anno/poses_val.json")
        self.checkpoint_path = os.path.join(self.datadir,
                                            "gl-hpe/lifting_monocular_events_to_hpe/dhp19_constantcount/checkpoints")
        self.P_mat_dir = os.path.join(self.datadir, 'gl-hpe/P_matrices/')

        # self.ch_idx = 3
        if dev:
            self.resultsPath = os.path.join(self.datadir, 'results_test', self.dataset)
        else:
            self.resultsPath = os.path.join(self.datadir, 'results', self.dataset)
        self.image_w_model = 346  # Size of the image expected by the model
        self.image_h_model = 260  #
        self.output_w = 346  # Size of the image expected by yarp
        self.output_h = 260  #
        self.fname = None
        self.fname_ts = None
        self.P_mat_cam = ['', '', '', '']
        self.extrinsics_matrix = list(range(4))
        self.camera_matrix = list(range(4))
        self.dhpcore = None
        self.model = None

    def get_cam(self, file):
        return int(os.path.basename(file)[3])

    def get_sub(self, file):
        return int(os.path.basename(file).split('_')[1][1:])

    def get_samplename(self, file):
        return '_'.join(os.path.basename(file).split('_')[:-1])

    def configure(self):
        # Initialise YARP
        #
        # read flags and parameters
        self.dhpcore = experimenting.dataset.DHP19Core('test',
                                                       data_dir=os.path.join(self.datadir,
                                                                             'time_count_dataset/movements_per_frame'),
                                                       joints_dir=os.path.join(self.datadir,
                                                                               "time_count_dataset/labels_full_joints/"),
                                                       hm_dir="", labels_dir="", preload_dir="", n_joints=13,
                                                       n_classes=33,
                                                       partition='cross-subject', n_channels=1, cams=[1, 3],
                                                       movements=None, test_subjects=[6, 7])
        print(self.checkpoint_path)
        print(os.path.exists(self.checkpoint_path))

        self.model = utilities.load_model(self.checkpoint_path, "MargiposeEstimator",
                                          core=self.dhpcore).eval().double()

        for ch_idx in range(4):

            if ch_idx == 0:
                self.P_mat_cam[ch_idx] = np.load(os.path.join(self.P_mat_dir, 'P1.npy'))
            elif ch_idx == 3:
                self.P_mat_cam[ch_idx] = np.load(os.path.join(self.P_mat_dir, 'P2.npy'))
            elif ch_idx == 2:
                self.P_mat_cam[ch_idx] = np.load(os.path.join(self.P_mat_dir, 'P3.npy'))
            elif ch_idx == 1:
                self.P_mat_cam[ch_idx] = np.load(os.path.join(self.P_mat_dir, 'P4.npy'))
            self.extrinsics_matrix[ch_idx], self.camera_matrix[ch_idx] = utils.decompose_projection_matrix(
                self.P_mat_cam[ch_idx])

        test_subs = list(range(13,18))
        with open(self.inputfilelist, 'r') as f:
            val_label_list = json.loads(f.readlines()[0])
        val_label_list_sorted = sorted(val_label_list, key=lambda d: d['img_name'])

        self.file_list = [[os.path.join(self.inputdir,i['img_name']),i['ts']] for i in val_label_list_sorted]
        print('Files found:', len(self.file_list))
        return True



    def run(self):
        print("Press space at the image window to end the program.")

        count = 0
        done = []
        for i in tqdm(range(len(self.file_list))):
            file = self.file_list[i][0]
            cam = self.get_cam(file)
            ts = self.file_list[i][1]
            print(ts)

            # dev mode exit strategy
            if dev:
                if cam in done:
                    continue
                if len(done) > 10:
                    exit()
                count += 1
                if count<1000:
                    continue
                if count > 1010:
                    count = 0
                    print(cam)
                    done.append(cam)
            image = cv2.imread(file)
            start = time()
            # assert (np.sum(image[:,:,0]-image[:,:,2])) == 0.0
            input_image = np.copy(image[:,:,0]) / 255.0


            if len(input_image.shape) == 2:
                input_image = np.expand_dims(input_image, -1)
            if len(input_image.shape) == 3:
                input_image = np.expand_dims(input_image, 0)
            #
            # Predict the pose
            torch_image = torch.from_numpy(input_image)
            preds, outs = self.model(torch_image.permute(0, -1, 1, 2))
            pred_sk = Skeleton(preds[0].detach().numpy()).denormalize(self.image_h, self.image_w,
                                                                      camera=torch.tensor(self.camera_matrix[cam])). \
                reproject_onto_world(torch.tensor(self.extrinsics_matrix[cam]))
            pred_joints = pred_sk.get_2d_points(self.image_h, self.image_w, p_mat=torch.tensor(self.P_mat_cam[cam]))
            if dev:
                if np.max(pred_joints[:,0])>260 or np.max(pred_joints[:,1])>346:
                    print('\n',np.max(pred_joints[:,0]),np.max(pred_joints[:,1]))

            h, w, _ = image.shape
            pred_corrected = pred_joints[:]
            pred_corrected[:,1] = h - pred_joints[:, 1]

            if dev:
                # Obtain the 2D prediction as an image
                fig2D = plot_skeleton_2d(input_image[0].squeeze(), pred_joints, return_figure=True, lines=True)
                fig2D.canvas.draw()
                img = np.fromstring(fig2D.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig2D.canvas.get_width_height()[::-1] + (3,))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Visualize the result
                cv2.imshow("output", img)
                print(img.shape)
                k = cv2.waitKey(10)

                # out_image = image
                # for joint in range(len(pred_joints[:,0])):
                #     cv2.circle(out_image,(int(pred_corrected[joint,0]),int(pred_corrected[joint,1])),3,(0,0,255),1)
                # # Visualize the result
                # cv2.imshow("output", out_image)
                # k = cv2.waitKey(10)
                if k == 32:
                    exit()

            ensure_loc(os.path.join(self.resultsPath,self.get_samplename(file)))
            output_filename = os.path.join(self.resultsPath,self.get_samplename(file),'gl-hpe.csv')
            write_output(output_filename, pred_corrected, timestamp= ts, delay= time()-start)


        return

        # End of input file
        # if self.last_timestamp == stamp:
        #     return False

        # self.counter += 1  # can be used to interrupt the program

        # cv2.imwrite(os.path.join(self.resultsPath, 'input_'+str(self.counter), '.png'), input_image)
        # cv2.imwrite(os.path.join(self.resultsPath, 'output_2D_'+str(self.counter),'.png'), img)

        # write the Results into numpy arrays
        # utilities.save_2D_prediction(pred_joints, fname=os.path.join(self.resultsPath, self.fname), overwrite=False)
        # utilities.save_timestamp(stamp, fname=os.path.join(self.resultsPath, self.fname_ts), overwrite=False)

        #        np_output[:, :] = img
        # self.output_port.write(self.yarp_image_out)
        # self.last_timestamp = stamp
        # if k == 32:
        #     return False
        # return True


if __name__ == '__main__':
    # prepare and configure the resource finder

    # create the module
    module = GlHpeModule()
    module.configure()

    module.run()
