#!/usr/bin/env python3
import sys
import yarp
import numpy as np
import cv2
import time

class OverlayViewerModule(yarp.RFModule):
    def __init__(self):
        yarp.RFModule.__init__(self)

        # Ports
        self.img_port = yarp.BufferedPortImageMono()
        self.sklt_port = yarp.BufferedPortBottle()
        self.out_port = yarp.BufferedPortImageRgb()

        self.stamp = yarp.Stamp()

        # Cache latest skeleton
        self.latest_joints = None  # (13, 2)
        self.latest_conf = None    # (13,)
        self.conf_thr = 0.3        # confidence threshold

        self._fps_t0 = time.time()
        self._fps_counter = 0

    def configure(self, rf):
        # Initialise YARP
        yarp.Network.init()
        if not yarp.Network.checkNetwork(2.0):
            print("Could not find YARP network, run yarpserver.")
            return False

        # Module name
        self.setName(rf.check("name", yarp.Value("/movenetViewer")).asString())

        # Open ports
        if not self.img_port.open(self.getName() + "/img:i"):
            print("Could not open image input port")
            return False

        if not self.sklt_port.open(self.getName() + "/sklt:i"):
            print("Could not open skeleton input port")
            return False

        if not self.out_port.open(self.getName() + "/overlay:o"):
            print("Could not open overlay output port")
            return False

        return True

    def getPeriod(self):
        # 50 Hz viewer
        return 0.01

    def interruptModule(self):
        self.img_port.interrupt()
        self.sklt_port.interrupt()
        self.out_port.interrupt()
        return True

    def close(self):
        self.img_port.close()
        self.sklt_port.close()
        self.out_port.close()
        return True

    # --- Helper: parse /movenet/sklt:o Bottle (13 joints) ---
    def _update_skeleton_from_bottle(self):
        b = self.sklt_port.read(False)  # non-blocking
        if b is None:
            return

        if b.size() < 2:
            return

        tag = b.get(0).asString()
        if tag != "SKLT":
            return

        data_list = b.get(1).asList()
        n_vals = data_list.size()
        if n_vals < 39:  # 13 joints -> 26 coords + 13 conf = 39
            return

        vals = np.array(
            [data_list.get(i).asFloat64() for i in range(n_vals)],
            dtype=np.float32
        )

        n_joints = 13
        coords = vals[:2 * n_joints].reshape((n_joints, 2))
        conf = vals[2 * n_joints:2 * n_joints + n_joints]

        self.latest_joints = coords
        self.latest_conf = conf
    
    def _draw_skeleton(self, vis_img):
        if self.latest_joints is None or self.latest_conf is None:
            return

        joints = self.latest_joints.copy()
        conf = self.latest_conf
        n_joints = joints.shape[0]

        h, w, _ = vis_img.shape

        # NO scaling, assume joints are already in image pixel coordinates
        # Just clamp to image bounds in case of small numerical overshoot
        joints[:, 0] = np.clip(joints[:, 0], 0, w - 1)
        joints[:, 1] = np.clip(joints[:, 1], 0, h - 1)

        # 13-joint mapping:
        # 0: head
        # 1: shoulder_right
        # 2: shoulder_left
        # 3: elbow_right
        # 4: elbow_left
        # 5: hip_left
        # 6: hip_right
        # 7: wrist_right
        # 8: wrist_left
        # 9: knee_right
        # 10: knee_left
        # 11: ankle_right
        # 12: ankle_left

        skeleton_edges = [
            (0, 1),   # head → shoulder_right
            (0, 2),   # head → shoulder_left
            (1, 2),   # shoulder_right ↔ shoulder_left
            (1, 3),   # right arm
            (3, 7),
            (2, 4),   # left arm
            (4, 8),
            (1, 6),   # shoulders → hips
            (2, 5),
            (5, 6),
            (6, 9),   # right leg
            (9, 11),
            (5, 10),  # left leg
            (10, 12)
        ]

        # Draw "blue" in BGR so it appears RED in yarpview (RGB)
        color = (255, 0, 0)
        thickness = 2

        # Draw limbs
        for (i, j) in skeleton_edges:
            if i < n_joints and j < n_joints:
                if conf[i] >= self.conf_thr and conf[j] >= self.conf_thr:
                    x1, y1 = int(joints[i, 0]), int(joints[i, 1])
                    x2, y2 = int(joints[j, 0]), int(joints[j, 1])
                    cv2.line(vis_img, (x1, y1), (x2, y2), color, thickness)

        # Draw joints
        for k in range(n_joints):
            if conf[k] >= self.conf_thr:
                x, y = int(joints[k, 0]), int(joints[k, 1])
                cv2.circle(vis_img, (x, y), 2, color, -1)


    # --- main periodic loop ---
    def updateModule(self):
        # Update skeleton cache (if new message arrived)
        self._update_skeleton_from_bottle()

        # Read latest image (non-blocking)
        img_in = self.img_port.read(False)
        if img_in is None:
            return True

        # Get envelope for timestamp
        self.img_port.getEnvelope(self.stamp)

        w = img_in.width()
        h = img_in.height()

        # Copy YARP mono image to numpy
        np_img = np.zeros((h, w), dtype=np.uint8)
        yarp_img = yarp.ImageMono()
        yarp_img.resize(w, h)
        yarp_img.setExternal(np_img.data, w, h)
        yarp_img.copy(img_in)

        # Convert to BGR for drawing
        vis = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)

        # Draw skeleton if available
        self._draw_skeleton(vis)

        # Send out overlay image (no color space conversion, YARP will interpret as RGB)
        out_img = self.out_port.prepare()
        out_img.resize(w, h)
        out_img.setExternal(vis.data, w, h)
        self.out_port.setEnvelope(self.stamp)
        self.out_port.write()

        self._fps_counter += 1
        now = time.time()
        if now - self._fps_t0 >= 1.0:
            hz = self._fps_counter / (now - self._fps_t0)
            print(f"[{self.getName()}] Running at {hz:.1f} Hz")
            self._fps_counter = 0
            self._fps_t0 = now

        return True


if __name__ == "__main__":
    rf = yarp.ResourceFinder()
    rf.setVerbose(False)
    rf.setDefaultContext("event-driven")
    rf.configure(sys.argv)

    module = OverlayViewerModule()
    module.runModule(rf)