#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import threading
import numpy as np

import rospy
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class SinkholeCalcNode:
    """
    Node:
      - Subscribes to YOLO result JSON (/camera_infer_pub_cpu_id/inference_result)
      - Subscribes to camera image (/usb_cam/image_raw)
      - When a sinkhole bbox is present, calculates darkest-in-ROI vs darkest-outside luma difference
      - Publishes the luma_diff to /sinkhole/brightness_diff (Float32)
    """

    def __init__(self):
        rospy.init_node("sinkhole_calc_node", anonymous=True)

        # ========= params =========
        self.infer_topic = rospy.get_param("~infer_topic", "/camera_infer_pub_cpu_id/inference_result")
        self.image_topic = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.sinkhole_id = int(rospy.get_param("~sinkhole_id", 4))
        self.max_age_sec = float(rospy.get_param("~max_age_sec", 0.5))
        self.verbose     = bool(rospy.get_param("~verbose", True))

        # ========= internal =========
        self.last_det = None
        self.lock     = threading.Lock()
        self.bridge   = CvBridge()

        # ========= subscribers =========
        rospy.Subscriber(self.infer_topic, String, self.infer_cb,  queue_size=10)
        rospy.Subscriber(self.image_topic, Image, self.image_cb,   queue_size=1)

        # ========= publisher =========
        self.pub_diff = rospy.Publisher("/sinkhole/brightness_diff", Float32, queue_size=10)

        rospy.loginfo(("[INIT] sinkhole_calc_node ready\n"
                       f"       infer_topic={self.infer_topic}\n"
                       f"       image_topic={self.image_topic}\n"
                       f"       sinkhole_id={self.sinkhole_id}  max_age_sec={self.max_age_sec:.3f}"))
        rospy.spin()

    def infer_cb(self, msg: String):
        try:
            info = json.loads(msg.data.strip())
            dets = info.get("detections", [])
            chosen = None
            for det in dets:
                if int(det.get("id", -1)) == self.sinkhole_id:
                    chosen = det
                    break

            if chosen is None:
                with self.lock:
                    self.last_det = None
                if self.verbose:
                    rospy.loginfo("[DETECT] sinkhole not found; cleared last_det")
                return

            bbox = chosen.get("bbox_xywh")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                with self.lock:
                    self.last_det = None
                rospy.logwarn("[DETECT] invalid bbox_xywh: %s", str(bbox))
                return

            cx, cy, w, h = bbox
            area = float(abs(w) * abs(h))
            rospy.loginfo("[DETECT] bbox (w=%.1f, h=%.1f) -> area=%.1f", w, h, area)

            with self.lock:
                self.last_det = {
                    "time": rospy.Time.now().to_sec(),
                    "bbox_xywh": [cx, cy, w, h]
                }

        except Exception as e:
            with self.lock:
                self.last_det = None
            rospy.logwarn("[DETECT] parse failed: %s", e)

    @staticmethod
    def rgb_to_luma(rgb_arr: np.ndarray) -> np.ndarray:
        arr = rgb_arr.astype(np.float32)
        return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]

    def image_cb(self, img_msg: Image):
        with self.lock:
            ld = None if self.last_det is None else dict(self.last_det)

        if ld is None:
            return

        age = rospy.Time.now().to_sec() - ld["time"]
        if age > self.max_age_sec:
            with self.lock:
                self.last_det = None
            if self.verbose:
                rospy.loginfo("[IMG] last_det expired (age=%.3fs)", age)
            return

        try:
            # image: BGR -> RGB
            cv_bgr = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            cv_rgb = cv_bgr[..., ::-1]
            H, W, _ = cv_rgb.shape

            cx, cy, bw, bh = ld["bbox_xywh"]
            x0 = max(0, int(round(cx - bw/2)))
            x1 = min(W, int(round(cx + bw/2)))
            y0 = max(0, int(round(cy - bh/2)))
            y1 = min(H, int(round(cy + bh/2)))

            if x1 <= x0 or y1 <= y0:
                with self.lock:
                    self.last_det = None
                rospy.logwarn("[IMG] invalid ROI bounds; cleared last_det")
                return

            roi = cv_rgb[y0:y1, x0:x1, :]
            if roi.size == 0:
                with self.lock:
                    self.last_det = None
                rospy.logwarn("[IMG] empty ROI; cleared last_det")
                return

            # --- darkest pixel in ROI ---
            luma_roi = self.rgb_to_luma(roi)
            yy_in, xx_in = np.unravel_index(np.argmin(luma_roi), luma_roi.shape)
            darkest_in_rgb = roi[yy_in, xx_in, :]

            # --- darkest pixel outside ROI ---
            full_luma = self.rgb_to_luma(cv_rgb)
            mask_out = np.ones((H, W), dtype=bool)
            mask_out[y0:y1, x0:x1] = False
            masked = np.where(mask_out, full_luma, np.inf)
            y_out, x_out = np.unravel_index(np.argmin(masked), masked.shape)
            darkest_out_rgb = cv_rgb[y_out, x_out, :]

            # --- luma difference ---
            luma_in  = float(0.2126*darkest_in_rgb[0] + 0.7152*darkest_in_rgb[1] + 0.0722*darkest_in_rgb[2])
            luma_out = float(0.2126*darkest_out_rgb[0] + 0.7152*darkest_out_rgb[1] + 0.0722*darkest_out_rgb[2])
            luma_diff = float(luma_in - luma_out)

            # publish
            self.pub_diff.publish(Float32(data=luma_diff))
            rospy.loginfo("[BRIGHTNESS] Published luma_diff=%.2f", luma_diff)

            with self.lock:
                self.last_det = None

        except Exception as e:
            with self.lock:
                self.last_det = None
            rospy.logwarn("[IMG] processing failed: %s", e)


if __name__ == "__main__":
    try:
        SinkholeCalcNode()
    except rospy.ROSInterruptException:
        pass

