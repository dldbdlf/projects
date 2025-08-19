#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distribution4.py
- distribution2.py가 퍼블리시한 YOLO 결과(JSON)를 구독.
- normal/speedbump/crack/water 모두 미검출일 때만 Coral(Edge TPU) 이진 분류로 sinkhole 여부 판정.
- yes / no / yet 를 std_msgs/String 으로 Publish.
- 추론을 수행한 경우 이미지를 저장.

의존:
  pip install pycoral
  sudo apt-get install ros-<distro>-cv-bridge
"""

import os
import math
import json
import time
import threading
from datetime import datetime

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import cv2

# Coral
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common


class SinkholePostProcessorV4:
    def __init__(self):
        # 노드명: distribution4 (혼동 방지)
        rospy.init_node("distribution4_sinkhole_postproc", anonymous=True)

        # ====== 파라미터 ======
        # 입력 토픽(기본: distribution2가 퍼블리시한다고 가정)
        self.result_topic   = rospy.get_param("~result_topic", "/camera_infer_pub_cpu_id/inference_result")
        self.image_topic    = rospy.get_param("~image_topic",  "/usb_cam/image_raw")
        # 출력 토픽
        self.decision_topic = rospy.get_param("~decision_topic", "/sinkhole/decision")

        # YOLO 클래스 ID 매핑(launch에서 덮어쓰기 가능)
        self.ID_NORMAL    = int(rospy.get_param("~id_normal",    0))
        self.ID_SPEEDBUMP = int(rospy.get_param("~id_speedbump", 1))
        self.ID_CRACK     = int(rospy.get_param("~id_crack",     2))
        self.ID_WATER     = int(rospy.get_param("~id_water",     3))
        self.ID_SINKHOLE  = int(rospy.get_param("~id_sinkhole",  4))

        # YOLO 판정 임계값
        self.yolo_score_th = float(rospy.get_param("~yolo_score_th", 0.25))

        # Coral(TPU) 모델
        self.tpu_model     = rospy.get_param("~tpu_model", "/home/ubuntu/my_model_edgetpu.tflite")
        self.bgr_to_rgb    = bool(rospy.get_param("~bgr_to_rgb", True))
        self.tpu_yes_th    = float(rospy.get_param("~tpu_yes_th", 0.5))  # sinkhole 확률 임계값
        self.tpu_sink_idx  = int(rospy.get_param("~tpu_sink_idx", 1))    # 2채널일 때 sinkhole 채널 index (보통 1)

        # TPU 호출 제어
        self.tpu_cooldown_ms = int(rospy.get_param("~tpu_cooldown_ms", 120))  # 과도호출 방지

        # 저장 경로
        self.save_images  = bool(rospy.get_param("~save_images", True))
        self.save_on      = rospy.get_param("~save_on", "all")  # "all"|"yes"|"no"
        self.save_dir     = rospy.get_param("~save_dir", "/dev/shm/sinkhole_snaps")
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 90))

        os.makedirs(os.path.join(self.save_dir, "yes"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "no"), exist_ok=True)

        # ====== 내부 상태 ======
        self.bridge = CvBridge()
        self.last_image = None
        self.last_image_stamp = None
        self.last_tpu_run_stamp = None
        self.lock = threading.Lock()

        # Coral 로드
        self._load_tpu(self.tpu_model)

        # 퍼블리셔 (latched)
        self.pub_decision = rospy.Publisher(self.decision_topic, String, queue_size=1, latch=True)

        # 최초 상태는 'yet'
        self._publish_decision("yet")

        # 구독 시작
        rospy.Subscriber(self.image_topic, Image, self._on_image, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.result_topic, String, self._on_result, queue_size=10)

        rospy.loginfo("[INIT][distribution4] ready. from=%s, image=%s → to=%s",
                      self.result_topic, self.image_topic, self.decision_topic)
        rospy.spin()

    # ================= Coral =================
    def _load_tpu(self, model_path):
        if not os.path.exists(model_path):
            rospy.logwarn("[TPU] 모델 경로가 존재하지 않습니다: %s", model_path)
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        w, h = common.input_size(self.interpreter)  # (width,height)
        self.tpu_in_w, self.tpu_in_h = int(w), int(h)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        rospy.loginfo("[TPU] Loaded: %s | input=%dx%d dtype=%s",
                      model_path, self.tpu_in_w, self.tpu_in_h, self.input_details[0]['dtype'].__name__)

    def _run_tpu_binary(self, bgr_img):
        """Coral 이진 분류: sinkhole 확률(0~1) 반환"""
        if bgr_img is None:
            return None

        img = cv2.resize(bgr_img, (self.tpu_in_w, self.tpu_in_h), interpolation=cv2.INTER_AREA)
        if self.bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        in_dtype = self.input_details[0]['dtype']
        if in_dtype == np.float32:
            inp = img.astype(np.float32) / 255.0
        else:
            inp = img.astype(np.uint8)

        inp = np.expand_dims(inp, axis=0)
        common.set_input(self.interpreter, inp)
        self.interpreter.invoke()

        out = common.output_tensor(self.interpreter, 0).squeeze()
        od = self.output_details[0]
        if 'quantization' in od and od['quantization'] != (0.0, 0):
            scale, zp = od['quantization']
            out = scale * (out.astype(np.float32) - zp)

        if out.ndim == 1 and out.size == 2:
            m = np.max(out)
            e = np.exp(out - m)
            probs = e / np.sum(e)
            return float(probs[self.tpu_sink_idx])

        if np.isscalar(out) or out.size == 1:
            x = float(out if np.isscalar(out) else out.ravel()[0])
            return 1.0 / (1.0 + math.exp(-x))

        return float(np.clip(np.mean(out), 0.0, 1.0))

    # ================= ROS 콜백 =================
    def _on_image(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn("[IMG] cv_bridge 변환 실패: %s", str(e))
            return
        with self.lock:
            self.last_image = bgr
            self.last_image_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def _on_result(self, msg):
        """
        distribution2.py가 퍼블리시하는 YOLO JSON 예시:
        {
          "stamp_sec": 1755415364.77,
          "recv_sec": 1755415365.17,
          "backend": "ultralytics-yolo-cpu",
          "img_size": [1280,720],
          "conf": 0.25,
          "iou": 0.5,
          "detections": [
            {"id":4, "score":0.50, "bbox_xywh":[...], "center_uv":[...]}
          ]
        }
        """
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn("[JSON] 파싱 실패: %s", str(e))
            return

        detections = data.get("detections", [])

        # 임계값 이상 클래스 집합
        pos_ids = set()
        for det in detections:
            try:
                cid = int(det.get("id", -1))
                sc  = float(det.get("score", 0.0))
            except Exception:
                continue
            if sc >= self.yolo_score_th:
                pos_ids.add(cid)

        # 1) YOLO가 sinkhole을 직접 검출 → 즉시 yes
        if self.ID_SINKHOLE in pos_ids:
            self._finalize("yes")
            return

        # 2) 4종(normal/speedbump/crack/water) 중 하나라도 검출 → no
        base_ids = {self.ID_NORMAL, self.ID_SPEEDBUMP, self.ID_CRACK, self.ID_WATER}
        if len(pos_ids.intersection(base_ids)) > 0:
            self._finalize("no")
            return

        # 3) 4종 모두 미검출 → Coral로 2차 판정
        with self.lock:
            bgr = None if self.last_image is None else self.last_image.copy()
            img_stamp = self.last_image_stamp

        if bgr is None:
            # 아직 이미지가 없으면 'yet'
            self._publish_decision("yet")
            return

        # 중복 TPU 호출 간격 제한
        now = rospy.Time.now()
        if self.last_tpu_run_stamp is not None:
            dt_ms = (now - self.last_tpu_run_stamp).to_sec() * 1000.0
            if dt_ms < self.tpu_cooldown_ms:
                return
        self.last_tpu_run_stamp = now

        sink_p = self._run_tpu_binary(bgr)
        if sink_p is None:
            self._publish_decision("yet")
            return

        decision = "yes" if sink_p >= self.tpu_yes_th else "no"
        self._finalize(decision, bgr=bgr, stamp=img_stamp, prob=sink_p)

    # ================= 유틸 =================
    def _finalize(self, decision, bgr=None, stamp=None, prob=None):
        """결정 publish + 이미지 저장"""
        self._publish_decision(decision)

        if self.save_images and decision in ("yes", "no"):
            if self.save_on == "yes" and decision != "yes":
                return
            if self.save_on == "no" and decision != "no":
                return
            if bgr is None:
                with self.lock:
                    bgr = None if self.last_image is None else self.last_image.copy()
                    stamp = self.last_image_stamp if stamp is None else stamp
            if bgr is not None:
                ts = stamp.to_sec() if stamp is not None else time.time()
                ts_str = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S_%f")
                subdir = "yes" if decision == "yes" else "no"
                fn = f"{decision}_{ts_str}"
                if prob is not None:
                    fn += f"_p{prob:.2f}"
                path = os.path.join(self.save_dir, subdir, fn + ".jpg")
                cv2.imwrite(path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)])
                rospy.loginfo("[SAVE] %s", path)

    def _publish_decision(self, val):
        self.pub_decision.publish(String(data=val))
        rospy.loginfo("[DECISION] %s", val)


if __name__ == "__main__":
    try:
        SinkholePostProcessorV4()
    except rospy.ROSInterruptException:
        pass
