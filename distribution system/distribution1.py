#!/usr/bin/env python3
import os, time, json
import rospy
import numpy as np
import cv2

from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# CPU 고정 & 쓰레드 제한
cv2.setNumThreads(1)
try:
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2 or 1))
except Exception:
    pass

def now_sec():
    return rospy.Time.now().to_sec()

class CameraInferPublisherCPUOnly:
    def __init__(self):
        # ===== ROS 파라미터 =====
        self.image_topic   = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.model_path    = rospy.get_param("~model",       "/home/wego/models/best_ep95.pt")
        self.conf_thres    = float(rospy.get_param("~conf",  0.25))
        self.iou_thres     = float(rospy.get_param("~iou",   0.50))
        self.imgsz         = int(rospy.get_param("~imgsz",   640))
        self.pub_annotated = bool(rospy.get_param("~pub_annotated", False))  # 지연 최소화를 위해 기본 False
        self.bgr_to_rgb    = bool(rospy.get_param("~bgr_to_rgb", True))

        # (선택) 카메라 발행 타임스탬프가 따로 있을 때 레이턴시 측정용
        self.ts_topic      = rospy.get_param("~timestamp_topic", "")
        self.last_cam_ts   = None

        # ===== 전체 결과 누적 & 저장 경로 =====
        self.all_results   = []  # 종료 시 한 번에 저장
        self.save_txt_path = rospy.get_param("~save_txt_path", "/home/wego/logs/infer_all.txt")
        try:
            os.makedirs(os.path.dirname(self.save_txt_path), exist_ok=True)
        except Exception:
            pass

        # ===== 이미지 저장 옵션 =====
        self.save_image_dir            = rospy.get_param("~save_image_dir", "/home/wego/bbox/frame_YYYYMMDD_HHMMSS_000001.jpg")   # 비어있으면 저장 안함
        self.save_only_when_detected   = bool(rospy.get_param("~save_only_when_detected", True))
        self.save_image_ext            = rospy.get_param("~save_image_ext", "jpg").lower()  # 'jpg' or 'png'
        self.save_image_quality        = int(rospy.get_param("~save_image_quality", 90))    # 1~100
        self.save_min_interval_ms      = int(rospy.get_param("~save_min_interval_ms", 0))   # 0이면 제한 없음
        self._last_save_ms             = 0
        self._save_counter             = 0

        if self.save_image_dir:
            os.makedirs(self.save_image_dir, exist_ok=True)

        rospy.loginfo(f"[INIT] Loading model on CPU: {self.model_path}")
        self.model = YOLO(self.model_path)

        # 퍼블리셔
        self.pub_result   = rospy.Publisher("~inference_result", String,  queue_size=10)
        self.pub_latency  = rospy.Publisher("~latency_ms",       Float64, queue_size=10)
        self.pub_img      = rospy.Publisher("~image",            Image,   queue_size=1) if self.pub_annotated else None

        # 서브스크라이버
        self.bridge = CvBridge()
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)
        if self.ts_topic:
            rospy.Subscriber(self.ts_topic, Float64, self.time_cb, queue_size=1)

        # 종료 시 전체 결과 저장 훅
        rospy.on_shutdown(self._flush_results)

        rospy.loginfo("[READY] CPU inference | Subscribed: %s -> Publishing: %s"
                      % (self.image_topic, self.pub_result.resolved_name))
        rospy.loginfo(f"[SAVE] 종료 시 전체 결과를 한 번에 저장합니다: {self.save_txt_path}")
        if self.save_image_dir:
            rospy.loginfo(f"[SAVE-IMG] 주석 이미지를 저장합니다 → dir={self.save_image_dir}, "
                          f"only_when_detected={self.save_only_when_detected}, ext={self.save_image_ext}, "
                          f"quality={self.save_image_quality}, min_interval_ms={self.save_min_interval_ms}")

    # -------- 종료 시 저장 --------
    def _flush_results(self):
        """종료될 때까지 모은 결과를 한 번에 저장(.txt = JSON 배열 pretty-print)"""
        try:
            with open(self.save_txt_path, "w", encoding="utf-8") as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)
            rospy.loginfo(f"[SAVE] 모든 결과를 {self.save_txt_path} 에 저장 완료 (총 {len(self.all_results)} 건)")
        except Exception as e:
            rospy.logerr(f"[SAVE] 저장 실패: {e}")

    # -------- 콜백들 --------
    def time_cb(self, msg: Float64):
        self.last_cam_ts = float(msg.data)

    def _should_save_image_now(self, has_detection: bool) -> bool:
        """이미지 저장 조건 체크: 디렉터리 설정 + (검출조건) + 저장 간격"""
        if not self.save_image_dir:
            return False
        if self.save_only_when_detected and not has_detection:
            return False
        if self.save_min_interval_ms > 0:
            now_ms = int(time.time() * 1000.0)
            if now_ms - self._last_save_ms < self.save_min_interval_ms:
                return False
        return True

    def _save_annotated_image(self, img_bgr):
        """주석 이미지 디스크 저장"""
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self._save_counter += 1
        fn = f"frame_{ts}_{self._save_counter:06d}.{self.save_image_ext}"
        path = os.path.join(self.save_image_dir, fn)

        params = []
        ext = self.save_image_ext
        if ext in ("jpg", "jpeg"):
            q = int(np.clip(self.save_image_quality, 1, 100))
            params = [cv2.IMWRITE_JPEG_QUALITY, q]
        elif ext == "png":
            # 0(무압축)~9(최대압축), 품질과 반비례(속도 중시 시 낮게)
            comp = 3
            params = [cv2.IMWRITE_PNG_COMPRESSION, comp]

        try:
            ok = cv2.imwrite(path, img_bgr, params)
            if ok:
                self._last_save_ms = int(time.time() * 1000.0)
                rospy.loginfo(f"[SAVE-IMG] {path}")
            else:
                rospy.logwarn(f"[SAVE-IMG] 저장 실패(알 수 없는 오류): {path}")
        except Exception as e:
            rospy.logwarn(f"[SAVE-IMG] 예외 발생: {e}")

    def image_cb(self, msg: Image):
        t0 = now_sec()
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"[CV_BRIDGE] {e}")
            return

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if self.bgr_to_rgb else frame_bgr

        # ===== YOLO 추론 (CPU 고정) =====
        t_infer0 = time.time()
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            device="cpu"          # ← 강제 CPU
        )
        t_infer1 = time.time()

        det_list = []
        # 저장이 활성화되었거나 퍼블리시가 켜져 있으면 복사본에 주석을 그린다.
        need_annotation_image = self.pub_annotated or bool(self.save_image_dir)
        annotated = frame_bgr.copy() if need_annotation_image else frame_bgr

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                k_int = int(np.rint(k))  # 정수 id
                w, h = float(x2 - x1), float(y2 - y1)
                u, v = float(x1 + w/2.0), float(y1 + h/2.0)

                det_list.append({
                    "id": k_int,
                    "score": float(c),
                    "bbox_xywh": [float(x1), float(y1), w, h],
                    "center_uv": [u, v]
                })

                if need_annotation_image:
                    # 라벨/점수 함께 주석
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(annotated, f"id={k_int} conf={c:.2f}",
                                (int(x1), max(int(y1)-6, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

        payload = {
            "stamp_sec": msg.header.stamp.to_sec() if msg.header.stamp else None,
            "recv_sec": t0,
            "backend": "ultralytics-yolo-cpu",
            "img_size": [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])],  # [W,H]
            "conf": self.conf_thres,
            "iou": self.iou_thres,
            "detections": det_list
        }

        # 퍼블리시
        self.pub_result.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        # 레이턴시 퍼블리시
        if self.last_cam_ts is not None:
            latency_ms = (now_sec() - self.last_cam_ts) * 1000.0
        else:
            latency_ms = (t_infer1 - t_infer0) * 1000.0
        self.pub_latency.publish(Float64(data=float(latency_ms)))

        # 어노테이션 이미지 퍼블리시(옵션)
        if self.pub_annotated and self.pub_img and self.pub_img.get_num_connections() > 0:
            try:
                self.pub_img.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
            except Exception as e:
                rospy.logwarn(f"[PUB IMG] {e}")

        # ===== 이미지 저장 =====
        has_detection = len(det_list) > 0
        if self._should_save_image_now(has_detection):
            # 검출 있을 때만 저장(True) or 항상 저장(False)
            # 저장 이미지는 항상 주석(박스/라벨) 포함
            self._save_annotated_image(annotated)

        # ===== 결과 누적(종료 시 한 번에 저장) =====
        self.all_results.append(payload)

def main():
    rospy.init_node("camera_infer_pub_cpu_id", anonymous=False)
    node = CameraInferPublisherCPUOnly()
    rospy.loginfo("camera_infer_pub_cpu_id started (CPU only, id-only output, save-on-exit).")
    rospy.spin()

if __name__ == "__main__":
    main()
