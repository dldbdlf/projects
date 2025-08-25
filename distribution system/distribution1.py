#!/usr/bin/env python3
import os, time, json
import rospy
import numpy as np
import cv2

from std_msgs.msg import String, Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# CPU 전용 모드: OpenCV/torch 스레드 제한
cv2.setNumThreads(1)
try:
    import torch
    torch.set_num_threads(max(1, os.cpu_count() // 2 or 1))
except Exception:
    pass

def now_sec():
    """현재 ROS 시간(sec)을 float으로 반환"""
    return rospy.Time.now().to_sec()

class CameraInferPublisherCPUOnly:
    """
    카메라 토픽(/usb_cam/image_raw) → YOLO(CPU) 추론 → 결과(JSON, Latency, Annotated Image) 퍼블리시
    - 검출 결과는 String(JSON)으로 발행
    - 레이턴시는 Float64(ms)로 발행
    - 주석 이미지(BGR)는 옵션에 따라 퍼블리시
    - 종료 시까지 누적된 전체 결과를 한 번에 저장(.txt JSON 배열)
    """

    def __init__(self):
        # ===== ROS 파라미터 =====
        self.image_topic   = rospy.get_param("~image_topic", "/usb_cam/image_raw")
        self.model_path    = rospy.get_param("~model",       "/home/wego/models/best_ep95.pt")
        self.conf_thres    = float(rospy.get_param("~conf",  0.25))   # Confidence threshold
        self.iou_thres     = float(rospy.get_param("~iou",   0.50))   # IoU threshold
        self.imgsz         = int(rospy.get_param("~imgsz",   640))    # 입력 이미지 크기
        self.pub_annotated = bool(rospy.get_param("~pub_annotated", False))  # 어노테이션 이미지 퍼블리시 여부
        self.bgr_to_rgb    = bool(rospy.get_param("~bgr_to_rgb", True))      # BGR→RGB 변환 여부

        # (선택) 카메라 타임스탬프 토픽 (레이턴시 측정용)
        self.ts_topic      = rospy.get_param("~timestamp_topic", "")
        self.last_cam_ts   = None

        # ===== 결과 저장 =====
        self.all_results   = []   # 누적된 모든 결과 저장
        self.save_txt_path = rospy.get_param("~save_txt_path", "/home/wego/logs/infer_all.txt")
        try:
            os.makedirs(os.path.dirname(self.save_txt_path), exist_ok=True)
        except Exception:
            pass

        # ===== 이미지 저장 옵션 =====
        self.save_image_dir          = rospy.get_param("~save_image_dir", "/home/wego/bbox")
        self.save_only_when_detected = bool(rospy.get_param("~save_only_when_detected", True)) # 검출 있을 때만 저장
        self.save_image_ext          = rospy.get_param("~save_image_ext", "jpg").lower()
        self.save_image_quality      = int(rospy.get_param("~save_image_quality", 90))
        self.save_min_interval_ms    = int(rospy.get_param("~save_min_interval_ms", 0))  # 저장 주기 제한(ms)
        self._last_save_ms           = 0
        self._save_counter           = 0

        if self.save_image_dir:
            os.makedirs(self.save_image_dir, exist_ok=True)

        # ===== YOLO 모델 로드 (CPU 고정) =====
        rospy.loginfo(f"[INIT] Loading model on CPU: {self.model_path}")
        self.model = YOLO(self.model_path)

        # ===== 퍼블리셔 =====
        self.pub_result   = rospy.Publisher("~inference_result", String,  queue_size=10)
        self.pub_latency  = rospy.Publisher("~latency_ms",       Float64, queue_size=10)
        self.pub_img      = rospy.Publisher("~image",            Image,   queue_size=1) if self.pub_annotated else None

        # ===== 서브스크라이버 =====
        self.bridge = CvBridge()
        rospy.Subscriber(self.image_topic, Image, self.image_cb, queue_size=1, buff_size=2**24)
        if self.ts_topic:  # 카메라 timestamp 토픽 구독
            rospy.Subscriber(self.ts_topic, Float64, self.time_cb, queue_size=1)

        # 종료 시 전체 결과 저장
        rospy.on_shutdown(self._flush_results)

        rospy.loginfo(f"[READY] CPU inference | Subscribed: {self.image_topic} -> Publishing: {self.pub_result.resolved_name}")
        rospy.loginfo(f"[SAVE] 종료 시 전체 결과를 한 번에 저장합니다: {self.save_txt_path}")
        if self.save_image_dir:
            rospy.loginfo(f"[SAVE-IMG] dir={self.save_image_dir}, only_when_detected={self.save_only_when_detected}, "
                          f"ext={self.save_image_ext}, quality={self.save_image_quality}, interval={self.save_min_interval_ms}ms")

    # ===== 종료 시 결과 저장 =====
    def _flush_results(self):
        """종료 시, 누적된 전체 결과(JSON)를 txt에 저장"""
        try:
            with open(self.save_txt_path, "w", encoding="utf-8") as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)
            rospy.loginfo(f"[SAVE] 모든 결과를 {self.save_txt_path} 에 저장 완료 (총 {len(self.all_results)} 건)")
        except Exception as e:
            rospy.logerr(f"[SAVE] 저장 실패: {e}")

    # ===== 콜백 =====
    def time_cb(self, msg: Float64):
        """카메라 timestamp 토픽 수신 → 레이턴시 계산용"""
        self.last_cam_ts = float(msg.data)

    def _should_save_image_now(self, has_detection: bool) -> bool:
        """이미지 저장 조건 체크"""
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
        """주석된 이미지 디스크 저장"""
        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self._save_counter += 1
        fn = f"frame_{ts}_{self._save_counter:06d}.{self.save_image_ext}"
        path = os.path.join(self.save_image_dir, fn)

        params = []
        if self.save_image_ext in ("jpg", "jpeg"):
            q = int(np.clip(self.save_image_quality, 1, 100))
            params = [cv2.IMWRITE_JPEG_QUALITY, q]
        elif self.save_image_ext == "png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # 기본 압축률

        try:
            ok = cv2.imwrite(path, img_bgr, params)
            if ok:
                self._last_save_ms = int(time.time() * 1000.0)
                rospy.loginfo(f"[SAVE-IMG] {path}")
            else:
                rospy.logwarn(f"[SAVE-IMG] 저장 실패: {path}")
        except Exception as e:
            rospy.logwarn(f"[SAVE-IMG] 예외 발생: {e}")

    def image_cb(self, msg: Image):
        """카메라 이미지 수신 시 → YOLO 추론 + 결과 퍼블리시"""
        t0 = now_sec()

        # ROS Image → OpenCV
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"[CV_BRIDGE] 변환 실패: {e}")
            return

        # BGR→RGB 변환 (YOLO는 RGB 입력 권장)
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if self.bgr_to_rgb else frame_bgr

        # ===== YOLO 추론 (CPU 강제) =====
        t_infer0 = time.time()
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            verbose=False,
            device="cpu"
        )
        t_infer1 = time.time()

        det_list = []  # 검출 결과 리스트
        need_annotation_image = self.pub_annotated or bool(self.save_image_dir)
        annotated = frame_bgr.copy() if need_annotation_image else frame_bgr

        # 추론 결과 파싱
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                k_int = int(np.rint(k))
                w, h = float(x2 - x1), float(y2 - y1)
                u, v = float(x1 + w/2.0), float(y1 + h/2.0)

                # JSON용 결과
                det_list.append({
                    "id": k_int,
                    "score": float(c),
                    "bbox_xywh": [float(x1), float(y1), w, h],
                    "center_uv": [u, v]
                })

                # 시각화 (박스+라벨) → confidence 제거
                if need_annotation_image:
                    cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(annotated, f"id={k_int}",
                                (int(x1), max(int(y1)-6, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)

        # 최종 payload (JSON)
        payload = {
            "stamp_sec": msg.header.stamp.to_sec() if msg.header.stamp else None,
            "recv_sec": t0,
            "backend": "ultralytics-yolo-cpu",
            "img_size": [int(frame_bgr.shape[1]), int(frame_bgr.shape[0])],  # [W,H]
            "conf": self.conf_thres,
            "iou": self.iou_thres,
            "detections": det_list
        }

        # ===== 퍼블리시 =====
        self.pub_result.publish(String(data=json.dumps(payload, ensure_ascii=False)))

        # 레이턴시(ms)
        if self.last_cam_ts is not None:
            latency_ms = (now_sec() - self.last_cam_ts) * 1000.0
        else:
            latency_ms = (t_infer1 - t_infer0) * 1000.0
        self.pub_latency.publish(Float64(data=float(latency_ms)))

        # 어노테이션 이미지 퍼블리시
        if self.pub_annotated and self.pub_img and self.pub_img.get_num_connections() > 0:
            try:
                self.pub_img.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
            except Exception as e:
                rospy.logwarn(f"[PUB IMG] {e}")

        # ===== 이미지 저장 =====
        has_detection = len(det_list) > 0
        if self._should_save_image_now(has_detection):
            self._save_annotated_image(annotated)

        # ===== 결과 누적 =====
        self.all_results.append(payload)

        # ===== OpenCV 윈도우 출력 =====
        try:
            cv2.imshow("YOLO Detection (CPU)", annotated)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logwarn(f"[SHOW IMG] {e}")

def main():
    rospy.init_node("camera_infer_pub_cpu_id", anonymous=False)
    node = CameraInferPublisherCPUOnly()
    rospy.loginfo("camera_infer_pub_cpu_id started (CPU only, id-only output, save-on-exit).")
    rospy.spin()

if __name__ == "__main__":
    main()
