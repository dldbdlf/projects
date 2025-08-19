#!/usr/bin/env python3
import rospy, json, time, math
from std_msgs.msg import String, Float64
from std_srvs.srv import Trigger, TriggerResponse


class DetectorSpeedSteerController:
    """
    distribution2.py에 angle_race.py의 '단 한 번 펄스 조향 + 평시 중립 유지' 로직을 통합한 버전.
    - 검출 결과에 따른 속도 제어(고속/저속/정지, e-stop 래치)
    - angle_race 스타일 조향: 시작 후 pulse_start 시점에 pulse_angle 라디안으로 1회만 조향,
      그 외에는 neutral_rad 유지.
    """
    def __init__(self):
        # ===== 토픽 =====
        self.result_topic = rospy.get_param("~result_topic", "/camera_infer_pub_cpu_id/inference_result")
        self.motor_topic  = rospy.get_param("~motor_topic",  "/commands/motor/speed")  # eRPM
        self.steer_topic  = rospy.get_param("~steer_topic",  "/commands/servo/unsmoothed_position")
        self.fb_topic     = rospy.get_param("~fb_topic",     "/sensors/servo_position_command")
        self.state_topic  = rospy.get_param("~state_topic",  "/control/state")  # 저빈도 상태 브로드캐스트

        # (옵션) 브레이크 전류 토픽
        self.brake_topic   = rospy.get_param("~brake_topic", "")
        self.brake_current = float(rospy.get_param("~brake_current", 5.0))  # Ampere
        self.brake_ms      = int(rospy.get_param("~brake_ms", 500))         # 적용 시간(ms)

        # e-stop 래치 사용 여부
        self.enable_estop_latch = bool(rospy.get_param("~enable_estop_latch", True))

        # ===== 클래스 ID 매핑 =====
        self.ID_NORMAL    = int(rospy.get_param("~id_normal",    0))
        self.ID_SPEEDBUMP = int(rospy.get_param("~id_speedbump", 1))
        self.ID_CRACK     = int(rospy.get_param("~id_crack",     2))
        self.ID_WATER     = int(rospy.get_param("~id_water",     3))
        self.ID_SINKHOLE  = int(rospy.get_param("~id_sinkhole",  4))

        self.label_to_id = {
            "normal": self.ID_NORMAL,
            "speedbump": self.ID_SPEEDBUMP,
            "crack": self.ID_CRACK,
            "water": self.ID_WATER,
            "sinkhole": self.ID_SINKHOLE,
        }

        # ===== 속도 제어 파라미터 =====
        self.fast_erpm = float(rospy.get_param("~fast_erpm", 170000.0))
        self.slow_erpm = float(rospy.get_param("~slow_erpm",   1550.0))
        self.stop_erpm = float(rospy.get_param("~stop_erpm",      0.0))
        self.min_conf  = float(rospy.get_param("~min_conf",       0.50))

        # ===== 퍼블리시 주기(고빈도 + 저빈도) =====
        self.pub_hz         = float(rospy.get_param("~pub_hz",          7000.0))  # VESC 타임아웃 방지
        self.slow_pub_hz    = float(rospy.get_param("~slow_pub_hz",        20.0))  # 모니터링/레코딩용
        self.enable_slowpub = bool(rospy.get_param("~enable_slow_publish",  True))
        self.msg_timeout_s  = float(rospy.get_param("~msg_timeout_s",       1.0))

        # ===== 조향(steering) 파라미터 (angle_race 스타일) =====
        self.max_steer_rad = float(rospy.get_param("~max_steer_rad", math.pi/6))  # ±30°
        self.pulse_angle   = float(rospy.get_param("~pulse_angle",    0.20))      # 1회 조향 각(라디안)
        self.pulse_start   = float(rospy.get_param("~pulse_start",    0.00))      # 시작 후 몇 초 뒤에 1회 조향
        self.neutral_rad   = float(rospy.get_param("~neutral_rad",    0.05))      # 평시 유지 각(라디안)
        self.steer_hz      = float(rospy.get_param("~steer_hz",       50.0))      # 조향 퍼블리시 주기(Hz)

        # ===== 상태 =====
        self.last_msg_time   = None
        self.target_erpm     = self.slow_erpm
        self._timeout_logged = False

        self.start_time      = time.time()
        self.pulse_done      = False
        self.feedback        = None  # 서보 피드백(0~1 스케일)
        self.last_state_text = "init"

        # e-stop 래치 상태
        self.emergency_stop      = False
        self.estop_since_ms      = 0
        self.brake_pub_started   = False

        # ===== ROS I/O =====
        self.motor_pub = rospy.Publisher(self.motor_topic, Float64, queue_size=10)
        self.steer_pub = rospy.Publisher(self.steer_topic, Float64, queue_size=10)
        self.state_pub = rospy.Publisher(self.state_topic, String,  queue_size=10)
        self.brake_pub = rospy.Publisher(self.brake_topic, Float64, queue_size=10) if self.brake_topic else None

        rospy.Subscriber(self.result_topic, String, self.result_cb, queue_size=10)
        rospy.Subscriber(self.fb_topic,     Float64, self.fb_callback, queue_size=10)

        # 서비스(e-stop 해제)
        self.srv_clear = rospy.Service("~clear_estop", Trigger, self._srv_clear_estop)

        # 타이머
        self.timer_motor = rospy.Timer(rospy.Duration(1.0/self.pub_hz),   self._tick_publish_motor)
        self.timer_steer = rospy.Timer(rospy.Duration(1.0/self.steer_hz), self._tick_publish_steer)
        if self.enable_slowpub and self.slow_pub_hz > 0:
            self.timer_slow = rospy.Timer(rospy.Duration(1.0/self.slow_pub_hz), self._tick_slow_publish)

        rospy.loginfo(f"[CTRL] 구독: {self.result_topic} (검출), {self.fb_topic} (서보피드백)")
        rospy.loginfo(f"[CTRL] 발행: {self.motor_topic} (eRPM), {self.steer_topic} (servo cmd), {self.state_topic} (state)")
        if self.brake_topic:
            rospy.loginfo(f"[CTRL] 브레이크: topic={self.brake_topic}, current={self.brake_current}A, ms={self.brake_ms}")
        rospy.loginfo(f"[CTRL] pub_hz={self.pub_hz} slow_pub_hz={self.slow_pub_hz} fast={self.fast_erpm} slow={self.slow_erpm} stop={self.stop_erpm}")
        rospy.loginfo(f"[STEER] max_steer_rad={self.max_steer_rad:.3f}, pulse={self.pulse_angle:.3f}rad @ {self.pulse_start:.1f}s, neutral={self.neutral_rad:.3f}rad, steer_hz={self.steer_hz}")
        rospy.loginfo(f"[E-STOP] enable_estop_latch={self.enable_estop_latch} (clear via ~clear_estop service)")

    # ========= 유틸 =========
    def _now_ms(self):
        return int(time.time() * 1000.0)

    def _get_conf(self, det: dict) -> float:
        v = det.get("score", det.get("confidence", 0.0))
        try:
            return float(v)
        except Exception:
            return 0.0

    def _get_id(self, det: dict) -> int:
        if "id" in det:
            try:
                return int(det["id"])
            except Exception:
                pass
        label = str(det.get("class", det.get("name", ""))).strip().lower()
        return self.label_to_id.get(label, -1)

    def _set_speed(self, erpm: float, reason: str = ""):
        # e-stop이면 어떤 경우에도 target 변경하지 않음
        if self.emergency_stop:
            return
        if self.target_erpm != erpm:
            self.target_erpm = erpm
            self.last_state_text = reason or self.last_state_text
            rospy.loginfo(f"[CTRL] 속도 변경(타깃) -> {erpm:.0f} eRPM ({self.last_state_text})")

    # ========= 서비스 =========
    def _srv_clear_estop(self, _req):
        if not self.emergency_stop:
            return TriggerResponse(success=True, message="E-STOP not active")
        self.emergency_stop    = False
        self.estop_since_ms    = 0
        self.brake_pub_started = False
        self.last_state_text   = "estop_cleared"
        rospy.logwarn("[E-STOP] cleared by service")
        return TriggerResponse(success=True, message="E-STOP cleared")

    # ========= 콜백/타이머 =========
    def fb_callback(self, msg: Float64):
        self.feedback = msg.data  # 0~1 (하드웨어 환경에 따라 다를 수 있음)

    def _tick_publish_motor(self, _evt):
        now = time.time()

        # e-stop 래치: 항상 정지 강제
        if self.emergency_stop:
            # 모터 0 eRPM
            self.motor_pub.publish(Float64(data=self.stop_erpm))
            # 브레이크 전류(옵션): e-stop 직후 ~brake_ms 동안만 퍼블리시
            if self.brake_pub is not None:
                if not self.brake_pub_started:
                    self.brake_pub_started = True
                    self.estop_since_ms = self.estop_since_ms or self._now_ms()
                elapsed_ms = self._now_ms() - self.estop_since_ms
                if elapsed_ms <= max(0, self.brake_ms):
                    self.brake_pub.publish(Float64(data=self.brake_current))
            return

        # 워치독: 메시지가 끊기면 저속 유지
        if self.last_msg_time is None or (now - self.last_msg_time) > self.msg_timeout_s:
            if self.target_erpm != self.slow_erpm:
                self._set_speed(self.slow_erpm, "timeout→slow")
            if not self._timeout_logged:
                rospy.logwarn("[CTRL] inference 메시지 타임아웃 → 저속 유지")
                self._timeout_logged = True
        else:
            self._timeout_logged = False

        # 고빈도 퍼블리시(타임아웃 방지)
        self.motor_pub.publish(Float64(data=self.target_erpm))

    def _tick_publish_steer(self, _evt):
        """
        angle_race 스타일:
          - 시작 후 pulse_start 시점에 단 1회 pulse_angle 라디안만큼 조향
          - 그 외에는 neutral_rad 유지
          - e-stop이면 완전 중앙(0.0 rad) 또는 필요시 neutral_rad
        """
        elapsed = time.time() - self.start_time

        if self.emergency_stop:
            angle_rad = 0.0  # 완전 중앙
        else:
            if (not self.pulse_done) and (elapsed >= self.pulse_start):
                angle_rad = self.pulse_angle
                self.pulse_done = True
            else:
                angle_rad = self.neutral_rad

        # 라디안 -> 0~1 스케일 변환 (중앙=0.5)
        cmd = 0.5 + (angle_rad / self.max_steer_rad) * 0.5
        cmd = max(0.0, min(1.0, cmd))
        self.steer_pub.publish(Float64(data=cmd))

        # 피드백 로그(있을 때만)
        if self.feedback is not None:
            fb_deg = (self.feedback - 0.5) * 60.0
            rospy.loginfo(f"[STEER] CMD={cmd:.3f} | FB={self.feedback:.3f} → {fb_deg:+.1f}°")
        else:
            rospy.logwarn_throttle(2.0, "[STEER] 피드백 없음")

    def _tick_slow_publish(self, _evt):
        """
        저빈도 보조 퍼블리시:
          - 동일 motor/steer 토픽에 현재 명령을 저빈도로 한 번 더 퍼블리시(간헐 구독자/레코더 대응)
          - 상태 문자열 브로드캐스트(/control/state)
        """
        # e-stop이면 정지/중립 재전송
        if self.emergency_stop:
            self.motor_pub.publish(Float64(data=self.stop_erpm))
            angle_rad = 0.0
        else:
            self.motor_pub.publish(Float64(data=self.target_erpm))
            angle_rad = self.neutral_rad if self.pulse_done else self.pulse_angle

        cmd = 0.5 + (angle_rad / self.max_steer_rad) * 0.5
        cmd = max(0.0, min(1.0, cmd))
        self.steer_pub.publish(Float64(data=cmd))

        # 상태 브로드캐스트
        state = {
            "target_erpm": round(self.stop_erpm if self.emergency_stop else self.target_erpm, 2),
            "state": "estop" if self.emergency_stop else self.last_state_text,
            "feedback": None if self.feedback is None else round(float(self.feedback), 4),
            "timestamp": time.time()
        }
        try:
            self.state_pub.publish(String(data=json.dumps(state)))
        except Exception:
            pass  # 안전

    def result_cb(self, msg: String):
        self.last_msg_time = time.time()
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[CTRL] JSON 파싱 오류: {e}")
            return

        dets_raw = data.get("detections", [])
        if not isinstance(dets_raw, list):
            dets_raw = []

        dets = [d for d in dets_raw if self._get_conf(d) >= self.min_conf]
        has_sinkhole = any(self._get_id(d) == self.ID_SINKHOLE for d in dets)
        has_slow_obj = any(self._get_id(d) in {self.ID_SPEEDBUMP, self.ID_CRACK, self.ID_WATER} for d in dets)
        has_normal   = any(self._get_id(d) == self.ID_NORMAL for d in dets)

        # ====== E-STOP: sinkhole 감지 → 즉시 정지 & 래치 ======
        if has_sinkhole:
            # 즉시 0 eRPM 퍼블리시(래치와 별도로 즉시 효과)
            self.motor_pub.publish(Float64(data=self.stop_erpm))
            self.last_state_text = "sinkhole→stop"

            if self.enable_estop_latch:
                if not self.emergency_stop:
                    self.emergency_stop    = True
                    self.estop_since_ms    = self._now_ms()
                    self.brake_pub_started = False
                    rospy.logerr("[E-STOP] sinkhole 감지 → 비상정지 래치 활성화 (모터 0 eRPM 강제)")
            else:
                rospy.logwarn("[CTRL] sinkhole 감지 → 정지 (래치 비활성화 상태)")

            return  # 추가 판단 불필요

        # e-stop 래치가 이미 걸려 있다면, 이후 입력은 무시
        if self.emergency_stop:
            return

        if has_slow_obj:
            self._set_speed(self.slow_erpm, "hazard→slow")
            rospy.loginfo("[CTRL] 위험 물체(speedbump/crack/water) 감지 → 저속")
            return

        if not dets:
            self._set_speed(self.slow_erpm, "no-detection→slow")
            rospy.loginfo("[CTRL] 검출 없음 → 저속")
            return

        # 정상일 때만 고속
        if has_normal and not (has_sinkhole or has_slow_obj):
            self._set_speed(self.fast_erpm, "normal→fast")
            rospy.loginfo("[CTRL] 정상 → 고속")
        else:
            # 미지정 라벨이지만 검출은 있는 경우 안전하게 저속
            self._set_speed(self.slow_erpm, "unknown→slow")
            rospy.loginfo("[CTRL] 미지정 라벨 검출 → 저속")


def main():
    rospy.init_node("detector_speed_steer_controller", anonymous=False)
    DetectorSpeedSteerController()
    rospy.spin()


if __name__ == "__main__":
    main()
