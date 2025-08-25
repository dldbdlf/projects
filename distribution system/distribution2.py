#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy, json, time, math
from std_msgs.msg import String, Float32, Float64
from std_srvs.srv import Trigger, TriggerResponse

class DetectorSpeedSteerController:
    """
    YOLO 결과 + 보조 지표(밝기차, water decision)로 속도/조향을 제어하는 노드.
    - YOLO: sinkhole → 정지(E-STOP 래치), water/speedbump/crack → 감속, normal → 가속
    - 밝기차(/sinkhole/brightness_diff): 0~1 구간일 때 2초간 후진, 1.0 이상이면 정지
    - /water/decision: "yes" 수신 시 YOLO 결과와 무관하게 감속(slow_erpm) 강제
    - angle_race 스타일: 시작 후 1회 펄스 조향, 그 외엔 중립 유지
    """

    def __init__(self):
        # ====== 파라미터 ======
        self.result_topic = rospy.get_param("~result_topic", "/camera_infer_pub_cpu_id/inference_result")
        self.motor_topic  = rospy.get_param("~motor_topic",  "/commands/motor/speed")  # eRPM
        self.steer_topic  = rospy.get_param("~steer_topic",  "/commands/servo/unsmoothed_position")
        self.fb_topic     = rospy.get_param("~fb_topic",     "/sensors/servo_position_command")
        self.state_topic  = rospy.get_param("~state_topic",  "/control/state")

        # (옵션) 브레이크 전류 토픽
        self.brake_topic   = rospy.get_param("~brake_topic", "")
        self.brake_current = float(rospy.get_param("~brake_current", 5.0))   # A
        self.brake_ms      = int(rospy.get_param("~brake_ms", 500))          # ms
        self.enable_estop_latch = bool(rospy.get_param("~enable_estop_latch", True))

        # 클래스 ID 매핑
        self.ID_NORMAL    = int(rospy.get_param("~id_normal",    0))
        self.ID_SPEEDBUMP = int(rospy.get_param("~id_speedbump", 1))
        self.ID_CRACK     = int(rospy.get_param("~id_crack",     2))
        self.ID_WATER     = int(rospy.get_param("~id_water",     3))
        self.ID_SINKHOLE  = int(rospy.get_param("~id_sinkhole",  4))

        # 속도/임계값
        self.fast_erpm    = float(rospy.get_param("~fast_erpm", 170000.0))
        self.slow_erpm    = float(rospy.get_param("~slow_erpm",   1550.0))
        self.stop_erpm    = float(rospy.get_param("~stop_erpm",      0.0))
        self.reverse_erpm = float(rospy.get_param("~reverse_erpm", -17000.0))
        self.min_conf     = float(rospy.get_param("~min_conf",       0.50))

        # 퍼블리시 주기
        self.pub_hz         = float(rospy.get_param("~pub_hz",          7000.0))  # VESC 타임아웃 방지
        self.slow_pub_hz    = float(rospy.get_param("~slow_pub_hz",        20.0))  # 저빈도 상태 브로드캐스트
        self.enable_slowpub = bool(rospy.get_param("~enable_slow_publish",  True))
        self.msg_timeout_s  = float(rospy.get_param("~msg_timeout_s",       1.0))

        # 조향 파라미터 (angle_race 스타일)
        self.max_steer_rad = float(rospy.get_param("~max_steer_rad", math.pi/6))  # ±30°
        self.pulse_angle   = float(rospy.get_param("~pulse_angle",    0.20))
        self.pulse_start   = float(rospy.get_param("~pulse_start",    0.00))
        self.neutral_rad   = float(rospy.get_param("~neutral_rad",    0.05))
        self.steer_hz      = float(rospy.get_param("~steer_hz",       50.0))

        # ====== 상태 변수 ======
        self.last_msg_time   = None
        self.target_erpm     = self.slow_erpm
        self.start_time      = time.time()
        self.pulse_done      = False
        self.feedback        = None  # 서보 피드백(0~1 스케일 등 환경 따라 다를 수 있음)
        self.last_state_text = "init"

        # e-stop 래치
        self.emergency_stop    = False
        self.estop_since_ms    = 0
        self.brake_pub_started = False

        # 밝기차/후진 타이머
        self.brightness_diff     = 0.0
        self.reverse_start_time  = None

        # water decision 상태
        self.water_decision = ""

        # ====== ROS Pub/Sub ======
        self.motor_pub = rospy.Publisher(self.motor_topic, Float64, queue_size=10)
        self.steer_pub = rospy.Publisher(self.steer_topic, Float64, queue_size=10)
        self.state_pub = rospy.Publisher(self.state_topic, String,  queue_size=10)
        self.brake_pub = rospy.Publisher(self.brake_topic, Float64, queue_size=10) if self.brake_topic else None

        rospy.Subscriber(self.result_topic, String, self.result_cb, queue_size=10)
        rospy.Subscriber(self.fb_topic,     Float32, self.fb_callback, queue_size=10)
        rospy.Subscriber("/sinkhole/brightness_diff", Float32, self.brightness_cb, queue_size=10)

        # 🔽 요청 사항: water decision 한 가지만 구독
        rospy.Subscriber("/water/decision", String, self.water_decision_cb, queue_size=10)

        # 서비스(e-stop 해제)
        self.srv_clear = rospy.Service("~clear_estop", Trigger, self._srv_clear_estop)

        # 타이머
        self.timer_motor = rospy.Timer(rospy.Duration(1.0/self.pub_hz),   self._tick_publish_motor)
        self.timer_steer = rospy.Timer(rospy.Duration(1.0/self.steer_hz), self._tick_publish_steer)
        if self.enable_slowpub and self.slow_pub_hz > 0:
            self.timer_slow = rospy.Timer(rospy.Duration(1.0/self.slow_pub_hz), self._tick_slow_publish)

        rospy.loginfo(f"[CTRL] Sub: result={self.result_topic}, fb={self.fb_topic}, bright=/sinkhole/brightness_diff, water=/water/decision")
        rospy.loginfo(f"[CTRL] Pub: motor={self.motor_topic}, steer={self.steer_topic}, state={self.state_topic}")
        rospy.loginfo(f"[CTRL] fast={self.fast_erpm} slow={self.slow_erpm} stop={self.stop_erpm} reverse={self.reverse_erpm}, min_conf={self.min_conf}")

    # ========= 유틸 =========
    def _now_ms(self):
        return int(time.time() * 1000.0)

    # ========= 콜백 =========
    def fb_callback(self, msg: Float32):
        self.feedback = msg.data

    def brightness_cb(self, msg: Float32):
        self.brightness_diff = msg.data

    def water_decision_cb(self, msg: String):
        """ /water/decision: yes/no/yet 수신 """
        self.water_decision = (msg.data or "").strip().lower()

    def result_cb(self, msg: String):
        """ YOLO 결과(JSON)를 수신하여 기본 속도 정책을 결정 """
        self.last_msg_time = time.time()
        try:
            data = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[CTRL] JSON 파싱 오류: {e}")
            return

        dets_raw = data.get("detections", [])
        if not isinstance(dets_raw, list):
            dets_raw = []

        # 스코어 필터
        dets = [d for d in dets_raw if float(d.get("score", 0.0)) >= self.min_conf]
        get_id = lambda d: int(d.get("id", -1))

        has_sinkhole = any(get_id(d) == self.ID_SINKHOLE for d in dets)
        has_slow_obj = any(get_id(d) in {self.ID_SPEEDBUMP, self.ID_CRACK, self.ID_WATER} for d in dets)
        has_normal   = any(get_id(d) == self.ID_NORMAL for d in dets)

        # sinkhole → 즉시 정지 & (옵션) 래치
        if has_sinkhole:
            self.motor_pub.publish(Float64(data=self.stop_erpm))
            self.last_state_text = "sinkhole→stop"
            if self.enable_estop_latch and not self.emergency_stop:
                self.emergency_stop    = True
                self.estop_since_ms    = self._now_ms()
                self.brake_pub_started = False
                rospy.logerr("[E-STOP] sinkhole 감지 → 비상정지 래치")
            return

        # 이미 e-stop이면 무시
        if self.emergency_stop:
            return

        # === 기본 속도 정책 (YOLO) ===
        if has_slow_obj:
            self.target_erpm = self.slow_erpm
            self.last_state_text = "hazard→slow"
        elif has_normal:
            self.target_erpm = self.fast_erpm
            self.last_state_text = "normal→fast"
        else:
            self.target_erpm = self.slow_erpm
            self.last_state_text = "unknown→slow"

        # === water decision 오버라이드 ===
        # "yes"면 YOLO 결과와 무관하게 감속 유지
        if self.water_decision == "yes":
            self.target_erpm = self.slow_erpm
            self.last_state_text = "water→slow"

    # ========= 퍼블리시 타이머 =========
    def _tick_publish_motor(self, _evt):
        now = time.time()

        # e-stop 래치: 강제 정지 & (옵션) 브레이크 전류 퍼블리시
        if self.emergency_stop:
            self.motor_pub.publish(Float64(data=self.stop_erpm))
            if self.brake_pub is not None:
                if not self.brake_pub_started:
                    self.brake_pub_started = True
                    self.estop_since_ms = self.estop_since_ms or self._now_ms()
                elapsed_ms = self._now_ms() - self.estop_since_ms
                if elapsed_ms <= max(0, self.brake_ms):
                    self.brake_pub.publish(Float64(data=self.brake_current))
            return

        # 밝기차: 0 < diff < 1.0 → 2초 후진
        if 0.0 < self.brightness_diff < 10.0:
            if self.reverse_start_time is None:
                self.reverse_start_time = now
            if (now - self.reverse_start_time) < 2.0:
                self.motor_pub.publish(Float64(data=self.reverse_erpm))
                self.last_state_text = "brightness→reverse"
                return
            else:
                self.reverse_start_time = None

        # 밝기차: >= 1.0 → 정지
        if self.brightness_diff >= 10.0:
            self.motor_pub.publish(Float64(data=self.stop_erpm))
            self.last_state_text = "brightness→stop"
            return

        # 메시지 타임아웃 → 저속
        if self.last_msg_time is None or (now - self.last_msg_time) > self.msg_timeout_s:
            self.target_erpm = self.slow_erpm
            self.last_state_text = "timeout→slow"

        # 최종 속도 퍼블리시
        self.motor_pub.publish(Float64(data=self.target_erpm))

    def _tick_publish_steer(self, _evt):
        """ angle_race 스타일: 시작 후 1회 펄스, 그 외엔 중립 """
        elapsed = time.time() - self.start_time
        if self.emergency_stop:
            angle_rad = 0.0
        else:
            if not self.pulse_done and elapsed >= self.pulse_start:
                angle_rad = self.pulse_angle
                self.pulse_done = True
            else:
                angle_rad = self.neutral_rad

        cmd = 0.5 + (angle_rad / self.max_steer_rad) * 0.5   # 중앙=0.5
        cmd = max(0.0, min(1.0, cmd))
        self.steer_pub.publish(Float64(data=cmd))

    def _tick_slow_publish(self, _evt):
        """ 상태 모니터링용 저빈도 브로드캐스트 """
        state = {
            "target_erpm": round(self.stop_erpm if self.emergency_stop else self.target_erpm, 2),
            "state": "estop" if self.emergency_stop else self.last_state_text,
            "feedback": None if self.feedback is None else round(float(self.feedback), 4),
            "water_decision": self.water_decision,
            "brightness_diff": round(float(self.brightness_diff), 4),
            "timestamp": time.time()
        }
        try:
            self.state_pub.publish(String(data=json.dumps(state)))
        except Exception:
            pass

    # ========= 서비스 =========
    def _srv_clear_estop(self, _req):
        """ e-stop 래치 해제 """
        if not self.emergency_stop:
            return TriggerResponse(success=True, message="E-STOP not active")
        self.emergency_stop    = False
        self.estop_since_ms    = 0
        self.brake_pub_started = False
        self.last_state_text   = "estop_cleared"
        rospy.logwarn("[E-STOP] cleared by service")
        return TriggerResponse(success=True, message="E-STOP cleared")


def main():
    rospy.init_node("detector_speed_steer_controller", anonymous=False)
    DetectorSpeedSteerController()
    rospy.spin()

if __name__ == "__main__":
    main()

