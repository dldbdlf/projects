#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
import os
import RPi.GPIO as GPIO
import threading
import queue

BUZZER_PIN = 16
AREA_THRESHOLD = 10000  # 면적 임계값

class SinkholeAreaLogger:
    def __init__(self):
        rospy.init_node("sinkhole_area_logger", anonymous=True)

        # 로그 파일
        self.log_file = os.path.join(os.path.expanduser("~"), "sinkhole_area.txt")
        self.fh = open(self.log_file, "a")

        # /sinkhole/size 항상 구독
        self.sub_size = rospy.Subscriber("/sinkhole/size", Float32, self.size_callback)

        # Lock for thread safety
        self.lock = threading.Lock()

        # GPIO 버저 설정
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        self.pwm = GPIO.PWM(BUZZER_PIN, 1)
        self.pwm.start(0)  # duty=0 → 처음엔 소리 안남

        # 버저 전용 스레드 & 큐
        self.buzzer_queue = queue.Queue()
        self.buzzer_thread = threading.Thread(target=self.buzzer_worker, daemon=True)
        self.buzzer_thread.start()

        rospy.loginfo(f"[INIT] Sinkhole logger ready. file={self.log_file}")
        rospy.on_shutdown(self.on_shutdown)
        rospy.spin()

    def size_callback(self, msg):
        """ /sinkhole/size 값이 들어왔을 때만 버저 작동 """
        if msg is None or msg.data is None:
            # 값이 없으면 아무 동작도 하지 않음
            return

        area = msg.data

        # 값이 들어왔을 때만 버저 큐에 전달
        self.buzzer_queue.put(area)

        # 로그 기록
        self.fh.write(f"{area:.1f}\n")
        self.fh.flush()

    def buzzer_worker(self):
        """버저 전용 스레드"""
        while True:
            area = self.buzzer_queue.get()
            if area is None:
                break
            self._adjust_buzzer(area)

    def _adjust_buzzer(self, area):
        """면적 값에 따라 주파수 조절"""
        try:
            # area가 0~AREA_THRESHOLD 범위를 벗어나지 않도록 제한
            area_clamped = min(max(area, 0), AREA_THRESHOLD)

            duty = 70  # 버저 소리 크기 고정

            if area_clamped >= AREA_THRESHOLD:
                freq = 6000  # 최고음 유지
            else:
                # 0 → 400Hz, AREA_THRESHOLD → 6000Hz
                freq = 400 + (area_clamped / AREA_THRESHOLD) * (6000 - 400)

            # 버저 적용
            self.pwm.ChangeDutyCycle(duty)
            self.pwm.ChangeFrequency(freq)

            rospy.loginfo(f"[BUZZER] area={area:.1f}, freq={freq:.1f}")

        except Exception as e:
            rospy.logwarn(f"[ERROR] adjust_buzzer: {e}")

    def on_shutdown(self):
        try:
            if self.fh:
                self.fh.close()
            if self.pwm:
                self.pwm.stop()
            GPIO.cleanup()
            self.buzzer_queue.put(None)
            rospy.loginfo("[SHUTDOWN] GPIO cleaned up, buzzer stopped")
        except Exception:
            pass

if __name__ == "__main__":
    try:
        SinkholeAreaLogger()
    except rospy.ROSInterruptException:
        pass

