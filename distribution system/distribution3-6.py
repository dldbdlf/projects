#!/usr/bin/env python3
import serial
import time
import rospy
from std_msgs.msg import String

# 시리얼 포트 설정
ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)  # 라즈베리파이 기본 UART 포트

def read_tfmini():
    """TFmini 센서 데이터 읽기 (거리 mm, 신호 강도)"""
    if ser.read() == b'\x59':  # frame header 1
        if ser.read() == b'\x59':  # frame header 2
            data = ser.read(7)  # 나머지 7바이트
            distance = data[0] + data[1]*256   # 거리(mm)
            strength = data[2] + data[3]*256   # 신호 강도
            return distance, strength
    return None, None

def main():
    rospy.init_node("tfmini_distance_diff", anonymous=True)
    pub = rospy.Publisher("/water/decision", String, queue_size=10)

    prev_distance = None

    rate = rospy.Rate(10)  # 10Hz
    while not rospy.is_shutdown():
        distance, strength = read_tfmini()

        if distance is None or distance == 0:
            pub.publish("yet")
        else:
            if prev_distance is not None:
                diff = abs(distance - prev_distance)
                if diff > 50:
                    pub.publish("yes")
                else:
                    pub.publish("no")
            # 이전 거리 업데이트
            prev_distance = distance

        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    finally:
        ser.close()
