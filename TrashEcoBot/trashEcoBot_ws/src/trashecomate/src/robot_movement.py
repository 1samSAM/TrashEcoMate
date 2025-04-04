#!/usr/bin/env python3
import rospy
import RPi.GPIO as GPIO
import firebase_admin
from firebase_admin import credentials, db
from std_msgs.msg import Int32
import time
import os
import numpy as np
import signal
import sys
import math


# GPIO Pins for L293D and Servo
LEFT_IN1, LEFT_IN2 = 17, 18
RIGHT_IN3, RIGHT_IN4 = 27, 22
TRIG, ECHO = 23, 24
ENABLE_1, ENABLE_2 = 19, 12
SERVO_PIN = 13  # Servo motor pin (Physical Pin 33)

# Flag to track if cleanup has been performed
cleaned_up = False

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([LEFT_IN1, LEFT_IN2, RIGHT_IN3, RIGHT_IN4, TRIG, ENABLE_1, ENABLE_2, SERVO_PIN], GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output([ENABLE_1, ENABLE_2], GPIO.HIGH)

# Servo setup using PWM
servo = GPIO.PWM(SERVO_PIN, 50)  # 50 Hz PWM frequency for servo
servo.start(0)  # Initialize servo at 0% duty cycle

# Firebase setup
cred_path = os.getenv("FIREBASE_CREDENTIALS", "/home/sam/waste_management_robot/serviceAccountKey.json")
firebase_url = os.getenv("FIREBASE_URL", "https://trashecomate-default-rtdb.asia-southeast1.firebasedatabase.app/")
try:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {"databaseURL": firebase_url})
    rospy.loginfo("Firebase initialized successfully")
except Exception as e:
    rospy.logerr(f"Failed to initialize Firebase: {e}")
    exit(1)

status_ref = db.reference("robot/status")

# Q-Learning setup
actions = ["forward", "backward", "turn_right", "turn_left", "stop"]
num_actions = len(actions)
distance_bins = [0, 10, 20, 50, 999]  # Discretize distance into bins
num_states = len(distance_bins) - 1
q_table = np.zeros((num_states, num_actions))  # Q-table
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate

# Robot position and orientation
robot_x = 0.0  # Starting position (x-coordinate in meters)
robot_y = 0.0  # Starting position (y-coordinate in meters)
robot_angle = 0.0  # Orientation in degrees (0 = facing positive x-axis, 90 = positive y-axis)

# Target location (e.g., bin location in the room)
target_x = 2.0  # meters
target_y = 1.0  # meters

# Robot movement parameters
SPEED = 0.5  # meters per second (adjust based on your robot's actual speed)
TURN_TIME = 0.5  # seconds to turn 90 degrees
MOVE_DISTANCE = 0.5  # meters per move

def discretize_distance(distance):
    for i in range(len(distance_bins) - 1):
        if distance_bins[i] <= distance < distance_bins[i + 1]:
            return i
    return len(distance_bins) - 2

def choose_action(state, distance_to_target):
    if distance_to_target > 0.5:  # Prioritize moving forward when far from target
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(num_actions)
        return actions.index("forward")
    else:
        if np.random.uniform(0, 1) < epsilon:
            return np.random.randint(num_actions)
        return np.argmax(q_table[state])

def set_servo_angle(angle):
    duty = 2.5 + (angle / 18.0)  # Linear mapping for 0-180 degrees
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)  # Allow time for servo to move
    servo.ChangeDutyCycle(0)  # Stop sending signal to prevent jitter

def move_forward(duration):
    global robot_x, robot_y
    GPIO.output(LEFT_IN1, GPIO.HIGH)
    GPIO.output(LEFT_IN2, GPIO.LOW)
    GPIO.output(RIGHT_IN3, GPIO.HIGH)
    GPIO.output(RIGHT_IN4, GPIO.LOW)
    try:
        status_ref.set("Moving Forward")
    except Exception as e:
        rospy.logerr(f"Firebase status update failed: {e}")
    rospy.loginfo(f"Moving Forward for {duration:.2f} seconds")
    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(0.01)  # Non-blocking sleep for smoother motion
    move_distance = SPEED * duration
    robot_x += move_distance * math.cos(math.radians(robot_angle))
    robot_y += move_distance * math.sin(math.radians(robot_angle))
    stop()

def move_backward(duration):
    global robot_x, robot_y
    GPIO.output(LEFT_IN1, GPIO.LOW)
    GPIO.output(LEFT_IN2, GPIO.HIGH)
    GPIO.output(RIGHT_IN3, GPIO.LOW)
    GPIO.output(RIGHT_IN4, GPIO.HIGH)
    try:
        status_ref.set("Moving Backward")
    except Exception as e:
        rospy.logerr(f"Firebase status update failed: {e}")
    rospy.loginfo(f"Moving Backward for {duration:.2f} seconds")
    start_time = time.time()
    while time.time() - start_time < duration:
        time.sleep(0.01)  # Non-blocking sleep
    move_distance = SPEED * duration
    robot_x -= move_distance * math.cos(math.radians(robot_angle))
    robot_y -= move_distance * math.sin(math.radians(robot_angle))
    stop()

def turn_right():
    global robot_angle
    GPIO.output(LEFT_IN1, GPIO.HIGH)
    GPIO.output(LEFT_IN2, GPIO.LOW)
    GPIO.output(RIGHT_IN3, GPIO.LOW)
    GPIO.output(RIGHT_IN4, GPIO.LOW)
    try:
        status_ref.set("Turning Right")
    except Exception as e:
        rospy.logerr(f"Firebase status update failed: {e}")
    rospy.loginfo("Turning Right")
    start_time = time.time()
    while time.time() - start_time < TURN_TIME:
        time.sleep(0.01)  # Non-blocking sleep
    robot_angle = (robot_angle - 90) % 360
    stop()

def turn_left():
    global robot_angle
    GPIO.output(LEFT_IN1, GPIO.LOW)
    GPIO.output(LEFT_IN2, GPIO.LOW)
    GPIO.output(RIGHT_IN3, GPIO.HIGH)
    GPIO.output(RIGHT_IN4, GPIO.LOW)
    try:
        status_ref.set("Turning Left")
    except Exception as e:
        rospy.logerr(f"Firebase status update failed: {e}")
    rospy.loginfo("Turning Left")
    start_time = time.time()
    while time.time() - start_time < TURN_TIME:
        time.sleep(0.01)  # Non-blocking sleep
    robot_angle = (robot_angle + 90) % 360
    stop()

def stop():
    global cleaned_up
    if not cleaned_up:
        GPIO.output([LEFT_IN1, LEFT_IN2, RIGHT_IN3, RIGHT_IN4], GPIO.LOW)
        try:
            status_ref.set("Off")
        except Exception as e:
            rospy.logerr(f"Firebase status update failed: {e}")
        rospy.loginfo("Stopping")

def get_distance():
    try:
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        start_time = time.time()
        stop_time = time.time()

        # Wait for ECHO to go high (start of echo pulse)
        while GPIO.input(ECHO) == 0 and time.time() - start_time < 0.1:
            start_time = time.time()

        # Wait for ECHO to go low (end of echo pulse)
        while GPIO.input(ECHO) == 1 and time.time() - start_time < 0.1:
            stop_time = time.time()

        elapsed = stop_time - start_time
        if elapsed <= 0 or time.time() - start_time >= 0.1:
            rospy.logwarn("Ultrasonic sensor timeout or invalid reading, returning default distance")
            return 999  # Default distance if sensor fails

        distance = (elapsed * 34300) / 2
        rospy.loginfo(f"Measured distance: {distance:.2f} cm")
        return distance
    except Exception as e:
        rospy.logerr(f"Error in get_distance: {e}")
        return 999  # Return default distance on error

def look_left():
    set_servo_angle(170)  # Look left
    distance = get_distance()
    set_servo_angle(90)  # Return to center
    return distance

def look_right():
    set_servo_angle(10)  # Look right
    distance = get_distance()
    set_servo_angle(90)  # Return to center
    return distance

def look_front():
    set_servo_angle(90)  # Look front
    distance = get_distance()
    return distance

def check_bin_status():
    try:
        waste_level = rospy.wait_for_message("/bin_level", Int32, timeout=5).data
        rospy.loginfo(f"Checked bin status: {waste_level}%")
        return waste_level
    except rospy.ROSException as e:
        rospy.logwarn(f"Failed to receive bin level: {e}")
        return None


def navigate_to_target(target_x, target_y):
    global robot_x, robot_y, robot_angle
    # Calculate the distance and angle to the target
    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = math.sqrt(dx**2 + dy**2)
    target_angle = math.degrees(math.atan2(dy, dx))  # Angle to target in degrees

    # Calculate the angle to turn
    angle_diff = (target_angle - robot_angle) % 360
    if angle_diff > 180:
        angle_diff -= 360

    # Turn to face the target
    if abs(angle_diff) > 5:
        turns = int(abs(angle_diff) / 90)
        direction = 1 if angle_diff > 0 else -1
        for _ in range(turns):
            if direction > 0:
                turn_left()
            else:
                turn_right()
            time.sleep(0.1)  # Short delay between turns
    return distance


# State machine states
IDLE = 0
NAVIGATING = 1
AVOIDING = 2
COLLECTING = 3



def motor_control():
    state = IDLE
    last_state_change_time = time.time()
    check_interval = 30  # Check bin status every 30 seconds
    last_check_time = time.time()

    set_servo_angle(90)  # Start facing forward

    while not rospy.is_shutdown():
        current_time = time.time()

        # Check bin status periodically
        if current_time - last_check_time >= check_interval:
            waste_level = check_bin_status()
            if waste_level is None:
                state = IDLE
                stop()
                try:
                    status_ref.set("Error: No bin level data")
                except Exception as e:
                    rospy.logerr(f"Firebase status update failed: {e}")
                continue
            if waste_level > 90 and state != COLLECTING:
                state = NAVIGATING
                try:
                    status_ref.set("Active")
                except Exception as e:
                    rospy.logerr(f"Firebase status update failed: {e}")
                rospy.loginfo("Waste level > 90%, starting navigation")
            else:
                state = IDLE
                stop()
                try:
                    status_ref.set("Idle")
                except Exception as e:
                    rospy.logerr(f"Firebase status update failed: {e}")
                rospy.loginfo("Waste level <= 90%, robot idle")
            last_check_time = current_time

        # State machine logic
        if state == NAVIGATING:
            remaining_distance = navigate_to_target(target_x, target_y)
            distance_front = look_front()
            if remaining_distance < 0.1:
                state = COLLECTING
                try:
                    waste_level_ref.set(0)
                    rospy.loginfo("Waste collected, resetting waste level to 0%")
                except Exception as e:
                    rospy.logerr(f"Failed to reset waste level in Firebase: {e}")
                try:
                    status_ref.set("Idle")
                except Exception as e:
                    rospy.logerr(f"Firebase status update failed: {e}")
                rospy.loginfo(f"Reached target ({target_x}, {target_y}), stopping")
                state = IDLE
            elif distance_front < 20 and distance_front != 999:
                state = AVOIDING
                move_backward(MOVE_DISTANCE / SPEED)  # Move back 0.5 meters
                rospy.loginfo("Obstacle detected, entering avoidance mode")

        elif state == AVOIDING:
            distance_left = look_left()
            distance_right = look_right()
            distance_front = look_front()

            distances = {"left": distance_left, "right": distance_right, "front": distance_front}
            safest_direction = max(distances, key=distances.get)
            safest_distance = distances[safest_direction]
            rospy.loginfo(f"Safest direction: {safest_direction} with distance {safest_distance:.2f} cm")

            if safest_distance < 20:
                move_backward(MOVE_DISTANCE / SPEED)  # Move back again if no safe path
                state = AVOIDING  # Stay in avoidance mode
            else:
                # Confirm the new direction is clear
                if safest_direction == "left":
                    turn_left()
                    time.sleep(0.1)
                    if look_front() < 20:
                        turn_right()  # Revert if still obstructed
                        turn_right()  # Turn back to original direction
                        state = AVOIDING
                    else:
                        move_forward(MOVE_DISTANCE / SPEED)
                        state = NAVIGATING
                elif safest_direction == "right":
                    turn_right()
                    time.sleep(0.1)
                    if look_front() < 20:
                        turn_left()  # Revert if still obstructed
                        turn_left()  # Turn back to original direction
                        state = AVOIDING
                    else:
                        move_forward(MOVE_DISTANCE / SPEED)
                        state = NAVIGATING
                else:
                    move_forward(MOVE_DISTANCE / SPEED)
                    state = NAVIGATING

            # Update Q-table
            state_idx = discretize_distance(distance_front)
            action_idx = actions.index("backward" if safest_distance < 20 else safest_direction)
            reward = 1 if safest_distance >= 20 else -1
            next_distance = look_front()
            next_state = discretize_distance(next_distance)
            q_table[state_idx, action_idx] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state_idx, action_idx])

        elif state == COLLECTING:
            stop()
            state = IDLE

        elif state == IDLE:
            stop()

        # Non-blocking delay to prevent 1-second stops
        rospy.sleep(0.01)

def cleanup():
    global cleaned_up
    if not cleaned_up:
        rospy.loginfo("Cleaning up resources")
        stop()
        servo.stop()
        GPIO.cleanup()
        cleaned_up = True

def signal_handler(sig, frame):
    rospy.loginfo("Termination signal received, cleaning up")
    cleanup()
    sys.exit(0)


if __name__ == "__main__":
    # Handle termination signals (e.g., Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        motor_control()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupted, cleaning up")
        cleanup()
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
        cleanup()
    finally:
        cleanup()