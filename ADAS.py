import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import pyttsx3
import tempfile
import time
import pymysql
from datetime import datetime
import re
from streamlit_option_menu import option_menu

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Database connection using pymysql
try:
    conn = pymysql.connect(
        host="localhost",
        user="root",
        password="your_password",
        database="your_database"
    )
    cursor = conn.cursor()
except pymysql.MySQLError as err:
    st.error(f"Failed to connect to database: {err}")

# Function to sanitize username for table names
def correct_username(name):
    name = re.sub(r'\W+', '_', name)
    return f"{name.lower()}"

# Function to create user-specific table securely
def create_user_table(username):
    updated_name = correct_username(username)
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS `{updated_name}` (
        id INT AUTO_INCREMENT PRIMARY KEY,
        object_type VARCHAR(100) NOT NULL,
        distance FLOAT NOT NULL,
        warning BOOLEAN NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(create_table_query)
    conn.commit()
    return updated_name

# Function to save detection history for a user
def save_to_db(username, object_type, distance, warning):
    updated_table = correct_username(username)
    timestamp = datetime.now()
    sql = f"INSERT INTO `{updated_table}` (object_type, distance, warning, timestamp) VALUES (%s, %s, %s, %s)"
    values = (object_type, distance, warning, timestamp)
    cursor.execute(sql, values)
    conn.commit()

# Night Vision
def apply_night_vision(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.equalizeHist(gray)
    night_vision = cv2.merge((np.zeros_like(enhanced), enhanced, np.zeros_like(enhanced)))
    return night_vision

# Distance calculation
def distance(pixels):
    focal_length = 700
    actual_width = 0.8
    return (actual_width * focal_length) / pixels if pixels != 0 else 0

# Voice Assist
def speak_warning(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Lane detection
prev_left = None
prev_right = None
alpha = 0.2

def make_coordinates(img, line_params):
    slope, intercept = line_params
    y1 = img.shape[0]
    y2 = int(y1 * 0.65)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(img, lines):
    global prev_left, prev_right
    left_lines, right_lines = [], []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < -0.5:
                left_lines.append((slope, intercept))
            elif slope > 0.5:
                right_lines.append((slope, intercept))

    left_avg = np.mean(left_lines, axis=0) if left_lines else None
    right_avg = np.mean(right_lines, axis=0) if right_lines else None

    if left_avg is not None:
        prev_left = left_avg if prev_left is None else alpha * left_avg + (1 - alpha) * prev_left
    if right_avg is not None:
        prev_right = right_avg if prev_right is None else alpha * right_avg + (1 - alpha) * prev_right

    left_line = make_coordinates(img, prev_left) if prev_left is not None else None
    right_line = make_coordinates(img, prev_right) if prev_right is not None else None
    return left_line, right_line

def draw_lines(img, lines, color=(0, 0, 255), thickness=5):
    line_img = np.zeros_like(img)
    if lines[0] is not None:
        cv2.line(line_img, (lines[0][0], lines[0][1]), (lines[0][2], lines[0][3]), color, thickness)
    if lines[1] is not None:
        cv2.line(line_img, (lines[1][0], lines[1][1]), (lines[1][2], lines[1][3]), color, thickness)

    if lines[0] is not None and lines[1] is not None:
        pts = np.array([
            [lines[0][0], lines[0][1]],
            [lines[0][2], lines[0][3]],
            [lines[1][2], lines[1][3]],
            [lines[1][0], lines[1][1]]
        ])
        cv2.fillPoly(line_img, [pts], (255, 0, 0))

    return cv2.addWeighted(img, 1, line_img, 0.8, 0)

# Mini Map
def create_mini_map(frame, left_line, right_line, results):
    height, width = frame.shape[:2]
    mini_map = np.zeros((150, 150, 3), dtype=np.uint8)

    for i in range(0, 151, 30):
        cv2.line(mini_map, (i, 0), (i, 150), (30, 30, 30), 1)
        cv2.line(mini_map, (0, i), (150, i), (30, 30, 30), 1)

    lx = rx = None
    if left_line is not None:
        lx = int((left_line[0] + left_line[2]) / 2 * 150 / width)
    if right_line is not None:
        rx = int((right_line[0] + right_line[2]) / 2 * 150 / width)

    if lx is not None:
        cv2.line(mini_map, (lx, 80), (lx, 150), (0, 0, 255), 2)
    if rx is not None:
        cv2.line(mini_map, (rx, 80), (rx, 150), (0, 0, 255), 2)

    if lx is not None and rx is not None:
        pts = np.array([[lx, 150], [lx, 80], [rx, 80], [rx, 150]], np.int32)
        cv2.fillPoly(mini_map, [pts], (255, 100, 50))

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        obj_cx = int(((x1 + x2) / 2) * 170 / width)
        obj_cy = int(((y1 + y2) / 2) * 70 / height) + 30
        cv2.circle(mini_map, (obj_cx, obj_cy), 4, (0, 0, 255), -1)
        if distance(x2 - x1) <= 3:
            cv2.rectangle(mini_map, (0, 0), (149, 149), (0, 0, 255), 3)
        cv2.rectangle(mini_map, (0, 0), (149, 149), (180, 180, 180), 1)

    cv2.rectangle(mini_map, (60, 140), (95, 150), (200, 200, 200), -1)
    cv2.arrowedLine(mini_map, (78, 135), (78, 100), (255, 255, 0), 2, tipLength=0.3)
    cv2.putText(mini_map, "MINI MAP", (5, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    frame[10:160, 10:160] = mini_map
    return frame

# Streamlit UI
with st.sidebar:
    option=option_menu("Dashboard", 
                       ["ADAS", "Register", "Detection History", "About Us"], 
                       icons=["truck", "key", "file-earmark-text", "info-circle"])

# ADAS Page
if option == "ADAS":
    st.header("🚗 Advance Driving Assistance System")
    st.markdown("Welcome to the **Advanced Driving Assistance System (ADAS)**. This system utilizes state-of-the-art technologies like **Object Detection**, **Lane Detection**, and **Distance Measurement** to enhance your driving experience. It ensures safety by alerting you about obstacles and other potential dangers on the road.")
    username = st.text_input("🔑 Enter Your Vehicle Number To LogIn:", placeholder="e.g. MH12AB1234")

    # if st.button("Login") and username:
    if username:
        car_clean = correct_username(username)
        cursor.execute("SELECT name FROM users WHERE car_number = %s", (car_clean,))
        user = cursor.fetchone()
        if user:
            st.success(f"👋 Welcome, {user[0]}!") 

        
            video_file = st.file_uploader("📹 Driving Camera Video:", type=['mp4', 'mov', 'avi'])
            night_mode = st.toggle("🌙 Toggle Night Vision Mode")

            if video_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())

                cap = cv2.VideoCapture(tfile.name)
                frame_placeholder = st.empty()

                is_video_playing = True
                while cap.isOpened() and is_video_playing:
                    ret, frame = cap.read()
                    if not ret:
                        st.info("Dashcam Stopped.")
                        break

                    if night_mode:
                        frame = apply_night_vision(frame)

                    height, width = frame.shape[:2]
                    results = model(frame)[0]

                    for box in results.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        label = model.names[cls]

                        d = distance(x2 - x1)
                        warning_flag = d <= 3
                        save_to_db(username, label, d, warning_flag)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"{d:.1f}m", (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                        if warning_flag:
                            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 5)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "WARNING!", (int(width / 3), int(height / 1.75)), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 10)
                            # speak_warning(f"Warning {label} ahead!")

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    edges = cv2.Canny(blur, 50, 150)
                    mask = np.zeros_like(edges)
                    roi = np.array([[
                        (0, height),
                        (width // 2 - 50, height // 2 + 50),
                        (width // 2 + 50, height // 2 + 50),
                        (width, height)
                    ]], dtype=np.int32)
                    cv2.fillPoly(mask, roi, 255)
                    masked = cv2.bitwise_and(edges, mask)
                    lines = cv2.HoughLinesP(masked, 2, np.pi / 180, threshold=100, minLineLength=40, maxLineGap=150)

                    left_line, right_line = None, None
                    if lines is not None:
                        left_line, right_line = average_slope_intercept(frame, lines)
                        frame = draw_lines(frame, (left_line, right_line))

                    frame = create_mini_map(frame, left_line, right_line, results)

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(rgb_frame, channels="RGB")

                    time.sleep(0.03)

                cap.release() 
        
        else:
            st.error("🚫 Car number not found. Please register.")
            speak_warning("Car number not found. Please register.")

# Registration Page
elif option == "Register":
    st.header("📝 ADAS Registration")
    st.markdown("Please register to use the ADAS features. Enter your details below and get started!")

    name = st.text_input("👨‍💼 Enter Your Name:", placeholder="e.g. John Doe")
    car = st.text_input("🚘 Enter Your Car Number:", placeholder="e.g. MH12AB1234")

    if st.button("✅ Register") and name and car:
        car_clean = correct_username(car)
        cursor.execute("SELECT * FROM users WHERE car_number = %s", (car_clean,))
        if cursor.fetchone():
            st.error("⚠️ Car number already registered.")
            speak_warning("Car number already registered.")
        else:
            cursor.execute("INSERT INTO users (name, car_number) VALUES (%s, %s)", (name, car_clean))
            conn.commit()
            create_user_table(car_clean)
            st.success("🎉 Registered successfully! You can now log in.")
            speak_warning("Registered successfully! You can now log in.")

# Detection History
elif option == "Detection History":
    st.header("📊 Detection History Records")
    st.markdown("Check your vehicle's detection logs, including object encounters, distances, and safety warnings.")

    detectin_history = st.text_input("🔍 Enter Car Number:", placeholder="e.g. MH12AB1234")
    car_table = correct_username(detectin_history)

    def show_history(query, title):
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            if rows:
                st.subheader(title)
                st.table({
                    "🚘 Object Type": [r[0] for r in rows],
                    "📏 Distance (m)": [f"{r[1]:.2f}" for r in rows],
                    "⚠️ Warning": ["Yes" if r[2] else "No" for r in rows],
                    "🕒 Timestamp": [r[3].strftime("%Y-%m-%d %H:%M:%S") for r in rows],
                })
            else:
                st.info("ℹ️ No detection history found.")
        except:
            st.error("❌ No detection history found. This vehicle has not been registered or recorded yet.")

    if st.button("📂 Show All Detection History"):
        show_history(f"SELECT object_type, distance, warning, timestamp FROM `{car_table}` ORDER BY timestamp DESC LIMIT 50", "All Detections")

    if st.button("🚨 Filtered Warning Detection"):
        show_history(f"SELECT object_type, distance, warning, timestamp FROM `{car_table}` WHERE warning = 1 ORDER BY timestamp DESC", "Warnings Only")

# About Section
elif option == "About Us":
    st.header("🚗 Advanced Driver Assistance System (ADAS)")
    
    st.subheader("Overview")
    st.write("""
    The Advanced Driver Assistance System (ADAS) is a safety-focused solution designed to support drivers through intelligent visual and spatial awareness.
    It leverages real-time video analysis and computer vision to detect objects, identify lanes, and monitor potential hazards on the road.
    The system helps enhance driver decision-making and promotes safer driving conditions.
    """)

    st.subheader("Key Features")
    st.markdown("""
     🎯 &nbsp;&nbsp; Real-time object detection using YOLOv8  
     📏 &nbsp;&nbsp; Distance estimation and collision warning alerts  
     🌙 &nbsp;&nbsp; Night vision enhancement for low-light conditions  
     🛣️ &nbsp;&nbsp; Lane detection and deviation tracking  
     🗂️ &nbsp;&nbsp; Detection history logging for each vehicle  
     🔊 &nbsp;&nbsp; Voice alerts for immediate danger notification  
     🧭 &nbsp;&nbsp; On-screen mini-map for enhanced awareness  
    """)

    st.subheader("How It Helps")
    st.write("""
    This system acts as an intelligent co-pilot, continuously analyzing the surroundings and alerting the driver to potential dangers. By doing so, it:
    
    - Promotes safer driving habits  
    - Reduces the likelihood of road accidents  
    - Assists drivers during challenging driving conditions  
    - Offers a data-driven record of past detections for accountability and diagnostics
    """)

    st.subheader("Contact")
    st.write("""
    For inquiries, collaboration, or further development, feel free to reach out:
    """)
    st.markdown("""
    📧 Email: adnansaif7474@example.com  
    💼 LinkedIn: [linkedin.com/in/adnan-saif](https://linkedin.com/in/your-profile)  
    💻 GitHub: [github.com/adnan-saif](https://github.com/your-repo)  
    """)

    st.markdown("---")
    st.markdown("*Driving smarter. Driving safer.*")
    
cursor.close()
conn.close()


