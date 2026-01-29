from flask import Flask, request, jsonify, send_from_directory, Response
import json, os, cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__, static_url_path='/static', static_folder='static')
USER_DATA_FILE = 'users.json'

# --- KHỞI TẠO MEDIAPIPE TOÀN CỤC ---
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# --- QUẢN LÝ TRẠNG THÁI ---
counters = {'LEFT': 0, 'RIGHT': 0}
stages = {'LEFT': None, 'RIGHT': None} 

hand_data = {
    "Left":  {"counter": 0, "stage": None, "color": (255, 255, 255)},
    "Right": {"counter": 0, "stage": None, "color": (255, 255, 255)}
}

elbow_counters = {'LEFT': 0, 'RIGHT': 0}
elbow_stages = {'LEFT': None, 'RIGHT': None}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180 else 360 - angle

def get_shoulder_eval(angle, side):
    global counters, stages
    UP_THRESHOLD, DOWN_THRESHOLD = 140, 40
    if angle > UP_THRESHOLD:
        stages[side] = "up"
        return "FULL ABDUCTION", (0, 255, 0)
    elif angle < DOWN_THRESHOLD:
        if stages[side] == 'up': counters[side] += 1
        stages[side] = "down"
        return "ARM AT SIDE", (200, 200, 200)
    return "LIFTING...", (0, 165, 255)

def get_eval_elbow(angle):
    if angle <= 50: return "EXCELLENT FLEXION", (0, 255, 0)
    elif angle >= 150: return "FULL EXTENSION", (0, 255, 0)
    return "MOVING...", (255, 255, 255)

# --- LUỒNG XỬ LÝ 1: VAI (SHOULDER) ---
def gen_shoulder():
    global counters, stages
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while True:
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Vẽ khung thân người
                l_s = tuple(np.multiply([landmarks[11].x, landmarks[11].y], [w, h]).astype(int))
                r_s = tuple(np.multiply([landmarks[12].x, landmarks[12].y], [w, h]).astype(int))
                l_h = tuple(np.multiply([landmarks[23].x, landmarks[23].y], [w, h]).astype(int))
                r_h = tuple(np.multiply([landmarks[24].x, landmarks[24].y], [w, h]).astype(int))
                cv2.line(frame, l_s, r_s, (255, 255, 255), 2)
                cv2.line(frame, r_s, r_h, (255, 255, 255), 2)
                cv2.line(frame, r_h, l_h, (255, 255, 255), 2)
                cv2.line(frame, l_h, l_s, (255, 255, 255), 2)

                # SỬA LOGIC HIỂN THỊ VAI: Đảo ids để nhãn LEFT hiện đúng bên trái màn hình
                for side, ids in [('LEFT', [24,12,14,16]), ('RIGHT', [23,11,13,15])]:
                    try:
                        hip, sho, elb, wri = [[landmarks[i].x, landmarks[i].y] for i in ids]
                        angle = calculate_angle(hip, sho, elb)
                        
                        # real_side để đếm đúng Reps cho tay vật lý
                        real_side = 'RIGHT' if side == 'LEFT' else 'LEFT'
                        eval_t, eval_c = get_shoulder_eval(angle, real_side)
                        
                        s_p, e_p, w_p = [tuple(np.multiply(c, [w, h]).astype(int)) for c in [sho, elb, wri]]
                        cv2.line(frame, s_p, e_p, (255, 255, 255), 4)
                        cv2.line(frame, e_p, w_p, (255, 255, 255), 4)
                        cv2.circle(frame, s_p, 10, eval_c, -1)

                        # UI Vai
                        pos_y = 60 if side == 'LEFT' else 200
                        box_color = (245, 117, 16) if side == 'LEFT' else (117, 66, 245)
                        cv2.rectangle(frame, (0, pos_y-45), (280, pos_y+85), box_color, -1)
                        cv2.putText(frame, f"{side}: {int(angle)} deg", (10, pos_y), 1, 1.2, (255,255,255), 2)
                        cv2.putText(frame, eval_t, (10, pos_y+35), 1, 1.2, eval_c, 2)
                        cv2.putText(frame, f"REPS: {counters[real_side]}", (10, pos_y+75), 1, 1.8, (0,255,255), 3)
                    except: pass
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# --- LUỒNG XỬ LÝ 2: KHUỶU TAY (ELBOW) ---
def gen_elbow():
    global elbow_counters, elbow_stages
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while True:
            success, frame = cap.read()
            if not success: break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # SỬA LOGIC HIỂN THỊ KHUỶU TAY: Đảo ids cho khớp Mirror
                for side, ids in [('LEFT', [12, 14, 16]), ('RIGHT', [11, 13, 15])]:
                    try:
                        s_r, e_r, w_r = [[landmarks[i].x, landmarks[i].y] for i in ids]
                         # Lấy tọa độ điểm cho tay riêng biệt
                        angle = calculate_angle(s_r, e_r, w_r)
                        eval_t, eval_c = get_eval_elbow(angle)
                        real_side = 'RIGHT' if side == 'LEFT' else 'LEFT'
                         # Duỗi tay (Extension) và tính Rep cho ĐÚNG BÊN
                        if angle < 50: 
                            elbow_stages[real_side] = "up"
                        if angle > 150 and elbow_stages[real_side] == "up":
                            elbow_stages[real_side] = "down"
                            elbow_counters[real_side] += 1
                         # Vẽ xương và khớp khuỷu tay
                        sp, ep, wp = [tuple(np.multiply(c, [w, h]).astype(int)) for c in [s_r, e_r, w_r]]
                        cv2.line(frame, sp, ep, (255, 255, 255), 3)
                        cv2.line(frame, ep, wp, (255, 255, 255), 3)
                        cv2.circle(frame, ep, 12, eval_c, -1)

                        # UI Khuỷu tay
                        pos_y = 50 if side == 'LEFT' else 170
                        box_color = (245, 117, 16) if side == 'LEFT' else (117, 66, 245)
                        cv2.rectangle(frame, (0, pos_y-35), (260, pos_y+75), box_color, -1)
                        cv2.putText(frame, f"{side} ELBOW: {int(angle)} deg", (10, pos_y), 1, 1.2, (255,255,255), 2)
                        cv2.putText(frame, eval_t, (10, pos_y+35), 1, 1.1, eval_c, 2)
                        cv2.putText(frame, f"REPS: {elbow_counters[real_side]}", (10, pos_y+65), 1, 1.3, (0,255,255), 2)
                    except: pass
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

# --- LUỒNG XỬ LÝ 3: BÀN TAY (HAND) ---
def gen_hand():
    global hand_data
    cap = cv2.VideoCapture(0)
    # Tăng độ phân giải để nhận diện khớp chuẩn hơn
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # CẤU HÌNH NÂNG CAO ĐỂ ỔN ĐỊNH KHUNG
    with mp_hands.Hands(
        static_image_mode=False,        # False để tối ưu cho video stream
        max_num_hands=2,                # Nhận diện tối đa 2 tay
        model_complexity=1,             # Tăng độ phức tạp để bắt khớp chuẩn hơn
        min_detection_confidence=0.8,   # Ngưỡng phát hiện lần đầu
        min_tracking_confidence=0.8     # Ngưỡng giữ khung xương ổn định
    ) as hands:
        while True:
            success, frame = cap.read()
            if not success: break
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    label = results.multi_handedness[idx].classification[0].label 
                    mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    # --- 1. LOGIC TÍNH REPS MỚI (CHỐNG SAI SỐ KHOẢNG CÁCH) ---
                    # Lấy tọa độ các khớp quan trọng
                    wrist = hand_lms.landmark[0]   # Cổ tay
                    mcp = hand_lms.landmark[9]     # Khớp gốc ngón giữa
                    tip = hand_lms.landmark[12]    # Đầu ngón giữa

                    # Tính "Độ dài bàn tay" làm chuẩn (từ cổ tay đến gốc ngón giữa)
                    base_dist = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)
                    # Tính "Khoảng cách hiện tại" (từ cổ tay đến đầu ngón giữa)
                    current_dist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
                    
                    # Tính tỷ lệ (Ratio)
                    # Nếu xòe tay: ratio sẽ lớn (> 2.0). Nếu nắm tay: ratio sẽ nhỏ (< 1.2)
                    ratio = current_dist / base_dist if base_dist != 0 else 0
                    
                    curr = hand_data[label]
                    
                    # Logic đếm Reps dựa trên tỷ lệ
                    if ratio > 2.1: # Trạng thái Xòe tay
                        curr["stage"] = "Open"
                    if ratio < 1.3 and curr["stage"] == "Open": # Trạng thái Nắm tay
                        curr["stage"] = "Closed"
                        curr["counter"] += 1
                    
                    # Tính Quality Score dựa trên ratio (từ 1.2 đến 2.2)
                    score = int(np.clip(np.interp(ratio, [1.2, 2.2], [100, 0]), 0, 100))
                    
                    # --- 2. THIẾT LẬP MÀU SẮC (TAY TRÁI ĐỎ - TAY PHẢI XANH BIỂN THEO Ý BÀ) ---
                    # Bà lưu ý: OpenCV dùng BGR nên Đỏ là (0,0,255), Xanh biển là (255,0,0)
                    if label == "Left":
                        box_color = (0, 0, 255) # Đỏ cho tay trái
                        ox = 30
                    else:
                        box_color = (255, 0, 0) # Xanh biển cho tay phải
                        ox = w - 280
                    
                    # --- 3. VẼ BẢNG THÔNG SỐ ---
                    cv2.rectangle(frame, (ox, 20), (ox + 250, 160), box_color, -1)
                    cv2.rectangle(frame, (ox, 20), (ox + 250, 160), (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"{label.upper()} HAND", (ox + 15, 55), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"REPS: {curr['counter']}", (ox + 15, 105), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                    cv2.putText(frame, f"QUALITY: {score}%", (ox + 15, 145), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # --- 4. VẼ THANH PROGRESS BAR ---
                    bar_x = ox + 80
                    bar_start_y = 190
                    bar_end_y = 480
                    # Màu động: Đỏ (0%) -> Xanh lá (100%)
                    dynamic_color = (0, int(score * 2.55), int(255 - score * 2.55))
                    bar_height = int(np.interp(score, [0, 100], [bar_end_y, bar_start_y]))
                    
                    cv2.rectangle(frame, (bar_x, bar_start_y), (bar_x + 40, bar_end_y), (255, 255, 255), 3)
                    cv2.rectangle(frame, (bar_x, bar_height), (bar_x + 40, bar_end_y), dynamic_color, -1)
                    cv2.putText(frame, f"{score}%", (bar_x + 50, bar_height + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    users = load_users()
    user = users.get(data['email'])
    if user and user['password'] == data['password']: return jsonify({"status": "success", "message": "Đăng nhập thành công!"})
    return jsonify({"status": "error", "message": "Sai mật khẩu!"})

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    users = load_users()
    if data['email'] in users: return jsonify({"status": "error", "message": "Email tồn tại!"})
    users[data['email']] = {"password": data['password'], "username": data.get('username')}
    save_users(users)
    return jsonify({"status": "success", "message": "Đăng ký thành công!"})

@app.route('/video_feed/<type>')
def video_feed(type):
    if type == 'shoulder': return Response(gen_shoulder(), mimetype='multipart/x-mixed-replace; boundary=frame')
    elif type == 'elbow': return Response(gen_elbow(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else: return Response(gen_hand(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register-page')
def register_page(): return send_from_directory('.', 'index2.html')

@app.route('/selection')
def selection(): return send_from_directory('.', 'index4.html')

@app.route('/')
def index(): return send_from_directory('.', 'index.html')

def load_users():
    if not os.path.exists(USER_DATA_FILE): return {}
    with open(USER_DATA_FILE, 'r') as f: return json.load(f)

def save_users(users):
    with open(USER_DATA_FILE, 'w') as f: json.dump(users, f)

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)