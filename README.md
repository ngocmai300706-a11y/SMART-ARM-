# SmartArm - Arm and Hand Motion Analysis System

**SmartArm** is a web-based motion analysis application designed to track and analyze arm and hand movements in real-time. By leveraging computer vision, the project aims to support fields like physical therapy, ergonomic assessment, and athletic training.

## 🚀 Key Features
* **Pose Estimation**: Uses MediaPipe Pose to identify skeletal landmarks and track arm orientation.
* **Hand Tracking**: Integrates MediaPipe Hands for detailed analysis of finger movements and gestures.
* **Automated Exercise Counter**: Features a built-in logic to count repetitions for both left and right arms based on movement stages.
* **Live Web Dashboard**: Streams processed video frames directly to a web interface using Flask.

## 🛠 Tech Stack
* **Language**: Python
* **Web Framework**: Flask
* **Computer Vision**: OpenCV and MediaPipe
* **Data Processing**: NumPy

## 📋 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/ngocmai300706-a11y/SMART-ARM-.git](https://github.com/ngocmai300706-a11y/SMART-ARM-.git)
   cd SMART-ARM-

2. **Install the required dependencies**:
   ```bash
   pip install flask mediapipe opencv-python numpy
   
3. **Run the application**:
   ```bash
   python app.py

4. **Open my broswer**:
   Navigate to `http://127.0.0.1:5000` to view the live analysis dashboard.

## 📂 Project Structure
* `app.py`: The main server file containing Flask routes and MediaPipe processing logic.
* `templates/`: Contains HTML files for the user interface (e.g., `index.html`, `index2.html`, `index4.html`).
* `static/`: Stores static assets like CSS and JavaScript files.
* `users.json`: A local file used for managing user-specific data.   
          
