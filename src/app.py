import streamlit as st
import cv2
import numpy as np
from pose_detector import PoseDetector

def main():
    st.title("バイオリン姿勢チェッカー")
    st.write("A線開放弦での右肘の角度をリアルタイムで確認できます")

    if 'detector' not in st.session_state:
        st.session_state.detector = PoseDetector()

    # カメラが利用可能か確認
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("カメラを認識できません")
        return

    frame_placeholder = st.empty()
    info_placeholder = st.empty()
    position_placeholder = st.empty()  # 弓の位置表示用
    
    stop_button = st.button("停止")

    while not stop_button:
        ret, frame = cap.read()
        if ret:
            keypoints = st.session_state.detector.detect_pose(frame)
            
            if keypoints is not None:
                frame = st.session_state.detector.draw_landmarks(frame.copy(), keypoints)
                is_correct, message = st.session_state.detector.check_posture_message(keypoints)
                bow_position, _ = st.session_state.detector.estimate_bow_position(keypoints)
                info_placeholder.text(message)
                position_placeholder.text(f"現在の弓の位置: {bow_position}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB", use_container_width=True)
        else:
            st.error("カメラからの映像を取得できません")
            break

    cap.release()

if __name__ == "__main__":
    main()