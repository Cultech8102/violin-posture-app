import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import traceback

class PoseDetector:
    def __init__(self):
        """姿勢検出器の初期化"""
        try:
            print("Loading MoveNet model...")
            model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
            self.movenet = model.signatures['serving_default']
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def detect_pose(self, image):
        """画像から姿勢を検出"""
        try:
            input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_image = tf.convert_to_tensor(input_image)
            input_image = tf.expand_dims(input_image, axis=0)
            input_image = tf.cast(
                tf.image.resize_with_pad(input_image, 192, 192),
                dtype=tf.int32
            )

            results = self.movenet(input_image)
            keypoints = results['output_0'].numpy()
            
            if keypoints is not None:
                print("Keypoints detected:", keypoints.shape)
                print("Keypoints values range:", np.min(keypoints), "-", np.max(keypoints))
                return keypoints[0, 0]
            return None
        except Exception as e:
            print(f"Pose detection error: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_angle(self, keypoints):
        """右肩を頂点とした、右肘-右肩-臀部の角度を計算"""
        try:
            right_shoulder = keypoints[6][:2]
            right_elbow = keypoints[8][:2]
            right_hip = keypoints[12][:2]

            if np.any(np.array([right_shoulder, right_elbow, right_hip]) < 0):
                return None

            elbow_to_shoulder = right_elbow - right_shoulder
            hip_to_shoulder = right_hip - right_shoulder

            cos_angle = np.dot(elbow_to_shoulder, hip_to_shoulder) / (
                np.linalg.norm(elbow_to_shoulder) * np.linalg.norm(hip_to_shoulder)
            )
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            return angle
        except Exception as e:
            print(f"Angle calculation error: {str(e)}")
            return None

    def estimate_bow_position(self, keypoints):
        """右手と顔の距離から弓の位置を推定（相対距離使用）"""
        try:
            # 右肩と左肩の距離を基準として使用
            right_shoulder = keypoints[6][:2]
            left_shoulder = keypoints[5][:2]
            shoulder_distance = abs(right_shoulder[1] - left_shoulder[1])
            
            # 右手首と鼻の距離を計算
            right_wrist = keypoints[10][:2]
            nose = keypoints[0][:2]
            wrist_nose_distance = abs(right_wrist[1] - nose[1])
            
            # 肩幅を基準とした相対距離を計算
            relative_distance = wrist_nose_distance / shoulder_distance
            
            # 相対距離に基づいて弓の位置を推定
            if relative_distance > 1.1:      # 弓先
                return 'tip', 1.0
            elif relative_distance > 0.7:    # 弓中
                return 'middle', 0.5
            else:                            # 弓元
                return 'frog', 0.0
                
        except Exception as e:
            print(f"Bow position estimation error: {str(e)}")
            return 'middle', 0.5

    def get_bow_stroke_angle_range(self, bow_position_ratio):
        """運弓位置に応じた許容角度範囲を計算"""
        # 基準角度の設定
        ranges = {
            "frog": (45, 70),   # 弓元
            "middle": (37, 50), # 弓中
            "tip": (35, 45)     # 弓先
        }
        
        if bow_position_ratio <= 0.33:
            # 弓元から弓中への補間
            t = bow_position_ratio * 3
            min_angle = ranges["frog"][0] + t * (ranges["middle"][0] - ranges["frog"][0])
            max_angle = ranges["frog"][1] + t * (ranges["middle"][1] - ranges["frog"][1])
        else:
            # 弓中から弓先への補間
            t = (bow_position_ratio - 0.33) * 1.5
            min_angle = ranges["middle"][0] + t * (ranges["tip"][0] - ranges["middle"][0])
            max_angle = ranges["middle"][1] + t * (ranges["tip"][1] - ranges["middle"][1])
        
        return min_angle, max_angle

    def check_posture_message(self, keypoints):
        """姿勢チェックとメッセージ生成（自動弓位置推定版）"""
        if keypoints is None:
            return False, "姿勢を検出できません"

        angle = self.calculate_angle(keypoints)
        if angle is None:
            return False, "角度を計算できません"

        bow_position, position_ratio = self.estimate_bow_position(keypoints)
        min_angle, max_angle = self.get_bow_stroke_angle_range(position_ratio)

        position_names = {
            'frog': '弓元',
            'middle': '弓中',
            'tip': '弓先'
        }

        if min_angle <= angle <= max_angle:
            return True, f"✅ {position_names[bow_position]}での適切な姿勢です (角度: {angle:.1f}°)"
        elif angle < min_angle:
            return False, f"⚠️ {position_names[bow_position]}での演奏時は右肘が下がりすぎています (角度: {angle:.1f}°)"
        else:
            return False, f"⚠️ {position_names[bow_position]}での演奏時は右肘が上がりすぎています (角度: {angle:.1f}°)"

    def draw_landmarks(self, image, keypoints):
        """検出された姿勢のランドマークを描画"""
        try:
            h, w, _ = image.shape
            # キーポイントの描画
            for idx, kp in enumerate(keypoints):
                x, y = int(kp[1] * w), int(kp[0] * h)
                if kp[2] > 0.3:  # 信頼度が一定以上の場合のみ描画
                    cv2.circle(image, (x, y), 4, (0, 255, 0), -1)

            # 特定のポイントを強調
            key_points = [
                (6, (255, 0, 0)),   # 右肩 (青)
                (8, (0, 255, 0)),   # 右肘 (緑)
                (12, (0, 0, 255))   # 右臀部 (赤)
            ]

            for idx, color in key_points:
                kp = keypoints[idx]
                if kp[2] > 0.3:
                    x, y = int(kp[1] * w), int(kp[0] * h)
                    cv2.circle(image, (x, y), 8, color, -1)

            # 線の描画
            connections = [
                (6, 8),   # 右肩 - 右肘
                (6, 12)   # 右肩 - 右臀部
            ]

            for start_idx, end_idx in connections:
                if keypoints[start_idx][2] > 0.3 and keypoints[end_idx][2] > 0.3:
                    start_point = keypoints[start_idx]
                    end_point = keypoints[end_idx]
                    start_x = int(start_point[1] * w)
                    start_y = int(start_point[0] * h)
                    end_x = int(end_point[1] * w)
                    end_y = int(end_point[0] * h)
                    cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            return image
        except Exception as e:
            print(f"Drawing error: {str(e)}")
            return image