# バイオリン姿勢チェッカー

ウェブカメラを使用して、バイオリン演奏時の姿勢（特にA線開放弦での右肘の角度）をリアルタイムでチェックするアプリケーションです。

## 環境要件

- OS: Windows 11
- Python: 3.12.4
- IDE: Visual Studio Code

## プロジェクト構造
violin_posture_app/
├── src/
│   ├── app.py            # メインアプリケーション
│   └── pose_detector.py  # 姿勢検出クラス
└── requirements.txt      # 依存パッケージリスト

## セットアップ手順

1. 仮想環境の作成
python -m venv venv

2. 仮想環境の有効化
# Windows
.\venv\Scripts\activate

3. 必要なパッケージのインストール
pip install -r requirements.txt

## アプリケーションの起動
streamlit run src/app.py

## 主要な依存パッケージ
tensorflow==2.12.0
tensorflow-hub==0.12.0
opencv-python==4.7.0.72
numpy==1.23.5
streamlit==1.21.0