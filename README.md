OS: Windows 11
Python: 3.12.4
IDE: Visual Studio Code

violin_posture_app/
├── venv/                  # 仮想環境
├── src/
│   ├── app.py            # メインアプリケーション
│   ├── pose_detector.py  # 姿勢検出クラス
│   
├── models/               # モデルファイル保存ディレクトリ
└── requirements.txt      # 依存パッケージリスト

# コアライブラリ
tensorflow==2.15.0
tensorflow-hub==0.15.0
opencv-python==4.10.0.84
numpy==1.26.4
streamlit==1.40.1

# その他の依存パッケージ
absl-py==2.1.0
altair==5.4.1
attrs==24.2.0
blinker==1.9.0
cachetools==5.5.0
certifi==2024.8.30
cffi==1.17.1
charset-normalizer==3.4.0
click==8.1.7
colorama==0.4.6
contourpy==1.3.1
cycler==0.12.1
flatbuffers==24.3.25
fonttools==4.55.0
gitdb==4.0.11
GitPython==3.1.43
idna==3.10
jax==0.4.35
jaxlib==0.4.35
Jinja2==3.1.4
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
kiwisolver==1.4.7
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.9.2
mdurl==0.1.2
ml_dtypes==0.5.0
packaging==24.2
pandas==2.2.3
pillow==11.0.0
protobuf==4.25.5
pyarrow==18.0.0
pycparser==2.22
pydeck==0.9.1
Pygments==2.18.0
pyparsing==3.2.0
python-dateutil==2.9.0.post0
pytz==2024.2
referencing==0.35.1
requests==2.32.3
rich==13.9.4
rpds-py==0.21.0
scipy==1.14.1
sentencepiece==0.2.0
six==1.16.0
smmap==5.0.1
sounddevice==0.5.1
tenacity==9.0.0
toml==0.10.2
tornado==6.4.1
typing_extensions==4.12.2
tzdata==2024.2
urllib3==2.2.3
watchdog==6.0.0

# 1. 仮想環境の作成
python -m venv venv

# 2. 仮想環境の有効化
# Windows
venv\Scripts\activate

# 3. 必要なパッケージのインストール
pip install -r requirements.txt

# テストの実行
python src/test_pose.py

# アプリケーションの起動
streamlit run src/app.py
