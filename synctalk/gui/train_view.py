"""Training wizard view."""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QProgressBar, QRadioButton,
    QFileDialog, QMessageBox, QButtonGroup,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal


class TrainWorker(QThread):
    """Background training thread."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, name, video_path, resolution):
        super().__init__()
        self.name = name
        self.video_path = video_path
        self.resolution = resolution

    def run(self):
        try:
            self.progress.emit(5, "正在提取音频和视频帧...")
            from synctalk.data.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()

            import shutil, os
            dataset_dir = f"./dataset/{self.name}"
            os.makedirs(dataset_dir, exist_ok=True)
            target_video = os.path.join(dataset_dir, f"{self.name}.mp4")
            if not os.path.exists(target_video):
                shutil.copy2(self.video_path, target_video)

            self.progress.emit(10, "正在提取音频...")
            preprocessor.extract_audio(target_video,
                                        os.path.join(dataset_dir, "aud.wav"))

            self.progress.emit(20, "正在提取视频帧...")
            preprocessor.extract_frames(target_video)

            self.progress.emit(35, "正在检测人脸关键点...")
            preprocessor.extract_landmarks(target_video)

            self.progress.emit(50, "正在提取音频特征...")
            preprocessor.extract_audio_features(
                os.path.join(dataset_dir, "aud.wav"))

            self.progress.emit(60, "正在训练 SyncNet...")
            from synctalk.configs import SyncTalkConfig
            from synctalk.training.trainer import Trainer

            config = SyncTalkConfig.from_resolution(self.resolution)
            config.train.epochs = 20
            config.train.syncnet_epochs = 20

            trainer = Trainer(config)
            trainer.train_syncnet(dataset_dir, f"./syncnet_ckpt/{self.name}")

            self.progress.emit(80, "正在训练 UNet 主模型...")
            from pathlib import Path
            ckpts = sorted(Path(f"./syncnet_ckpt/{self.name}").glob("*.pth"))
            syncnet_ckpt = str(ckpts[-1]) if ckpts else None
            trainer.train_unet(dataset_dir, f"./checkpoint/{self.name}",
                                syncnet_checkpoint=syncnet_ckpt)

            self.progress.emit(100, "训练完成！")
            self.finished.emit(True, f"角色 '{self.name}' 训练完成")
        except Exception as e:
            self.finished.emit(False, str(e))


class TrainView(QWidget):
    """Training wizard interface."""

    def __init__(self):
        super().__init__()
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("训练数字人")
        header.setObjectName("sectionTitle")
        layout.addWidget(header)

        form = QGroupBox("基本信息")
        form_layout = QVBoxLayout(form)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("数字人名称:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如: 我的数字人")
        row1.addWidget(self.name_input)
        form_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("训练视频:"))
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("选择 MP4 视频文件")
        self.path_input.setReadOnly(True)
        row2.addWidget(self.path_input)
        btn_browse = QPushButton("选择文件...")
        btn_browse.clicked.connect(self._browse_video)
        row2.addWidget(btn_browse)
        form_layout.addLayout(row2)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("分辨率:"))
        self.res_group = QButtonGroup(self)
        self.radio_160 = QRadioButton("标清 (160px, 更快)")
        self.radio_328 = QRadioButton("高清 (328px, 推荐)")
        self.radio_328.setChecked(True)
        self.res_group.addButton(self.radio_160, 160)
        self.res_group.addButton(self.radio_328, 328)
        res_row.addWidget(self.radio_160)
        res_row.addWidget(self.radio_328)
        res_row.addStretch()
        form_layout.addLayout(res_row)
        layout.addWidget(form)

        tips = QGroupBox("视频要求")
        tips_layout = QVBoxLayout(tips)
        for tip in [
            "• 录制 3-5 分钟正脸视频，光线充足",
            "• 保持头部正对摄像头，不要大幅移动",
            "• 背景稳定，不要有第二个人的声音",
            "• 视频开头和结尾留 5 秒静音",
            "• 格式: MP4，分辨率不限（自动处理）",
        ]:
            lbl = QLabel(tip)
            lbl.setStyleSheet("color: #aaa; font-size: 12px;")
            tips_layout.addWidget(lbl)
        layout.addWidget(tips)

        progress_box = QGroupBox("训练进度")
        progress_layout = QVBoxLayout(progress_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("等待开始...")
        self.progress_label.setStyleSheet("color: #888;")
        progress_layout.addWidget(self.progress_label)

        layout.addWidget(progress_box)

        layout.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_train = QPushButton("🎯 开始训练")
        self.btn_train.setObjectName("primary")
        self.btn_train.setMinimumWidth(180)
        self.btn_train.setMinimumHeight(42)
        self.btn_train.clicked.connect(self._start_training)
        btn_row.addWidget(self.btn_train)
        layout.addLayout(btn_row)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择训练视频", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)"
        )
        if path:
            self.path_input.setText(path)

    def _start_training(self):
        name = self.name_input.text().strip()
        video = self.path_input.text().strip()

        if not name:
            QMessageBox.warning(self, "提示", "请输入数字人名称")
            return
        if not video or not os.path.exists(video):
            QMessageBox.warning(self, "提示", "请选择有效的视频文件")
            return

        resolution = self.res_group.checkedId()
        self.btn_train.setEnabled(False)
        self.btn_train.setText("训练中...")

        self._worker = TrainWorker(name, video, resolution)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def _on_finished(self, success, message):
        self.btn_train.setEnabled(True)
        self.btn_train.setText("🎯 开始训练")
        if success:
            self.progress_label.setText("✅ " + message)
            QMessageBox.information(self, "完成", message)
        else:
            self.progress_label.setText("❌ 训练失败: " + message)
            QMessageBox.critical(self, "训练失败", message)
