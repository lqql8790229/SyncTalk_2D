"""Training view with character list and retrain support."""

import os
import shutil
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QProgressBar, QRadioButton,
    QFileDialog, QMessageBox, QButtonGroup, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QSplitter,
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
            dataset_dir = f"./dataset/{self.name}"
            os.makedirs(dataset_dir, exist_ok=True)
            target_video = os.path.join(dataset_dir, f"{self.name}.mp4")
            if self.video_path and not os.path.exists(target_video):
                shutil.copy2(self.video_path, target_video)

            self.progress.emit(5, "提取音频和视频帧...")
            from synctalk.data.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()

            self.progress.emit(10, "提取音频...")
            preprocessor.extract_audio(target_video,
                                        os.path.join(dataset_dir, "aud.wav"))

            self.progress.emit(20, "提取视频帧...")
            preprocessor.extract_frames(target_video)

            self.progress.emit(35, "检测人脸关键点...")
            preprocessor.extract_landmarks(target_video)

            self.progress.emit(50, "提取音频特征...")
            preprocessor.extract_audio_features(
                os.path.join(dataset_dir, "aud.wav"))

            self.progress.emit(60, "训练 SyncNet...")
            from synctalk.configs import SyncTalkConfig
            from synctalk.training.trainer import Trainer

            config = SyncTalkConfig.from_resolution(self.resolution)
            config.train.epochs = 20
            config.train.syncnet_epochs = 20

            trainer = Trainer(config)
            trainer.train_syncnet(dataset_dir, f"./syncnet_ckpt/{self.name}")

            self.progress.emit(80, "训练 UNet 主模型...")
            from pathlib import Path
            ckpts = sorted(Path(f"./syncnet_ckpt/{self.name}").glob("*.pth"))
            syncnet_ckpt = str(ckpts[-1]) if ckpts else None
            trainer.train_unet(dataset_dir, f"./checkpoint/{self.name}",
                                syncnet_checkpoint=syncnet_ckpt)

            self.progress.emit(100, "训练完成！")
            self.finished.emit(True, f"角色「{self.name}」训练完成")
        except Exception as e:
            self.finished.emit(False, str(e))


class TrainView(QWidget):
    """Character list + training wizard interface."""

    def __init__(self):
        super().__init__()
        self._worker = None
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: Character List ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 5, 0)

        list_header = QHBoxLayout()
        lbl = QLabel("已训练的数字人")
        lbl.setObjectName("sectionTitle")
        list_header.addWidget(lbl)
        list_header.addStretch()
        btn_refresh = QPushButton("刷新")
        btn_refresh.setMaximumWidth(60)
        btn_refresh.clicked.connect(self._refresh_list)
        list_header.addWidget(btn_refresh)
        left_layout.addLayout(list_header)

        self.char_table = QTableWidget()
        self.char_table.setColumnCount(4)
        self.char_table.setHorizontalHeaderLabels(["名称", "分辨率", "帧数", "状态"])
        self.char_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self.char_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Fixed)
        self.char_table.setColumnWidth(1, 70)
        self.char_table.setColumnWidth(2, 60)
        self.char_table.setColumnWidth(3, 70)
        self.char_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.char_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.char_table.verticalHeader().hide()
        self.char_table.setStyleSheet(
            "QTableWidget { background-color: #16213e; border: 1px solid #0f3460; }"
            "QTableWidget::item { padding: 6px; }"
            "QTableWidget::item:selected { background-color: #0f3460; }"
            "QHeaderView::section { background-color: #0d0d1a; color: #00d4ff; "
            "padding: 6px; border: 1px solid #0f3460; font-weight: bold; }"
        )
        left_layout.addWidget(self.char_table)

        btn_row = QHBoxLayout()
        self.btn_retrain = QPushButton("🔄 重新训练")
        self.btn_retrain.clicked.connect(self._retrain_selected)
        btn_row.addWidget(self.btn_retrain)

        self.btn_delete = QPushButton("🗑️ 删除")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(self._delete_selected)
        btn_row.addWidget(self.btn_delete)
        left_layout.addLayout(btn_row)

        splitter.addWidget(left_widget)

        # ── Right: New Training Form ──
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 0, 0, 0)

        header = QLabel("新建数字人")
        header.setObjectName("sectionTitle")
        right_layout.addWidget(header)

        form = QGroupBox("基本信息")
        form_layout = QVBoxLayout(form)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("名称:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如：小美")
        row1.addWidget(self.name_input)
        form_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("视频:"))
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("选择 MP4 视频文件")
        self.path_input.setReadOnly(True)
        row2.addWidget(self.path_input)
        btn_browse = QPushButton("选择...")
        btn_browse.setMaximumWidth(70)
        btn_browse.clicked.connect(self._browse_video)
        row2.addWidget(btn_browse)
        form_layout.addLayout(row2)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("分辨率:"))
        self.res_group = QButtonGroup(self)
        self.radio_160 = QRadioButton("标清 160px")
        self.radio_328 = QRadioButton("高清 328px")
        self.radio_328.setChecked(True)
        self.res_group.addButton(self.radio_160, 160)
        self.res_group.addButton(self.radio_328, 328)
        res_row.addWidget(self.radio_160)
        res_row.addWidget(self.radio_328)
        res_row.addStretch()
        form_layout.addLayout(res_row)
        right_layout.addWidget(form)

        tips = QGroupBox("视频要求")
        tips_layout = QVBoxLayout(tips)
        for tip in [
            "• 3-5 分钟正脸视频，光线充足",
            "• 头部正对摄像头，背景稳定",
            "• 无第二人声，首尾留 5 秒静音",
            "• 格式 MP4，分辨率不限",
        ]:
            lbl = QLabel(tip)
            lbl.setStyleSheet("color: #aaa; font-size: 11px;")
            tips_layout.addWidget(lbl)
        right_layout.addWidget(tips)

        progress_box = QGroupBox("训练进度")
        progress_layout = QVBoxLayout(progress_box)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("就绪")
        self.progress_label.setStyleSheet("color: #888; font-size: 12px;")
        progress_layout.addWidget(self.progress_label)
        right_layout.addWidget(progress_box)

        right_layout.addStretch()

        self.btn_train = QPushButton("🎯 开始训练")
        self.btn_train.setObjectName("primary")
        self.btn_train.setMinimumHeight(42)
        self.btn_train.clicked.connect(self._start_training)
        right_layout.addWidget(self.btn_train)

        splitter.addWidget(right_widget)
        splitter.setSizes([400, 400])

        layout.addWidget(splitter)

    def _refresh_list(self):
        self.char_table.setRowCount(0)
        dataset_dir = "./dataset"
        checkpoint_dir = "./checkpoint"
        if not os.path.isdir(dataset_dir):
            return

        for name in sorted(os.listdir(dataset_dir)):
            full = os.path.join(dataset_dir, name)
            if not os.path.isdir(full):
                continue

            frame_dir = os.path.join(full, "full_body_img")
            frame_count = len(os.listdir(frame_dir)) if os.path.isdir(frame_dir) else 0

            has_ckpt = os.path.isdir(os.path.join(checkpoint_dir, name))
            has_frames = frame_count > 0
            if has_ckpt:
                status = "✅ 就绪"
            elif has_frames:
                status = "⏳ 未训练"
            else:
                status = "📂 仅视频"

            aud_ave = os.path.join(full, "aud_ave.npy")
            resolution = "328px" if os.path.exists(aud_ave) else "--"

            row = self.char_table.rowCount()
            self.char_table.insertRow(row)
            self.char_table.setItem(row, 0, QTableWidgetItem(name))
            self.char_table.setItem(row, 1, QTableWidgetItem(resolution))
            self.char_table.setItem(row, 2, QTableWidgetItem(str(frame_count)))

            status_item = QTableWidgetItem(status)
            if "✅" in status:
                status_item.setForeground(Qt.GlobalColor.green)
            elif "⏳" in status:
                status_item.setForeground(Qt.GlobalColor.yellow)
            self.char_table.setItem(row, 3, status_item)

    def _get_selected_name(self) -> str:
        rows = self.char_table.selectionModel().selectedRows()
        if not rows:
            return ""
        return self.char_table.item(rows[0].row(), 0).text()

    def _retrain_selected(self):
        name = self._get_selected_name()
        if not name:
            QMessageBox.warning(self, "提示", "请先选择一个数字人")
            return

        reply = QMessageBox.question(
            self, "确认重新训练",
            f"将重新训练角色「{name}」，之前的模型权重会被覆盖。\n\n继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        dataset_dir = f"./dataset/{name}"
        video_files = [f for f in os.listdir(dataset_dir) if f.endswith(".mp4")]
        if not video_files:
            QMessageBox.warning(self, "错误", f"未找到训练视频: {dataset_dir}/*.mp4")
            return

        video_path = os.path.join(dataset_dir, video_files[0])
        self.name_input.setText(name)
        self.path_input.setText(video_path)
        self._start_training(retrain=True)

    def _delete_selected(self):
        name = self._get_selected_name()
        if not name:
            QMessageBox.warning(self, "提示", "请先选择一个数字人")
            return

        reply = QMessageBox.question(
            self, "确认删除",
            f"将永久删除角色「{name}」的所有数据（视频、帧、模型）。\n\n此操作不可撤销，确定？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        for d in [f"./dataset/{name}", f"./checkpoint/{name}", f"./syncnet_ckpt/{name}"]:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)

        self._refresh_list()
        QMessageBox.information(self, "完成", f"角色「{name}」已删除")

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择训练视频", "",
            "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)",
        )
        if path:
            self.path_input.setText(path)

    def _start_training(self, retrain=False):
        name = self.name_input.text().strip()
        video = self.path_input.text().strip()

        if not name:
            QMessageBox.warning(self, "提示", "请输入数字人名称")
            return
        if not retrain and (not video or not os.path.exists(video)):
            QMessageBox.warning(self, "提示", "请选择有效的视频文件")
            return

        resolution = self.res_group.checkedId()
        self.btn_train.setEnabled(False)
        self.btn_train.setText("训练中...")
        self.btn_retrain.setEnabled(False)

        self._worker = TrainWorker(name, video if not retrain else "", resolution)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def _on_finished(self, success, message):
        self.btn_train.setEnabled(True)
        self.btn_train.setText("🎯 开始训练")
        self.btn_retrain.setEnabled(True)

        if success:
            self.progress_label.setText("✅ " + message)
            self._refresh_list()
            QMessageBox.information(self, "训练完成", message)
        else:
            self.progress_label.setText("❌ " + message)
            QMessageBox.critical(self, "训练失败", message)
