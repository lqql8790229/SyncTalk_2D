"""Character management view — Card grid + detail/create pages."""

import os
import shutil
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QProgressBar, QRadioButton,
    QFileDialog, QMessageBox, QButtonGroup, QScrollArea,
    QGridLayout, QFrame, QStackedWidget, QSizePolicy,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QCursor


# ── Train Worker (background thread) ──

class TrainWorker(QThread):
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
            if self.video_path and os.path.exists(self.video_path) and not os.path.exists(target_video):
                shutil.copy2(self.video_path, target_video)
            elif not os.path.exists(target_video):
                video_files = [f for f in os.listdir(dataset_dir) if f.endswith(".mp4")]
                if video_files:
                    target_video = os.path.join(dataset_dir, video_files[0])
                else:
                    self.finished.emit(False, "未找到训练视频文件")
                    return

            self.progress.emit(10, "提取音频...")
            from synctalk.data.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            preprocessor.extract_audio(target_video, os.path.join(dataset_dir, "aud.wav"))

            self.progress.emit(20, "提取视频帧...")
            preprocessor.extract_frames(target_video)

            self.progress.emit(35, "检测人脸关键点...")
            preprocessor.extract_landmarks(target_video)

            self.progress.emit(50, "提取音频特征...")
            preprocessor.extract_audio_features(os.path.join(dataset_dir, "aud.wav"))

            self.progress.emit(60, "训练 SyncNet...")
            from synctalk.configs import SyncTalkConfig
            from synctalk.training.trainer import Trainer
            config = SyncTalkConfig.from_resolution(self.resolution)
            config.train.epochs = 20
            config.train.syncnet_epochs = 20
            trainer = Trainer(config)
            trainer.train_syncnet(dataset_dir, f"./syncnet_ckpt/{self.name}")

            self.progress.emit(80, "训练 UNet...")
            from pathlib import Path
            ckpts = sorted(Path(f"./syncnet_ckpt/{self.name}").glob("*.pth"))
            syncnet_ckpt = str(ckpts[-1]) if ckpts else None
            trainer.train_unet(dataset_dir, f"./checkpoint/{self.name}",
                                syncnet_checkpoint=syncnet_ckpt)

            self.progress.emit(100, "训练完成！")
            self.finished.emit(True, f"「{self.name}」训练完成")
        except Exception as e:
            self.finished.emit(False, str(e))


# ── Character Card Widget ──

class CharacterCard(QFrame):
    """Clickable card representing a single character."""
    clicked = pyqtSignal(str)

    def __init__(self, name: str, frame_count: int, status: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedSize(200, 160)
        self.setStyleSheet("""
            CharacterCard {
                background-color: #16213e;
                border: 2px solid #0f3460;
                border-radius: 10px;
            }
            CharacterCard:hover {
                border-color: #00d4ff;
                background-color: #1a2744;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(16, 14, 16, 14)

        icon = QLabel("🧑" if "✅" in status else "⏳")
        icon.setStyleSheet("font-size: 36px; border: none;")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        name_lbl = QLabel(name)
        name_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #fff; border: none;")
        layout.addWidget(name_lbl)

        info = QLabel(f"{frame_count} 帧")
        info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info.setStyleSheet("color: #888; font-size: 11px; border: none;")
        layout.addWidget(info)

        status_lbl = QLabel(status)
        status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        color = "#4caf50" if "就绪" in status else "#ff9800"
        status_lbl.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold; border: none;")
        layout.addWidget(status_lbl)

    def mousePressEvent(self, event):
        self.clicked.emit(self.name)


# ── Add Card (+ button) ──

class AddCard(QFrame):
    """'+' card for creating a new character."""
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedSize(200, 160)
        self.setStyleSheet("""
            AddCard {
                background-color: #0f1a2e;
                border: 2px dashed #0f3460;
                border-radius: 10px;
            }
            AddCard:hover {
                border-color: #00d4ff;
                background-color: #162240;
            }
        """)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        plus = QLabel("+")
        plus.setStyleSheet("font-size: 40px; color: #0f3460; border: none;")
        plus.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(plus)

        lbl = QLabel("创建数字人")
        lbl.setStyleSheet("color: #555; font-size: 13px; border: none;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

    def mousePressEvent(self, event):
        self.clicked.emit()


# ── Character Detail Page ──

class CharacterDetailPage(QWidget):
    """Detail page for a selected character (retrain / delete)."""
    back_requested = pyqtSignal()
    retrain_requested = pyqtSignal(str, int)
    delete_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._name = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 15, 20, 15)

        top = QHBoxLayout()
        btn_back = QPushButton("← 返回")
        btn_back.setMaximumWidth(80)
        btn_back.clicked.connect(self.back_requested.emit)
        top.addWidget(btn_back)
        self.detail_title = QLabel("")
        self.detail_title.setObjectName("sectionTitle")
        top.addWidget(self.detail_title)
        top.addStretch()
        layout.addLayout(top)

        info_box = QGroupBox("角色信息")
        info_layout = QVBoxLayout(info_box)
        self.info_name = QLabel("")
        self.info_frames = QLabel("")
        self.info_status = QLabel("")
        self.info_path = QLabel("")
        for w in [self.info_name, self.info_frames, self.info_status, self.info_path]:
            w.setStyleSheet("font-size: 13px; padding: 2px 0;")
            info_layout.addWidget(w)
        layout.addWidget(info_box)

        res_box = QGroupBox("重新训练")
        res_layout = QVBoxLayout(res_box)
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
        res_layout.addLayout(res_row)

        self.detail_progress = QProgressBar()
        self.detail_progress.setValue(0)
        res_layout.addWidget(self.detail_progress)
        self.detail_progress_label = QLabel("就绪")
        self.detail_progress_label.setStyleSheet("color:#888; font-size:12px;")
        res_layout.addWidget(self.detail_progress_label)
        layout.addWidget(res_box)

        layout.addStretch()

        btn_row = QHBoxLayout()
        self.btn_retrain = QPushButton("🔄 重新训练")
        self.btn_retrain.setObjectName("primary")
        self.btn_retrain.setMinimumHeight(40)
        self.btn_retrain.clicked.connect(self._on_retrain)
        btn_row.addWidget(self.btn_retrain)

        self.btn_delete = QPushButton("🗑️ 删除角色")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.setMinimumHeight(40)
        self.btn_delete.clicked.connect(self._on_delete)
        btn_row.addWidget(self.btn_delete)
        layout.addLayout(btn_row)

    def load_character(self, name: str):
        self._name = name
        self.detail_title.setText(f"角色: {name}")
        self.detail_progress.setValue(0)
        self.detail_progress_label.setText("就绪")

        dataset_dir = f"./dataset/{name}"
        checkpoint_dir = f"./checkpoint/{name}"
        frame_dir = os.path.join(dataset_dir, "full_body_img")
        frame_count = len(os.listdir(frame_dir)) if os.path.isdir(frame_dir) else 0
        has_ckpt = os.path.isdir(checkpoint_dir)

        self.info_name.setText(f"名称:   {name}")
        self.info_frames.setText(f"帧数:   {frame_count}")
        self.info_status.setText(f"状态:   {'✅ 已训练就绪' if has_ckpt else '⏳ 未训练'}")
        self.info_path.setText(f"路径:   {dataset_dir}")

    def _on_retrain(self):
        reply = QMessageBox.question(
            self, "确认重新训练",
            f"将重新训练「{self._name}」，之前的模型会被覆盖。\n继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.retrain_requested.emit(self._name, self.res_group.checkedId())

    def _on_delete(self):
        reply = QMessageBox.question(
            self, "确认删除",
            f"将永久删除角色「{self._name}」的所有数据。\n此操作不可撤销，确定？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.delete_requested.emit(self._name)


# ── Create Page ──

class CreatePage(QWidget):
    """New character creation form."""
    back_requested = pyqtSignal()
    create_requested = pyqtSignal(str, str, int)

    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 15, 20, 15)

        top = QHBoxLayout()
        btn_back = QPushButton("← 返回")
        btn_back.setMaximumWidth(80)
        btn_back.clicked.connect(self.back_requested.emit)
        top.addWidget(btn_back)
        header = QLabel("创建数字人")
        header.setObjectName("sectionTitle")
        top.addWidget(header)
        top.addStretch()
        layout.addLayout(top)

        form = QGroupBox("基本信息")
        form_layout = QVBoxLayout(form)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("名称:"))
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("例如：小美")
        self.name_input.setFixedHeight(34)
        row1.addWidget(self.name_input)
        form_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("视频:"))
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("选择 MP4 视频文件")
        self.path_input.setReadOnly(True)
        self.path_input.setFixedHeight(34)
        row2.addWidget(self.path_input)
        btn_browse = QPushButton("选择...")
        btn_browse.setMaximumWidth(70)
        btn_browse.clicked.connect(self._browse_video)
        row2.addWidget(btn_browse)
        form_layout.addLayout(row2)

        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("分辨率:"))
        self.res_group = QButtonGroup(self)
        r160 = QRadioButton("标清 160px")
        r328 = QRadioButton("高清 328px")
        r328.setChecked(True)
        self.res_group.addButton(r160, 160)
        self.res_group.addButton(r328, 328)
        res_row.addWidget(r160)
        res_row.addWidget(r328)
        res_row.addStretch()
        form_layout.addLayout(res_row)
        layout.addWidget(form)

        tips = QGroupBox("视频要求")
        tips_layout = QVBoxLayout(tips)
        for tip in ["• 3-5 分钟正脸视频，光线充足",
                     "• 头部正对摄像头，背景稳定",
                     "• 无第二人声，首尾留 5 秒静音",
                     "• 格式 MP4，分辨率不限"]:
            lbl = QLabel(tip)
            lbl.setStyleSheet("color: #aaa; font-size: 11px;")
            tips_layout.addWidget(lbl)
        layout.addWidget(tips)

        prog_box = QGroupBox("训练进度")
        prog_layout = QVBoxLayout(prog_box)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        prog_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("就绪")
        self.progress_label.setStyleSheet("color: #888; font-size: 12px;")
        prog_layout.addWidget(self.progress_label)
        layout.addWidget(prog_box)

        layout.addStretch()

        self.btn_train = QPushButton("🎯 开始训练")
        self.btn_train.setObjectName("primary")
        self.btn_train.setMinimumHeight(42)
        self.btn_train.clicked.connect(self._on_start)
        layout.addWidget(self.btn_train)

    def _browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择训练视频", "",
            "视频文件 (*.mp4 *.avi *.mov);;所有文件 (*)",
        )
        if path:
            self.path_input.setText(path)

    def _on_start(self):
        name = self.name_input.text().strip()
        video = self.path_input.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入数字人名称")
            return
        if not video or not os.path.exists(video):
            QMessageBox.warning(self, "提示", "请选择有效的视频文件")
            return
        self.create_requested.emit(name, video, self.res_group.checkedId())

    def reset(self):
        self.name_input.clear()
        self.path_input.clear()
        self.progress_bar.setValue(0)
        self.progress_label.setText("就绪")
        self.btn_train.setEnabled(True)
        self.btn_train.setText("🎯 开始训练")


# ── Main TrainView ──

class TrainView(QWidget):
    """Character management: card grid → detail page / create page."""

    def __init__(self):
        super().__init__()
        self._worker = None
        self._build_ui()
        self._refresh_cards()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()

        self.grid_page = self._build_grid_page()
        self.stack.addWidget(self.grid_page)

        self.detail_page = CharacterDetailPage()
        self.detail_page.back_requested.connect(self._show_grid)
        self.detail_page.retrain_requested.connect(self._retrain)
        self.detail_page.delete_requested.connect(self._delete)
        self.stack.addWidget(self.detail_page)

        self.create_page = CreatePage()
        self.create_page.back_requested.connect(self._show_grid)
        self.create_page.create_requested.connect(self._create)
        self.stack.addWidget(self.create_page)

        layout.addWidget(self.stack)

    def _build_grid_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 15, 20, 10)

        header = QHBoxLayout()
        title = QLabel("我的数字人")
        title.setObjectName("sectionTitle")
        header.addWidget(title)
        header.addStretch()
        btn_refresh = QPushButton("刷新")
        btn_refresh.setMaximumWidth(60)
        btn_refresh.clicked.connect(self._refresh_cards)
        header.addWidget(btn_refresh)
        layout.addLayout(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(16)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll.setWidget(self.grid_container)

        layout.addWidget(scroll)
        return page

    def _refresh_cards(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        col = 0
        row = 0
        max_cols = 4
        dataset_dir = "./dataset"
        checkpoint_dir = "./checkpoint"

        if os.path.isdir(dataset_dir):
            for name in sorted(os.listdir(dataset_dir)):
                full = os.path.join(dataset_dir, name)
                if not os.path.isdir(full):
                    continue
                frame_dir = os.path.join(full, "full_body_img")
                frame_count = len(os.listdir(frame_dir)) if os.path.isdir(frame_dir) else 0
                has_ckpt = os.path.isdir(os.path.join(checkpoint_dir, name))
                status = "✅ 就绪" if has_ckpt else "⏳ 未训练"

                card = CharacterCard(name, frame_count, status)
                card.clicked.connect(self._on_card_clicked)
                self.grid_layout.addWidget(card, row, col)
                col += 1
                if col >= max_cols:
                    col = 0
                    row += 1

        add_card = AddCard()
        add_card.clicked.connect(self._show_create)
        self.grid_layout.addWidget(add_card, row, col)

    def _on_card_clicked(self, name: str):
        self.detail_page.load_character(name)
        self.stack.setCurrentWidget(self.detail_page)

    def _show_create(self):
        self.create_page.reset()
        self.stack.setCurrentWidget(self.create_page)

    def _show_grid(self):
        self._refresh_cards()
        self.stack.setCurrentWidget(self.grid_page)

    def _create(self, name: str, video: str, resolution: int):
        self.create_page.btn_train.setEnabled(False)
        self.create_page.btn_train.setText("训练中...")
        self._worker = TrainWorker(name, video, resolution)
        self._worker.progress.connect(self.create_page.progress_bar.setValue)
        self._worker.progress.connect(lambda _, msg: self.create_page.progress_label.setText(msg))
        self._worker.finished.connect(self._on_create_finished)
        self._worker.start()

    def _on_create_finished(self, success, message):
        self.create_page.btn_train.setEnabled(True)
        self.create_page.btn_train.setText("🎯 开始训练")
        if success:
            self.create_page.progress_label.setText("✅ " + message)
            QMessageBox.information(self, "完成", message)
            self._show_grid()
        else:
            self.create_page.progress_label.setText("❌ " + message)
            QMessageBox.critical(self, "训练失败", message)

    def _retrain(self, name: str, resolution: int):
        self.detail_page.btn_retrain.setEnabled(False)
        self.detail_page.btn_retrain.setText("训练中...")
        self._worker = TrainWorker(name, "", resolution)
        self._worker.progress.connect(self.detail_page.detail_progress.setValue)
        self._worker.progress.connect(lambda _, msg: self.detail_page.detail_progress_label.setText(msg))
        self._worker.finished.connect(self._on_retrain_finished)
        self._worker.start()

    def _on_retrain_finished(self, success, message):
        self.detail_page.btn_retrain.setEnabled(True)
        self.detail_page.btn_retrain.setText("🔄 重新训练")
        if success:
            QMessageBox.information(self, "完成", message)
            self._show_grid()
        else:
            QMessageBox.critical(self, "训练失败", message)

    def _delete(self, name: str):
        for d in [f"./dataset/{name}", f"./checkpoint/{name}", f"./syncnet_ckpt/{name}"]:
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        QMessageBox.information(self, "完成", f"角色「{name}」已删除")
        self._show_grid()
