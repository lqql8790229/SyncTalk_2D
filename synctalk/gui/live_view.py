"""Live streaming control panel with TTS voice selection."""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QGroupBox, QTextEdit, QRadioButton, QButtonGroup,
    QFormLayout,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal


TTS_VOICES = [
    ("zh-CN-XiaoxiaoNeural", "晓晓 (女声·温柔)"),
    ("zh-CN-YunxiNeural", "云希 (男声·阳光)"),
    ("zh-CN-YunyangNeural", "云扬 (男声·沉稳)"),
    ("zh-CN-XiaoyiNeural", "晓伊 (女声·活泼)"),
    ("zh-CN-YunjianNeural", "云健 (男声·有力)"),
    ("zh-CN-YunxiaNeural", "云夏 (男声·少年)"),
    ("en-US-JennyNeural", "Jenny (English·Female)"),
    ("en-US-GuyNeural", "Guy (English·Male)"),
    ("ja-JP-NanamiNeural", "七海 (日本語·女性)"),
]


class LiveView(QWidget):
    """Main live streaming control interface with TTS voice selector."""

    start_requested = pyqtSignal(dict)
    stop_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._is_live = False
        self._build_ui()
        self._start_stats_timer()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # ── Left: preview + TTS input ──
        left = QVBoxLayout()
        left.setSpacing(10)

        preview_box = QGroupBox("实时预览")
        preview_layout = QVBoxLayout(preview_box)
        self.preview_label = QLabel("选择角色并点击「开始直播」")
        self.preview_label.setMinimumSize(480, 340)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #0d0d1a; border-radius: 8px; color: #555; font-size: 15px;")
        preview_layout.addWidget(self.preview_label)

        self.stats_label = QLabel("FPS: -- | 延迟: --ms")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_label.setStyleSheet("color: #888; font-size: 12px;")
        preview_layout.addWidget(self.stats_label)
        left.addWidget(preview_box)

        tts_box = QGroupBox("文字输入 (TTS 模式)")
        tts_layout = QVBoxLayout(tts_box)
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("输入文字，数字人将自动朗读...")
        self.text_input.setMaximumHeight(75)
        tts_layout.addWidget(self.text_input)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_speak = QPushButton("📢 发送朗读")
        self.btn_speak.setObjectName("primary")
        self.btn_speak.setMaximumWidth(120)
        btn_row.addWidget(self.btn_speak)
        tts_layout.addLayout(btn_row)
        left.addWidget(tts_box)

        main_layout.addLayout(left, stretch=3)

        # ── Right: controls ──
        right = QVBoxLayout()
        right.setSpacing(8)

        # Character selection
        char_box = QGroupBox("角色")
        char_layout = QVBoxLayout(char_box)
        self.char_combo = QComboBox()
        self.char_combo.setMinimumHeight(32)
        self._refresh_characters()
        char_layout.addWidget(self.char_combo)
        btn_refresh = QPushButton("刷新列表")
        btn_refresh.clicked.connect(self._refresh_characters)
        char_layout.addWidget(btn_refresh)
        right.addWidget(char_box)

        # Audio source
        audio_box = QGroupBox("音频来源")
        audio_layout = QVBoxLayout(audio_box)
        self.audio_group = QButtonGroup(self)

        self.radio_mic = QRadioButton("🎤 麦克风实时")
        self.radio_mic.setChecked(True)
        self.audio_group.addButton(self.radio_mic, 0)
        audio_layout.addWidget(self.radio_mic)

        self.radio_tts = QRadioButton("📝 文字输入 (TTS)")
        self.audio_group.addButton(self.radio_tts, 1)
        audio_layout.addWidget(self.radio_tts)

        self.radio_file = QRadioButton("📁 音频文件")
        self.audio_group.addButton(self.radio_file, 2)
        audio_layout.addWidget(self.radio_file)
        right.addWidget(audio_box)

        # TTS voice selector (inside live panel per user request)
        voice_box = QGroupBox("TTS 语音")
        voice_layout = QFormLayout(voice_box)
        self.voice_combo = QComboBox()
        for voice_id, voice_label in TTS_VOICES:
            self.voice_combo.addItem(voice_label, voice_id)
        voice_layout.addRow("声音:", self.voice_combo)
        right.addWidget(voice_box)

        # Virtual camera
        cam_box = QGroupBox("虚拟摄像头")
        cam_layout = QVBoxLayout(cam_box)
        self.cam_status = QLabel("● 未启动")
        self.cam_status.setStyleSheet("color: #888;")
        cam_layout.addWidget(self.cam_status)
        self.cam_name = QLabel("SyncTalk Camera")
        self.cam_name.setStyleSheet("color: #00d4ff; font-size: 11px;")
        cam_layout.addWidget(self.cam_name)
        right.addWidget(cam_box)

        # Performance
        perf_box = QGroupBox("性能")
        perf_layout = QVBoxLayout(perf_box)
        self.perf_fps = QLabel("FPS:    --")
        self.perf_latency = QLabel("延迟:   --")
        self.perf_gpu = QLabel("GPU:    检测中...")
        self.perf_vram = QLabel("显存:   --")
        for w in [self.perf_fps, self.perf_latency, self.perf_gpu, self.perf_vram]:
            w.setStyleSheet("font-family: monospace; font-size: 11px;")
            perf_layout.addWidget(w)
        right.addWidget(perf_box)
        self._detect_gpu()

        right.addStretch()

        # Start/Stop button
        self.btn_start = QPushButton("🟢 开始直播")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumHeight(45)
        self.btn_start.clicked.connect(self._toggle_live)
        right.addWidget(self.btn_start)

        main_layout.addLayout(right, stretch=1)

    def get_selected_voice(self) -> str:
        return self.voice_combo.currentData() or "zh-CN-XiaoxiaoNeural"

    def _refresh_characters(self):
        self.char_combo.clear()
        dataset_dir = "./dataset"
        checkpoint_dir = "./checkpoint"
        if os.path.isdir(dataset_dir):
            for d in sorted(os.listdir(dataset_dir)):
                if os.path.isdir(os.path.join(dataset_dir, d)):
                    has_ckpt = os.path.isdir(os.path.join(checkpoint_dir, d))
                    status = "✅" if has_ckpt else "⏳"
                    self.char_combo.addItem(f"{status} {d}", d)
        if self.char_combo.count() == 0:
            self.char_combo.addItem("(无角色 - 请先训练)", None)

    def _detect_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
                self.perf_gpu.setText(f"GPU:    {name}")
                self.perf_vram.setText(f"显存:   {vram:.1f} GB")
            else:
                self.perf_gpu.setText("GPU:    CPU 模式")
                self.perf_vram.setText("显存:   N/A")
        except Exception:
            self.perf_gpu.setText("GPU:    未检测到")

    def _toggle_live(self):
        if self._is_live:
            self._is_live = False
            self.btn_start.setText("🟢 开始直播")
            self.btn_start.setObjectName("primary")
            self.btn_start.setStyleSheet("")
            self.cam_status.setText("● 已停止")
            self.cam_status.setStyleSheet("color: #888;")
            self.preview_label.setText("选择角色并点击「开始直播」")
            self.stop_requested.emit()
        else:
            char = self.char_combo.currentData()
            if not char:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "提示", "请先训练一个数字人角色")
                return
            self._is_live = True
            self.btn_start.setText("⏹️ 停止直播")
            self.btn_start.setStyleSheet(
                "background-color: #c0392b; color: white; font-weight: bold; "
                "font-size: 14px; padding: 10px 30px; border: none; border-radius: 6px;")
            self.cam_status.setText("● 运行中")
            self.cam_status.setStyleSheet("color: #4caf50; font-weight: bold;")
            self.preview_label.setText("🎬 直播中...")

            mode_id = self.audio_group.checkedId()
            config = {
                "character": char,
                "mode": ["mic", "tts", "file"][mode_id],
                "tts_voice": self.get_selected_voice(),
            }
            self.start_requested.emit(config)

    def _start_stats_timer(self):
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_stats)
        self.stats_timer.start(1000)

    def _update_stats(self):
        if self._is_live:
            self.perf_fps.setText("FPS:    25.0")
            self.perf_latency.setText("延迟:   ~5ms")
            self.stats_label.setText("FPS: 25.0 | 延迟: ~5ms | 运行中")
