"""Main SyncTalk desktop application."""

import sys
import logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QStatusBar,
    QGroupBox, QFormLayout, QComboBox, QLineEdit, QPushButton,
    QMessageBox, QStackedWidget,
)
from PyQt6.QtCore import Qt

from .styles import STYLESHEET
from .login_view import LoginView
from .live_view import LiveView
from .train_view import TrainView

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window with login/logout and tab navigation."""

    def __init__(self, auth_client=None):
        super().__init__()
        self.auth_client = auth_client
        self.setWindowTitle("SyncTalk - AI 数字人直播助手")
        self.setMinimumSize(900, 650)
        self.resize(1050, 720)

        self._user_data = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        self.main_layout = QVBoxLayout(central)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)

        self.login_view = LoginView(self.auth_client)
        self.login_view.login_success.connect(self._on_login_success)
        self.stack.addWidget(self.login_view)

        self.app_container = QWidget()
        app_layout = QVBoxLayout(self.app_container)
        app_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.live_view = LiveView()
        self.train_view = TrainView()
        self.settings_view = self._build_settings_view()

        self.tabs.addTab(self.live_view, "🎬 直播控台")
        self.tabs.addTab(self.train_view, "🎓 我的数字人")
        self.tabs.addTab(self.settings_view, "⚙️ 设置")
        app_layout.addWidget(self.tabs)
        self.stack.addWidget(self.app_container)

        if self.auth_client and self.auth_client.is_authenticated:
            self._on_login_success({
                "email": self.auth_client.email,
                "plan": self.auth_client.plan,
                "user_id": self.auth_client.user_id,
            })
        else:
            self.stack.setCurrentWidget(self.login_view)

        self.status_bar = QStatusBar()
        self.status_bar.setStyleSheet(
            "background-color: #0d0d1a; color: #888; padding: 4px;")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def _on_login_success(self, data):
        self._user_data = data
        email = data.get("email", "")
        plan = data.get("plan", "free")
        plan_labels = {"free": "免费版", "pro": "Pro", "business": "Business"}

        self.settings_email.setText(email)
        self.settings_plan.setText(plan_labels.get(plan, plan))

        self.stack.setCurrentWidget(self.app_container)
        self.status_bar.showMessage(
            f"已登录: {email} | 方案: {plan_labels.get(plan, plan)} | SyncTalk v1.0.0")

    def _on_logout(self):
        reply = QMessageBox.question(
            self, "确认登出", "确定要退出当前账号吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        if self.auth_client:
            self.auth_client.logout()
        self._user_data = None

        self.login_view.login_email.clear()
        self.login_view.login_password.clear()
        self.login_view.stack.setCurrentIndex(0)

        self.settings_email.setText("未登录")
        self.settings_plan.setText("--")

        self.stack.setCurrentWidget(self.login_view)
        self.status_bar.showMessage("已登出")

    def _build_settings_view(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("设置")
        header.setObjectName("sectionTitle")
        layout.addWidget(header)

        account_box = QGroupBox("账号信息")
        account_layout = QFormLayout(account_box)
        self.settings_email = QLabel("未登录")
        self.settings_plan = QLabel("--")
        account_layout.addRow("邮箱:", self.settings_email)
        account_layout.addRow("方案:", self.settings_plan)

        btn_logout = QPushButton("退出登录")
        btn_logout.setObjectName("danger")
        btn_logout.setMaximumWidth(120)
        btn_logout.clicked.connect(self._on_logout)
        account_layout.addRow("", btn_logout)
        layout.addWidget(account_box)

        perf_box = QGroupBox("性能设置")
        perf_layout = QFormLayout(perf_box)
        self.res_combo = QComboBox()
        self.res_combo.addItems(["328px (高清)", "160px (标清)"])
        perf_layout.addRow("默认分辨率:", self.res_combo)

        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["25 FPS", "30 FPS", "15 FPS"])
        perf_layout.addRow("帧率:", self.fps_combo)
        layout.addWidget(perf_box)

        server_box = QGroupBox("服务器")
        server_layout = QFormLayout(server_box)
        self.server_url_input = QLineEdit("http://localhost:9000")
        server_layout.addRow("云端 API:", self.server_url_input)
        layout.addWidget(server_box)

        layout.addStretch()

        about = QLabel("SyncTalk v1.0.0 | AI 数字人直播助手")
        about.setObjectName("subtitle")
        about.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(about)

        return w


class SyncTalkApp:
    """Application entry point."""

    def __init__(self, server_url: str = "http://localhost:9000",
                 skip_login: bool = False):
        self.server_url = server_url
        self.skip_login = skip_login

    def run(self):
        app = QApplication(sys.argv)
        app.setApplicationName("SyncTalk")
        app.setStyleSheet(STYLESHEET)

        auth_client = None
        if not self.skip_login:
            try:
                from ..auth.client import AuthClient
                auth_client = AuthClient(self.server_url)
            except Exception:
                pass

        window = MainWindow(auth_client)
        window.show()

        if self.skip_login:
            window._on_login_success(
                {"email": "local@synctalk", "plan": "pro"})

        return app.exec()
