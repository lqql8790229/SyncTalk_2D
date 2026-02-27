"""Login / Register view — compact centered layout."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QStackedWidget, QMessageBox, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal


class LoginView(QWidget):
    """Compact, centered login and registration interface."""

    login_success = pyqtSignal(dict)

    def __init__(self, auth_client=None):
        super().__init__()
        self.auth_client = auth_client
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setFixedWidth(380)
        card.setStyleSheet(
            "QFrame { background-color: #16213e; border-radius: 12px; }")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(12)
        card_layout.setContentsMargins(36, 32, 36, 28)

        title = QLabel("SyncTalk")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)

        subtitle = QLabel("AI 数字人直播助手")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(subtitle)

        card_layout.addSpacing(8)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_login_form())
        self.stack.addWidget(self._build_register_form())
        card_layout.addWidget(self.stack)

        outer.addWidget(card)

    def _build_login_form(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("邮箱"))
        self.login_email = QLineEdit()
        self.login_email.setPlaceholderText("your@email.com")
        self.login_email.setFixedHeight(36)
        layout.addWidget(self.login_email)

        layout.addSpacing(4)
        layout.addWidget(QLabel("密码"))
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("输入密码")
        self.login_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.login_password.setFixedHeight(36)
        layout.addWidget(self.login_password)

        layout.addSpacing(10)
        btn = QPushButton("登 录")
        btn.setObjectName("primary")
        btn.setFixedHeight(38)
        btn.clicked.connect(self._on_login)
        layout.addWidget(btn)

        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.addStretch()
        row.addWidget(QLabel("没有账号？"))
        link = QPushButton("注册")
        link.setFlat(True)
        link.setCursor(Qt.CursorShape.PointingHandCursor)
        link.setStyleSheet("color:#00d4ff; border:none; font-weight:bold; padding:0;")
        link.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        row.addWidget(link)
        row.addStretch()
        layout.addLayout(row)
        return w

    def _build_register_form(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("昵称"))
        self.reg_name = QLineEdit()
        self.reg_name.setPlaceholderText("你的昵称")
        self.reg_name.setFixedHeight(36)
        layout.addWidget(self.reg_name)

        layout.addSpacing(4)
        layout.addWidget(QLabel("邮箱"))
        self.reg_email = QLineEdit()
        self.reg_email.setPlaceholderText("your@email.com")
        self.reg_email.setFixedHeight(36)
        layout.addWidget(self.reg_email)

        layout.addSpacing(4)
        layout.addWidget(QLabel("密码"))
        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("至少 6 位")
        self.reg_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.reg_password.setFixedHeight(36)
        layout.addWidget(self.reg_password)

        layout.addSpacing(10)
        btn = QPushButton("注 册")
        btn.setObjectName("primary")
        btn.setFixedHeight(38)
        btn.clicked.connect(self._on_register)
        layout.addWidget(btn)

        row = QHBoxLayout()
        row.setContentsMargins(0, 4, 0, 0)
        row.addStretch()
        row.addWidget(QLabel("已有账号？"))
        link = QPushButton("登录")
        link.setFlat(True)
        link.setCursor(Qt.CursorShape.PointingHandCursor)
        link.setStyleSheet("color:#00d4ff; border:none; font-weight:bold; padding:0;")
        link.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        row.addWidget(link)
        row.addStretch()
        layout.addLayout(row)
        return w

    def _on_login(self):
        email = self.login_email.text().strip()
        password = self.login_password.text()
        if not email or not password:
            QMessageBox.warning(self, "提示", "请填写邮箱和密码")
            return
        try:
            if self.auth_client:
                data = self.auth_client.login(email, password)
            else:
                data = {"user_id": "local", "email": email, "plan": "free"}
            self.login_success.emit(data)
        except Exception as e:
            QMessageBox.critical(self, "登录失败", str(e))

    def _on_register(self):
        name = self.reg_name.text().strip()
        email = self.reg_email.text().strip()
        password = self.reg_password.text()
        if not email or not password:
            QMessageBox.warning(self, "提示", "请填写必要信息")
            return
        try:
            if self.auth_client:
                data = self.auth_client.register(email, password, name)
            else:
                data = {"user_id": "local", "email": email, "plan": "free"}
            self.login_success.emit(data)
        except Exception as e:
            QMessageBox.critical(self, "注册失败", str(e))
