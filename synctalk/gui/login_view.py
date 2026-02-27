"""Login / Register view."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QStackedWidget, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal


class LoginView(QWidget):
    """Login and registration interface."""

    login_success = pyqtSignal(dict)

    def __init__(self, auth_client=None):
        super().__init__()
        self.auth_client = auth_client
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(15)
        layout.setContentsMargins(80, 40, 80, 40)

        title = QLabel("SyncTalk")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("AI 数字人直播助手")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_login_form())
        self.stack.addWidget(self._build_register_form())
        layout.addWidget(self.stack)

    def _build_login_form(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        layout.addWidget(QLabel("邮箱"))
        self.login_email = QLineEdit()
        self.login_email.setPlaceholderText("your@email.com")
        layout.addWidget(self.login_email)

        layout.addWidget(QLabel("密码"))
        self.login_password = QLineEdit()
        self.login_password.setPlaceholderText("输入密码")
        self.login_password.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.login_password)

        layout.addSpacing(10)

        btn_login = QPushButton("登 录")
        btn_login.setObjectName("primary")
        btn_login.clicked.connect(self._on_login)
        layout.addWidget(btn_login)

        row = QHBoxLayout()
        row.addStretch()
        lbl = QLabel("没有账号？")
        lbl.setObjectName("subtitle")
        row.addWidget(lbl)
        btn_switch = QPushButton("注册")
        btn_switch.setFlat(True)
        btn_switch.setStyleSheet("color: #00d4ff; border: none; font-weight: bold;")
        btn_switch.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        row.addWidget(btn_switch)
        row.addStretch()
        layout.addLayout(row)

        return w

    def _build_register_form(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(10)

        layout.addWidget(QLabel("昵称"))
        self.reg_name = QLineEdit()
        self.reg_name.setPlaceholderText("你的昵称")
        layout.addWidget(self.reg_name)

        layout.addWidget(QLabel("邮箱"))
        self.reg_email = QLineEdit()
        self.reg_email.setPlaceholderText("your@email.com")
        layout.addWidget(self.reg_email)

        layout.addWidget(QLabel("密码"))
        self.reg_password = QLineEdit()
        self.reg_password.setPlaceholderText("至少 6 位")
        self.reg_password.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.reg_password)

        layout.addSpacing(10)

        btn_reg = QPushButton("注 册")
        btn_reg.setObjectName("primary")
        btn_reg.clicked.connect(self._on_register)
        layout.addWidget(btn_reg)

        row = QHBoxLayout()
        row.addStretch()
        lbl = QLabel("已有账号？")
        lbl.setObjectName("subtitle")
        row.addWidget(lbl)
        btn_back = QPushButton("登录")
        btn_back.setFlat(True)
        btn_back.setStyleSheet("color: #00d4ff; border: none; font-weight: bold;")
        btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        row.addWidget(btn_back)
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
