"""Global stylesheet for SyncTalk GUI."""

STYLESHEET = """
QMainWindow {
    background-color: #1a1a2e;
}

QWidget {
    color: #e0e0e0;
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}

QLabel#title {
    font-size: 22px;
    font-weight: bold;
    color: #00d4ff;
}

QLabel#subtitle {
    font-size: 13px;
    color: #888;
}

QLabel#sectionTitle {
    font-size: 15px;
    font-weight: bold;
    color: #ffffff;
    padding: 8px 0;
}

QLabel#statusOk {
    color: #4caf50;
    font-weight: bold;
}

QLabel#statusError {
    color: #f44336;
    font-weight: bold;
}

QPushButton {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 8px 20px;
    color: #e0e0e0;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #0f3460;
    border-color: #00d4ff;
}

QPushButton:pressed {
    background-color: #00d4ff;
    color: #1a1a2e;
}

QPushButton#primary {
    background-color: #00d4ff;
    color: #1a1a2e;
    font-weight: bold;
    font-size: 14px;
    padding: 10px 30px;
    border: none;
}

QPushButton#primary:hover {
    background-color: #00b8d9;
}

QPushButton#danger {
    background-color: #c0392b;
    color: white;
    border: none;
}

QPushButton#danger:hover {
    background-color: #e74c3c;
}

QLineEdit {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e0e0e0;
    font-size: 13px;
}

QLineEdit:focus {
    border-color: #00d4ff;
}

QTextEdit {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 8px;
    color: #e0e0e0;
}

QComboBox {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 6px;
    padding: 6px 12px;
    color: #e0e0e0;
}

QComboBox::drop-down {
    border: none;
}

QProgressBar {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 6px;
    text-align: center;
    color: white;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #00d4ff;
    border-radius: 5px;
}

QTabWidget::pane {
    border: 1px solid #0f3460;
    background-color: #1a1a2e;
    border-radius: 6px;
}

QTabBar::tab {
    background-color: #16213e;
    color: #888;
    padding: 10px 25px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}

QTabBar::tab:selected {
    background-color: #0f3460;
    color: #00d4ff;
}

QTabBar::tab:hover {
    color: #ffffff;
}

QGroupBox {
    border: 1px solid #0f3460;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #00d4ff;
}
"""
