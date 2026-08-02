import sys

from PySide6.QtWidgets import QApplication

from mahjong.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("血流麻将")
    window = MainWindow()
    window.show()
    window.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
