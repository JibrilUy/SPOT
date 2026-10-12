from PyQt6.QtCore import QTimer
from datetime import datetime


class DateTimeManager:

    def __init__(self, ui):

        self.ui = ui

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_datetime)

        self.update_datetime()
        self.timer.start(1000)

    def update_datetime(self):

        now = datetime.now()

        self.ui.datetime_current_date.setText(
            now.strftime("%B %d, %Y")
        )

        self.ui.datetime_day_of_the_week.setText(
            now.strftime("%A")
        )

        self.ui.datetime_current_time.setText(
            now.strftime("%I:%M:%S")
        )

        self.ui.datetime_meridiem_indicator.setText(
            now.strftime("%p")
        )