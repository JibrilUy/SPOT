# controllers/floor_switcher.py

class FloorSwitcher:
    def __init__(self, stack, count_window):
        self.stack = stack
        self.count_window = count_window

        self.floor_windows = {}

    def register_floor(self, floor_num, window):
        self.floor_windows[floor_num] = window

    def show_floor(self, floor_num):
        if floor_num in self.floor_windows:
            self.stack.setCurrentWidget(
                self.floor_windows[floor_num]
            )
        else:
            self.stack.setCurrentWidget(
                self.count_window
            )

    def show_dashboard(self, dashboard_window):
        self.stack.setCurrentWidget(dashboard_window)

    def show_login(self, login_window):
        self.stack.setCurrentWidget(login_window)