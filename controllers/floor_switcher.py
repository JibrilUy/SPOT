# controllers/floor_switcher.py

class FloorSwitcher:
    def __init__(self, stack, count_window, before_show_floor=None):
        self.stack = stack
        self.count_window = count_window
        self.before_show_floor = before_show_floor

        self.floor_windows = {}

    def register_floor(self, floor_num, window):
        self.floor_windows[floor_num] = window

    def show_floor(self, floor_num):
        if floor_num not in self.floor_windows:
            return
        # This single gate handles dashboard navigation AND floor nav links.
        if self.before_show_floor and not self.before_show_floor(floor_num):
            return
        self.stack.setCurrentWidget(self.floor_windows[floor_num])

    def show_dashboard(self, dashboard_window):
        self.stack.setCurrentWidget(dashboard_window)

    def show_login(self, login_window):
        self.stack.setCurrentWidget(login_window)