# hydro_mpc/core/first_order_filter.py

class FirstOrderFilter:
    def __init__(self, alpha: float = 0.2, initial_value: float = 0.0):
        self.alpha = alpha
        self.prev = initial_value
        self.initialized = False

    def filter(self, new_value: float) -> float:
        if not self.initialized:
            self.prev = new_value
            self.initialized = True
        smoothed = self.alpha * new_value + (1.0 - self.alpha) * self.prev
        self.prev = smoothed
        return smoothed
