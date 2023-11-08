import time


class TimeEstimator:
    def __init__(self, total_iter, step_size):
        self.avg_time_window = []  # window-based average
        self.exp_avg_time = None  # exponential moving average
        self.alpha = 0.7  # for exponential moving average

        self.last_time = time.time()  # would not be accurate for the first iteration but well
        self.total_iter = total_iter
        self.step_size = step_size

        self.buffering_exp = True

    # call this at a fixed interval
    # does not have to be every step
    def update(self):
        curr_time = time.time()
        time_per_iter = curr_time - self.last_time
        self.last_time = curr_time

        self.avg_time_window.append(time_per_iter)

        if self.buffering_exp:
            if self.exp_avg_time is not None:
                # discard the first iteration call to not pollute the ema
                self.buffering_exp = False
            self.exp_avg_time = time_per_iter
        else:
            self.exp_avg_time = self.alpha * self.exp_avg_time + (1 - self.alpha) * time_per_iter

    def get_est_remaining(self, it):
        if self.exp_avg_time is None:
            return 0

        remaining_iter = self.total_iter - it
        return remaining_iter * self.exp_avg_time / self.step_size

    def get_and_reset_avg_time(self):
        avg = sum(self.avg_time_window) / len(self.avg_time_window) / self.step_size
        self.avg_time_window = []
        return avg
