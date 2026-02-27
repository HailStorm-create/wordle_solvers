import matplotlib.pyplot as plt


class LiveStats:
    def __init__(self):
        self.counts = {i: 0 for i in range(1, 7)}
        self.counts["fail"] = 0

        self.total_games = 0
        self.total_turns = 0
        self.avg_history = []

        plt.ion()
        self.fig, (self.ax_bar, self.ax_line) = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.suptitle("Wordle AI Stats (Live)")
        self._init_plots()
        self.update_interval = 1  # update every frame

    def _init_plots(self):
        self.labels = ["1", "2", "3", "4", "5", "6", "Fail"]
        self.bar_rects = self.ax_bar.bar(self.labels, [0]*7)
        self.ax_bar.set_title("Guess Distribution")
        self.ax_bar.set_xlabel("Guesses")
        self.ax_bar.set_ylabel("Games")

        self.line, = self.ax_line.plot([], [], linewidth=2)
        self.ax_line.set_title("Average Guesses Over Time")
        self.ax_line.set_xlabel("Games Played")
        self.ax_line.set_ylabel("Average Guesses")
        self.ax_line.set_xlim(0, 100)
        self.ax_line.set_ylim(3, 5)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.01)

    def record(self, turns, live_update=True):
        self.total_games += 1

        if turns is None:
            self.counts["fail"] += 1
        else:
            self.counts[turns] += 1
            self.total_turns += turns

        solved = sum(self.counts[i] for i in range(1, 7))
        avg = self.total_turns / solved if solved else 0
        self.avg_history.append(avg)

        if live_update and self.total_games % self.update_interval == 0:
            self._refresh_plots()

    def _refresh_plots(self):
        values = [
            self.counts[1], self.counts[2], self.counts[3],
            self.counts[4], self.counts[5], self.counts[6],
            self.counts["fail"]
        ]
        for rect, val in zip(self.bar_rects, values):
            rect.set_height(val)
        self.ax_bar.set_ylim(0, max(values) * 1.1 + 1)

        self.line.set_data(range(len(self.avg_history)), self.avg_history)
        self.ax_line.set_xlim(0, max(len(self.avg_history), 100))
        if self.avg_history:
            min_avg = min(self.avg_history)
            max_avg = max(self.avg_history)
            self.ax_line.set_ylim(max(0, min_avg - 0.5), max_avg + 0.5)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def show_final(self):
        plt.ioff()
        self._refresh_plots()
        self.fig.suptitle("Wordle AI Stats (Final)")
        plt.show()
