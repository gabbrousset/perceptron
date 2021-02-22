import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, initial_theta=None, initial_offset=0, margins=False, through_origin=False, max_loops=10000,
                 visual=False, debugging=False):
        if initial_theta is None:
            initial_theta = [0.0, 0.0]
        self.theta = np.array(initial_theta)
        self.offset = initial_offset
        self.through_origin = through_origin
        self.mistakes, self.loops = 0, 0
        self.changed = True
        self.converged = True
        self.max_loops = max_loops
        self.margins = margins
        self.visual = visual
        self.debugging = debugging

        if self.visual:
            self.fig = plt.figure()
            self.fig.canvas.set_window_title('Perceptron')
            plt.grid(True, color='lightgrey', linewidth=0.5, zorder=0)
            plt.axhline(0, color='grey', linewidth=3, zorder=10)
            plt.axvline(0, color='grey', linewidth=3, zorder=10)

    @staticmethod
    def format_data(data):
        training_set = []
        labels = []

        for d in data:
            training_set.append(d[0])
            labels.append(d[1])

        return np.array(training_set), np.array(labels)

    def format_results(self):
        stats = {
            'theta': [v for v in self.theta],
            'offset': self.offset,
            'converged': self.converged,
            'mistakes': self.mistakes,
            'loops': self.loops
        }

        return '\n'.join([f"{str(k + ':'):>10} {str(v):>12}" for k, v in stats.items()])

    @staticmethod
    def graph_points(points, labels):
        x, y = points.T

        colors = ['r' if label > 0 else 'b' for label in labels]
        plt.scatter(x, y, color=colors, zorder=30)

    def graph_theta(self, horizontal, vertical):

        main1, main2 = standard_line_to_points(self.theta[0], self.theta[1], self.offset)

        if main1 != main2:
            plt.axline(main1, main2, color='darkgreen' if self.converged else 'darkviolet', linewidth=2, zorder=20)

        if self.converged and self.margins:
            margin1_x1, margin1_x2 = standard_line_to_points(self.theta[0], self.theta[1], self.offset - 1)
            plt.axline(margin1_x1, margin1_x2, color='green', zorder=20)

            margin2_x1, margin2_x2 = standard_line_to_points(self.theta[0], self.theta[1], self.offset + 1)
            plt.axline(margin2_x1, margin2_x2, color='green', zorder=20)

        plt.axis((-horizontal, horizontal, -vertical, vertical))

    def train(self, t_set, t_labels):
        if self.visual:
            self.graph_points(t_set, t_labels)

        while self.changed:
            # Prevents an infinite loop
            if self.loops >= self.max_loops:
                self.converged = False
                break

            self.loops += 1
            self.changed = False
            for i in range(len(t_set)):
                if t_labels[i] * (np.dot(self.theta, t_set[i]) + self.offset) <= 0:
                    self.theta += np.multiply(t_labels[i], t_set[i])
                    if not self.through_origin:
                        self.offset += t_labels[i]
                    self.changed = True
                    self.mistakes += 1
                    if self.debugging:
                        print(
                            f'{self.loops:>5}.- point: {str([v for v in t_set[i]]):<10} theta: {str([v for v in self.theta]):<10} offset: {self.offset:>3}'
                        )
        if self.debugging:
            print('\n\n', self.format_results(), sep='')
        if self.visual:
            x, y = t_set.T[0], t_set.T[1]
            left = min(x)
            right = max(x)
            down = min(y)
            up = max(y)
            self.graph_theta(max(abs(left), right)+1, max(abs(down), up)+1)
            plt.show()


def standard_line_to_points(a, b, c):
    if b != 0:
        point1 = (-b, a - c / b)
        point2 = (b, -a - c / b)
    else:
        point1 = (-b - c, a)
        point2 = (b - c, -a)
    return point1, point2


def main():
    x_1 = [[np.cos(np.pi), 0], 1]
    x_2 = [[0, np.cos(np.pi * 2)], 1]

    t_data = [x_1, x_2]

    perceptron = Perceptron(visual=True, debugging=True, through_origin=True)
    training_set, training_labels = perceptron.format_data(t_data)
    perceptron.train(training_set, training_labels)


main()
