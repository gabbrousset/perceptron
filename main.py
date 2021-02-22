import numpy as np


def perceptron(t_set, t_labels):
    theta = np.zeros((1, 2))
    theta = [[-1, 1]]
    offset = -1
    changed = True
    mistakes, loops = 0, 0
    while changed:
        loops += 1
        changed = False
        for i in range(len(t_set)):
            if t_labels[i] * (np.dot(theta, t_set[i]) + offset) <= 0:
                theta += np.multiply(t_labels[i], t_set[i])
                offset += t_labels[i]
                changed = True
                mistakes += 1
                print(f'{loops}.- theta: {theta}, offset: {offset}, point: {t_set[i]}')
    return {'mistakes': mistakes, 'loops': loops}


x_1 = [[-4, 2], 1]
x_2 = [[-2, 1], 1]
x_3 = [[-1, -1], -1]
x_4 = [[2, 2], -1]
x_5 = [[1, -2], -1]

data = [x_1, x_2, x_3, x_4, x_5]


def main():
    training_set = []
    labels = []
    for d in data:
        training_set.append(d[0])
        labels.append(d[1])
    stats = perceptron(training_set, labels)
    print('\n', ', '.join([f'{k}: {v}' for k, v in stats.items()]), sep='')


main()
