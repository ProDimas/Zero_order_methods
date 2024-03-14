import numpy as np
import pandas as pd
from hooke_jeeves_solver import hooke_jeeves_solver
from nelder_mead_solver import nelder_mead_solver

def main():
    # Task 1: Hooke-Jeeves method
    f = lambda x: (x[0] - 5) ** 2 + 2 * x[1] ** 2
    x0 = np.array([8, 12])
    dx = np.array([0.6, 0.8])
    # If x0 is excluded from basic points in task, then this number must be +1 the number mentioned in task (because here, x0 is included)
    number_of_basic_points = 4

    task1_solver = hooke_jeeves_solver(x0, dx, f)
    task1_solver.solve_for_points(number_of_basic_points)
    print('Task 1')
    print('-' * 30)
    print(f'First {number_of_basic_points} basic points (including start point x_0): ')
    for i, (x, y) in enumerate(task1_solver.basic_points):
        print(f'x_{i}: {x}; f(x_{i}): {y}')
    print(f'Delta x end criteria: {task1_solver.compute_delta_x_end_criteria()}')
    print(f'Relative end criteria: {task1_solver.compute_relative_end_criteria()}')
    print(f'Current basic point:')
    print(f'x_cb: {task1_solver.x_current_basic[0]}; f(x_cb): {task1_solver.x_current_basic[1]}')
    print(f'Current dx: {task1_solver.dx}')
    print('-' * 30)

    # Task 2: Nelder-Mead method
    f = lambda x: 5 * (x[0] - 4) ** 2 + (x[1] - 3) ** 2
    x1 = np.array([7, 4])
    x2 = np.array([5, 8])
    x3 = np.array([7, 9])
    a = 1
    b = 0.5
    g = 2
    M = 3
    number_of_reductions = 2

    task2_solver = nelder_mead_solver([x1, x2, x3], a, b, g, M, f)
    task2_solver.solve_for_reductions(number_of_reductions)
    print('Task 2')
    print('-' * 30)
    print(f'First {number_of_reductions} reductions (including first simplex after last reduction): ')
    output = pd.DataFrame(columns=['x_h', 'f(x_h)', 'x_g', 'f(x_g)', 'x_l', 'f(x_l)'])
    for r in task2_solver.reductions:
        for s in r:
            output.loc[len(output) + 1] = [f'x_{s[0] + 1}: {task2_solver.x[s[0]][0]}', f'f(x_{s[0] + 1}): {task2_solver.x[s[0]][1]}',
                                        f'x_{s[1] + 1}: {task2_solver.x[s[1]][0]}', f'f(x_{s[1] + 1}): {task2_solver.x[s[1]][1]}',
                                        f'x_{s[2] + 1}: {task2_solver.x[s[2]][0]}', f'f(x_{s[2] + 1}): {task2_solver.x[s[2]][1]}']
    
    print(output)
    reductions_sizes = list(map(len, task2_solver.reductions[:-1]))
    for i in range(1, len(reductions_sizes)):
        reductions_sizes[i] += reductions_sizes[i - 1]
    print(f'Reductions after (numbers indicate last simplex before each reduction): {reductions_sizes}')
    print(f'End criteria: {task2_solver.compute_end_criteria()}')
    print('-' * 30)

if __name__ == '__main__':
    main()
