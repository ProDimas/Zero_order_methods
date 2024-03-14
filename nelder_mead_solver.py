from typing import Callable, TypeAlias
from copy import deepcopy
from operator import itemgetter
from collections import Counter
from numpy.typing import NDArray
import numpy as np

Point: TypeAlias = NDArray[np.float64]

class nelder_mead_solver:
    def __init__(self, simplex: list[Point], a: float, b: float, g: float, M: float, func: Callable[[Point], float]):
        # Rounding precision
        self.__after_comma: int = 3

        self.__a: float = np.round(a, self.__after_comma)
        self.__b: float = np.round(b, self.__after_comma)
        self.__g: float = np.round(g, self.__after_comma)
        self.__M: float = np.round(M, self.__after_comma)
        
        self.__undecorated_func: Callable[[Point], float] = deepcopy(func)
        self.__no_point_func: Callable[[Point], float] = lambda x: np.round(self.__undecorated_func(x), self.__after_comma)
        self.__func: Callable[[Point], tuple[Point, float]] = lambda x: (x, self.__no_point_func(x))

        simplex = list(map(lambda p: self.__func(p), simplex))

        # The convention is that index i of point in this list indicates that it is point x_(i + 1)
        self.__x: list[tuple[Point, float]] = simplex
        enum_simplex = list(enumerate(simplex))
        sorted_simplex = sorted(enum_simplex, key=lambda x: x[1][1], reverse=True)
        self.__reductions: list[list[tuple[int, int, int]]] = [[tuple(map(itemgetter(0), sorted_simplex))]]
    
    def solve_for_reductions(self, number_of_reductions: int):
        s_counter = Counter(self.__reductions[-1][-1])
        while len(self.__reductions) != number_of_reductions + 1:
            s = self.__reductions[-1][-1]

            x_c = np.round((self.__x[s[1]][0] + self.__x[s[2]][0]) / 2, self.__after_comma)
            x_new = np.round(self.__x[s[0]][0] + (1 + self.__a) * (x_c - self.__x[s[0]][0]), self.__after_comma)
            new = self.__func(x_new)

            theta = self.__locate_new(new[1])
            if theta == self.__a:
                x = new
            else:
                arg = np.round(self.__x[s[0]][0] + (1 + theta) * (x_c - self.__x[s[0]][0]), self.__after_comma)
                x = self.__func(arg)
            
            self.__x.append(x)
            new_simplex = [s[1], s[2], len(self.__x) - 1]
            sorted_new_simplex = sorted(new_simplex, key=lambda i: self.__x[i][1], reverse=True)
            self.__reductions[-1].append(sorted_new_simplex)
            new_simplex_counter = Counter(sorted_new_simplex)
            for e in set(s_counter):
                if not(e in new_simplex_counter):
                    del s_counter[e]
            
            s_counter.update(new_simplex_counter)
            for e in s_counter:
                if s_counter[e] == self.__M:
                    x1 = np.round((self.__x[sorted_new_simplex[2]][0] + self.__x[sorted_new_simplex[0]][0]) / 2, self.__after_comma)
                    x2 = np.round((self.__x[sorted_new_simplex[2]][0] + self.__x[sorted_new_simplex[1]][0]) / 2, self.__after_comma)
                    vertex1 = self.__func(x1)
                    vertex2 = self.__func(x2)
                    self.__x.append(vertex1)
                    self.__x.append(vertex2)
                    new_simplex = [sorted_new_simplex[2], len(self.__x) - 1, len(self.__x) - 2]
                    sorted_new_simplex = sorted(new_simplex, key=lambda i: self.__x[i][1], reverse=True)
                    self.__reductions.append([sorted_new_simplex])
                    s_counter = Counter(self.__reductions[-1][-1])
                    break

    def solve_for_precision(self, precision: float):
        s_counter = Counter(self.__reductions[-1][-1])
        while self.compute_end_criteria() > precision:
            s = self.__reductions[-1][-1]

            x_c = np.round((self.__x[s[1]][0] + self.__x[s[2]][0]) / 2, self.__after_comma)
            x_new = np.round(self.__x[s[0]][0] + (1 + self.__a) * (x_c - self.__x[s[0]][0]), self.__after_comma)
            new = self.__func(x_new)

            theta = self.__locate_new(new[1])
            if theta == self.__a:
                x = new
            else:
                arg = np.round(self.__x[s[0]][0] + (1 + theta) * (x_c - self.__x[s[0]][0]), self.__after_comma)
                x = self.__func(arg)
            
            self.__x.append(x)
            new_simplex = [s[1], s[2], len(self.__x) - 1]
            sorted_new_simplex = sorted(new_simplex, key=lambda i: self.__x[i][1], reverse=True)
            self.__reductions[-1].append(sorted_new_simplex)

            new_simplex_counter = Counter(sorted_new_simplex)
            for e in set(s_counter):
                if not(e in new_simplex_counter):
                    del s_counter[e]
            
            s_counter.update(new_simplex_counter)
            for e in s_counter:
                if s_counter[e] == self.__M:
                    x1 = np.round((self.__x[sorted_new_simplex[2]][0] + self.__x[sorted_new_simplex[0]][0]) / 2, self.__after_comma)
                    x2 = np.round((self.__x[sorted_new_simplex[2]][0] + self.__x[sorted_new_simplex[1]][0]) / 2, self.__after_comma)
                    vertex1 = self.__func(x1)
                    vertex2 = self.__func(x2)
                    self.__x.append(vertex1)
                    self.__x.append(vertex2)
                    new_simplex = [sorted_new_simplex[2], len(self.__x) - 1, len(self.__x) - 2]
                    sorted_new_simplex = sorted(new_simplex, key=lambda i: self.__x[i][1], reverse=True)
                    self.__reductions.append([sorted_new_simplex])
                    s_counter = Counter(self.__reductions[-1][-1])
                    break

    def __locate_new(self, y_new: float) -> float:
        s = self.__reductions[-1][-1]
        if y_new >= self.__x[s[0]][1]:
            return -self.__b
        elif y_new >= self.__x[s[1]][1]:
            return self.__b
        elif y_new > self.__x[s[2]][1]:
            return self.__a
        else:
            return self.__g
        
    @property
    def x(self):
        return self.__x

    @property
    def reductions(self):
        return self.__reductions

    def compute_end_criteria(self) -> float:
        simplex = self.__reductions[-1][-1]
        x_c = 0
        for i in simplex:
            x_c += self.__x[i][0] / len(simplex)

        c = self.__func(np.round(x_c, self.__after_comma))
        f = np.array(list(map(lambda i: self.__x[i][1], simplex)))
        f = np.round((f - c[1]) ** 2, self.__after_comma)
        result = np.round(np.sqrt(np.sum(f) / len(simplex)), self.__after_comma)
        return result
