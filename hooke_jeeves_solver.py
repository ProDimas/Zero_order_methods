from typing import Callable, TypeAlias
from copy import deepcopy
from numpy.typing import NDArray
import numpy as np

Point: TypeAlias = NDArray[np.float64]

class hooke_jeeves_solver:
    DELTA_X_END_CRITERIA: int = 0

    RELATIVE_END_CRITERIA: int = 1

    def __init__(self, x0: Point, dx: Point, func: Callable[[Point], float]):
        # Rounding precision
        self.__after_comma: int = 3
        
        # Number to make less dx vector
        self.__alpha: float = 2

        self.__x0: Point = x0.astype(np.float64)
        self.__dx: Point = dx.astype(np.float64)
        self.__undecorated_func: Callable[[Point], float] = deepcopy(func)
        self.__no_point_func: Callable[[Point], float] = lambda x: np.round(self.__undecorated_func(x), self.__after_comma)
        self.__func: Callable[[Point], tuple[Point, float]] = lambda x: (x, self.__no_point_func(x))
        self.__basic_points: list[tuple[Point, float]] = [self.__func(self.__x0)]
    
    def solve_for_points(self, number_of_basic_points: int):
        # As we just want some number of basic point, we don't want solve task with any precision at all.
        # Defined as -1, this precision is only used as passed argument to __exploratory_search().
        # Be aware that this method of solving without precision can fall in loop if start point x0 is already point of minimum.
        precision = -1
        sucs, self.__x_current_basic = self.__exploratory_search(*self.__basic_points[-1], precision)
        if not sucs:
            return
        
        while len(self.__basic_points) != number_of_basic_points:
            x_working = self.__func(2 * self.__x_current_basic[0] - self.__basic_points[-1][0])
            sucs, x_after_working = self.__exploratory_search(*x_working, precision)
            if not sucs:
                # Exception because no information about actions in such case has been found
                raise Exception('Exploring search applied to x_working hasn\'t found next point - undefined behaviour')
            
            if x_after_working[1] < self.__x_current_basic[1]:
                self.__basic_points.append(self.__x_current_basic)
                self.__x_current_basic = x_after_working
            else:
                self.__dx = np.round(self.__dx / self.__alpha, self.__after_comma)
                sucs, self.__x_current_basic = self.__exploratory_search(*self.__basic_points[-1], precision)
                if not sucs:
                    return

    def solve_for_precision(self, precision: float, criteria: int):
        def check_criteria():
            if criteria == hooke_jeeves_solver.DELTA_X_END_CRITERIA:
                return self.compute_delta_x_end_criteria() <= precision
            elif criteria == hooke_jeeves_solver.RELATIVE_END_CRITERIA:
                val1, val2 = self.compute_relative_end_criteria()
                return (val1 <= precision) and (val2 <= precision)
            else:
                raise Exception('Unknown end criteria')
            
        sucs, self.__x_current_basic = self.__exploratory_search(*self.__basic_points[-1], precision)
        if not sucs:
            return
        
        end = check_criteria()
        while not end:
            x_working = self.__func(2 * self.__x_current_basic[0] - self.__basic_points[-1][0])
            sucs, x_after_working = self.__exploratory_search(*x_working, precision)
            if not sucs:
                # Exception because no information about actions in such case has been found
                raise Exception('Exploring search applied to x_working hasn\'t found next point - undefined behaviour')
            
            if x_after_working[1] < self.__x_current_basic[1]:
                self.__basic_points.append(self.__x_current_basic)
                self.__x_current_basic = x_after_working
            else:
                self.__dx = np.round(self.__dx / self.__alpha, self.__after_comma)
                sucs, self.__x_current_basic = self.__exploratory_search(*self.__basic_points[-1], precision)
                if not sucs:
                    return
                
            end = check_criteria()

    # Returns bool value that indicates whether the new point has been found (returns True) 
    # or algorithm coudn't find any new point while delta_x_end_criteria is satisfied after dx reductions (returns False)
    def __exploratory_search(self, init_x: Point, f_x: float, precision: float) -> tuple[bool, tuple[Point, float]]:
        x = deepcopy(init_x)

        def search() -> tuple[bool, float]:
            search_successful = False
            for i in range(len(self.__dx)):
                x[i] += self.__dx[i]
                f_next_x = self.__no_point_func(x)
                if f_next_x <= f_x:
                    search_successful = True
                    continue
                
                x[i] = init_x[i] - self.__dx[i]
                f_next_x = self.__no_point_func(x)
                if f_next_x <= f_x:
                    search_successful = True
                    continue
                
                x[i] = init_x[i]
                
            return (search_successful, f_next_x if search_successful else f_x)
        
        success, f_next_x = search()
        while(success != True):
            if self.compute_delta_x_end_criteria() <= precision:
                break
            
            self.__dx = np.round(self.__dx / self.__alpha, self.__after_comma)
            success, f_next_x = search()

        return (success, (x, f_next_x))

    @property
    def basic_points(self) -> list[tuple[Point, float]]:
        return self.__basic_points
    
    @property
    def dx(self) -> Point:
        return self.__dx
    
    @property
    def x_current_basic(self) -> tuple[Point, float]:
        return self.__x_current_basic
    
    def __euclidian_metric(self, x: Point) -> float:
        return np.sqrt(np.sum(x ** 2))

    def compute_delta_x_end_criteria(self) -> float:
        return np.round(self.__euclidian_metric(self.__dx), self.__after_comma)

    def compute_relative_end_criteria(self) -> tuple[float, float]:        
        x_last, f_x_last = self.__x_current_basic
        x_prelast, f_x_prelast = self.__basic_points[-1]

        criteria1 = self.__euclidian_metric(x_last - x_prelast) / self.__euclidian_metric(x_prelast)
        criteria2 = np.abs(f_x_last - f_x_prelast) / np.abs(f_x_prelast)

        return (np.round(criteria1, self.__after_comma),
                np.round(criteria2, self.__after_comma)
                )
