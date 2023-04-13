import pytest
from module_lab2_task1 import *
import numpy as np


def test_method_1():

    test_solver = LUSolver()

    test_solver.read_system_from_file('problem1.txt')

    A = np.array([[7., 2., 1.], [28., 9., 10.], [14., 11., 45.]])

    b = np.array([[23.], [144.], [418.]])

    assert(np.all(test_solver.vector_b == b) and np.all(test_solver.matrix_a == A))


def test_method_2():
    """ regular test """

    test_solver = LUSolver()

    test_solver.matrix_a = np.array([[2, -1, 3], [-8, 3, -8], [-2, -2, 7]])

    test_solver.lu_factors()

    l = np.array([[1, 0, 0], [-4, 1, 0], [-1, 3, 1]])
    u = np.array([[2, -1, 3], [0, -1, 4], [0, 0, -2]])

    assert(np.all(test_solver.matrix_l == l) and np.all(test_solver.matrix_u == u))


def test_method_2_weirder():
    """ Test with L and b containing big numbers """

    test_solver = LUSolver()

    test_solver.matrix_a = np.array([[1, 1, 1], [3, 1, -3], [1, -2, -5]])

    test_solver.lu_factors()

    l = np.array([[1, 0, 0], [3, 1, 0], [1, 1.5, 1]])
    u = np.array([[1, 1, 1], [0, -2, -6], [0, 0, 3]])

    assert (np.all(test_solver.matrix_l == l) and np.all(test_solver.matrix_u == u))


def test_method_3():
    """ regular test """

    test_solver = LUSolver()

    test_solver.matrix_l = np.array([[1, 0, 0], [-4, 1, 0], [-1, 3, 1]])
    test_solver.vector_b = np.array([[-5], [20], [3]])

    test_solver.forward_sub()

    y = np.array([[-5.], [0.], [-2.]])

    assert(np.all(test_solver.vector_y == y))


def test_method_3_weirder():
    """ Test with L and b containing big numbers """

    test_solver = LUSolver()

    test_solver.matrix_l = np.array([[6, 2, 3], [3, 1, 1], [10, 3, 4]])
    test_solver.vector_b = np.array([[1000], [0.5], [0]])

    test_solver.forward_sub()

    y = np.array([[-1000.5], [2003], [999]])

    atol = 1.e-6

    assert np.mean(np.abs(y - test_solver.vector_y)) < atol


def test_method_4_special_case():
    """ Test for a special case matrix (identity matrix)"""

    test_solver = LUSolver()

    test_solver.matrix_u = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    test_solver.vector_y = np.array([[10], [20], [30]])

    test_solver.backward_sub()

    x = np.array([[10], [20], [30]])

    assert(np.all(test_solver.vector_x == x))


def test_method_4():
    """ Test for a regular matrix"""

    test_solver = LUSolver()

    test_solver.matrix_u = np.array([[1,1,1], [0, -2, -6], [0, 0, 3]])
    test_solver.vector_y = np.array([[1], [2], [6]])

    test_solver.backward_sub()

    x = np.array([[6], [-7], [2]])

    assert(np.all(test_solver.vector_x == x))


def test_method_5():
    test_solver = LUSolver()
    test_solver.vector_x = [1., 4., 8.]
    test_solver.write_solution_to_file('solution1.txt')
    with open('solution1.txt', 'r') as fp:
        x1 = fp.readline().strip()
        x2 = fp.readline().strip()
        x3 = fp.readline().strip()
    assert(int(x1) == 1, int(x2) == 4, int(x3) == 8)
