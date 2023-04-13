import os
import shutil
from module_lab2_task1 import *

os.makedirs('solutions')  # make a folder for solutions
Solver = LUSolver()  # make new object

i = 1

while i < 101:
    problem_filename = 'problem' + str(i) + '.txt'  # iterates the number of the problem we want to solve
    Solver.read_system_from_file(problem_filename)  # read file and apply our LU factorisation methods
    Solver.lu_factors()
    Solver.forward_sub()
    Solver.backward_sub()
    solution_filename = 'solution' + str(i) + '.txt'  # iterates the number of the solution we want to write to
    Solver.write_solution_to_file(solution_filename)  # this will create the file in the 'ENGSCI-233-Lab-2' folder

    dst = 'solutions' + os.sep + solution_filename  # moves the file to the 'solutions' folder
    shutil.move(solution_filename, dst)

    i += 1



