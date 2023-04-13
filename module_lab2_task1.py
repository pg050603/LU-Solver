import numpy as np
import os


class LUSolver(object):
    """
        Solves a system of linear equations using LU factorization.

        Attributes
        __________
        matrix_a: list, None
            initialised as None, stores matrix a values in a list
        matrix_l: list, None
            initialised as None, stores matrix l values in a list
        matrix_u: list, None
            initialised as None, stores matrix u values in a list
        vector_b: list, None
            initialised as None, stores vector b value in a list
        vector_x: list, None
            initialised as None, stored vector x values in a list
        vector_y: list, None
            initialised as None, stored vector y values in a list

        Methods
        _______

        read_system_from_file
            Reads in a data file representing a system of linear equations, and updates matrix A and vector b
        lu_factors
            Calculates L and U for a given matrix A through Gaussian elimination
        forward_sub
            Calculates intermediate solution of vector y using forward substitution.
        backward_sub
            Calculates intermediate solution of vector x using backward substitutio
        write_solution_to_file
            Writes solution x to a data file
    """

    def __init__(self):
        """
            Initialises matrices and vectors to be used in LUSolver
        """
        self.matrix_a = None
        self.matrix_l = None
        self.matrix_u = None
        self.vector_b = None
        self.vector_x = None
        self.vector_y = None

    def read_system_from_file(self, str):
        """
                Reads in a data file representing a system of linear equations,
                and updates matrix A and vector b accordingly

                Arguments
                ---------
                str: str
                     represents the relative path to the data file

        """

        a = []
        b = []
        os.chdir('problems')  # change to problems directory

        fp = open(str, 'r')
        n = fp.readline().strip()  # read the header to determine n (number of rows, columns in A)
        n = int(n)  # convert string to integer
        line = fp.readline().strip()  # read the first line of data and remove the \n

        for i in range(n):  # go through the first n rows and extract data from the first n columns
            items = line.split(',')  # retrieves the comma separated values
            for j in range(n):
                a.append(int(float(items[j])))  # puts it into our temporary a-matrix storage list
            line = fp.readline().strip()  # goes to the next line

        self.matrix_a = np.array(a).reshape((n, n))  # puts the list into an n x n array with numpy

        for i in range(n):  # then go through the last n rows and extract the data, assign this to b
            items = line.split(',')
            b.append(int(float(items[0])))
            line = fp.readline().strip()

        self.vector_b = np.array(b).reshape((n, 1))

        fp.close()

    def lu_factors(self):
        """
                Calculates L and U for a given matrix A through Gaussian elimination
        """

        # Finding out the dimensions of the array a
        a = self.matrix_a
        n = len(a)

        # Initialising the U matrix
        U = a

        # Creating the initial state of the L matrix
        L = np.zeros((n, n))
        i = 0
        while i < n:
            L[i, i] = 1
            i = i + 1
        # LU factorisation
        x = 0
        while x < n:
            j = x + 1
            k = 0
            while j < n:
                z = j - (1 + k)
                while j < n:
                    b = 0
                    L[j, z] = U[j, z] / U[z, z]
                    while b < n:
                        U[j, b] = (U[j, b] - L[j, z] * U[z, b])
                        b = b + 1

                    j = j + 1
                    k = k + 1
            x = x + 1

        # assigning the arrays to their matrix
        self.matrix_l = np.array(L)
        self.matrix_u = np.array(U)

    def forward_sub(self):
        """
                Calculates intermediate solution of vector y from Ly = b
                using forward substitution. Updates value of vector_y attribute.

        """
        inv_l = np.linalg.inv(self.matrix_l)
        y = np.matmul(inv_l, self.vector_b)  # now does proper order of matrix mult.
        self.vector_y = y

    def backward_sub(self):
        """
                Calculates intermediate solution of vector x from Ux = y using backward substitution
                Updates vector_x attribute
        """
        # Calculates the inverse of matrix_u
        inv_u = np.linalg.inv(self.matrix_u)
        # Solves X as matrix multiplication between inverse_u and vector_y
        x = np.matmul(inv_u, self.vector_y)
        # Assigns calculated solutions to solution vector x
        self.vector_x = x

    def write_solution_to_file(self, str):
        """
                Writes solution x to a data file

                Arguments
                ---------
                str: str
                    represents the relative path to the data file.

        """
        os.chdir("..")  # change back to main directory
        with open(str, 'w') as fp:
            for i in self.vector_x:
                fp.write(f"{round(float(i))}\n")
