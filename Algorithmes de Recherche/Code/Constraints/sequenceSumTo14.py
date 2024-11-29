from pycsp3 import *

# Define a sequence of 4 integer variables, x1, x2, x3, x4
x = VarArray(size=4, dom=range(-100, 101))  # Domain can be adjusted as needed

# Constraint: The sum of these four numbers should equal 14
satisfy(
    Sum(x) == 14,

    # Constraint: Each variable must be successive integers
    [x[i] + 1 == x[i + 1] for i in range(3)]
)

# Solve the problem and print solutions
if solve() is SAT:
    print("Successive sequence:", values(x))
else:
    print("No solution found")
