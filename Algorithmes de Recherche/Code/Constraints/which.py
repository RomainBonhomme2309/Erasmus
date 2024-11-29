from pycsp3 import *

# Decision variables
x = Var(range(400))
y = Var(range(400))

# Constraints
satisfy(
    3 * x + 2 * y <= 800,
    x + 3 * y <= 700,
    x + 2 * y <= 400
)

# Objective function to maximize
maximize(
    4 * x + 5 * y
)

# Solve the problem
if solve() == OPTIMUM:
    print("Solution found:")
    print(f"x = {value(x)}")
    print(f"y = {value(y)}")
    print(f"Maximum value = {4 * value(x) + 5 * value(y)}")
else:
    print("No optimal solution found.")
