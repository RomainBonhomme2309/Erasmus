from pycsp3 import *

# Example 9x9 Sudoku grid with 0 representing empty cells
# You can replace this grid with any other Sudoku puzzle
sudoku_grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

# Decision variables: a 9x9 grid of variables, each ranging from 1 to 9
grid = VarArray(size=[9, 9], dom=range(1, 10))

# Constraints
satisfy(
    # Pre-filled cells constraints
    [grid[i][j] == sudoku_grid[i][j] for i in range(9) for j in range(9) if sudoku_grid[i][j] != 0],

    # Rows must contain all numbers from 1 to 9 exactly once
    [AllDifferent(row) for row in grid],

    # Columns must contain all numbers from 1 to 9 exactly once
    [AllDifferent(col) for col in columns(grid)],

    # Each 3x3 subgrid must contain all numbers from 1 to 9 exactly once
    [AllDifferent([grid[i + k][j + l] for k in range(3) for l in range(3)]) for i in range(0, 9, 3) for j in range(0, 9, 3)]
)

# Solve the Sudoku puzzle
if solve() is SAT:
    print("Sudoku solution:")
    for i in range(9):
        print([value(grid[i][j]) for j in range(9)])
else:
    print("No solution found.")
