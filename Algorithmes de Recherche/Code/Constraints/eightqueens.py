import random

class EightQueens:
    def __init__(self, size=8):
        self.size = size
        # Start with one queen in each row, placed randomly in any column
        self.board = [random.randint(0, size - 1) for _ in range(size)]
    
    def calculate_conflicts(self, board):
        """Calculate the total number of conflicts on the board."""
        conflicts = 0
        for row1 in range(self.size):
            for row2 in range(row1 + 1, self.size):
                if board[row1] == board[row2]:  # Same column conflict
                    conflicts += 1
                elif abs(board[row1] - board[row2]) == abs(row1 - row2):  # Diagonal conflict
                    conflicts += 1
        return conflicts
    
    def find_min_conflict_position(self, row):
        """Find the column in the given row with the fewest conflicts."""
        min_conflicts = float('inf')
        best_column = self.board[row]
        
        for col in range(self.size):
            if col == self.board[row]:
                continue  # Skip current position
            
            # Temporarily place the queen at the new column
            original_column = self.board[row]
            self.board[row] = col
            conflicts = self.calculate_conflicts(self.board)
            self.board[row] = original_column  # Restore the original column
            
            if conflicts < min_conflicts:
                min_conflicts = conflicts
                best_column = col
        
        return best_column
    
    def solve(self, max_iterations=1000):
        """Try to solve the problem by local search (hill climbing)."""
        for iteration in range(max_iterations):
            conflicts = self.calculate_conflicts(self.board)
            if conflicts == 0:
                return self.board  # Solution found
            
            # Choose a random row to adjust its queen's position
            row = random.randint(0, self.size - 1)
            # Move the queen in this row to a column with the fewest conflicts
            self.board[row] = self.find_min_conflict_position(row)
        
        # If max iterations reached, return None indicating failure to find solution
        return None if self.calculate_conflicts(self.board) > 0 else self.board

solver = EightQueens()
solution = solver.solve()

if solution:
    print("Solution found:")
    for row in range(solver.size):
        print(" ".join("Q" if col == solution[row] else "." for col in range(solver.size)))
else:
    print("No solution found within the given iterations.")
