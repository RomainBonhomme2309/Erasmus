import subprocess
from random import randint

# Reads a sudoku from file
# Columns are separated by |, lines by newlines
# Example of a 4x4 sudoku:
# |1| | | |
# | | | |3|
# | | |2| |
# | |2| | |
# Spaces and empty lines are ignored

# Read Sudoku
def sudoku_read(filename):
    myfile = open(filename, 'r')
    sudoku = []
    N = 0
    for line in myfile:
        line = line.replace(" ", "")
        if line == "":
            continue
        line = line.split("|")
        if line[0] != '':
            exit("Illegal Input: every line should start with | !\n")
        line = line[1:]
        if line.pop() != '\n':
            exit("Illegal Input !\n")
        if N == 0:
            N = len(line)
            if N != 4 and N != 9 and N != 16 and N != 25:
                exit("Illegal Input: only size 4, 9, 16 and 25 are supported !\n")
        elif N != len(line):
            exit("Illegal Input: number of columns is not invariant !\n")
        line = [int(x) if x != '' and int(x) >= 0 and int(x) <= N else 0 for x in line]
        sudoku += [line]
    return sudoku

# Print Sudoku
def sudoku_print(myfile, sudoku):
    if sudoku == []:
        myfile.write("Sudoku is empty !\n")
    N = len(sudoku)
    for line in sudoku:
        myfile.write("|")
        for number in line:
            if N > 9 and number < 10:
                myfile.write(" ")
            myfile.write(" " if number == 0 else str(number))
            myfile.write("|")
        myfile.write("\n")

# Save the sudoku in a new file
def sudoku_save_to_file(filename, sudoku):
    with open(filename, 'w') as file:
        sudoku_print(file, sudoku)

# Get the number of constraints for the sudoku
def sudoku_constraints_number(sudoku):
    N = len(sudoku)

    # Generate the number of constraints based on the number of clauses (binary and N-ary)
    count = 4 * N ** 2 * ((N ** 2 - N) / 2) + 4 * N ** 2
    
    # Each filled cell represents a new constraints
    for l in sudoku:
        for c in l:
            if c > 0:
                count += 1
    return count

# Print the generic constraints for a sudoku of size N
def sudoku_generic_constraints(myfile, N):
    def output(s):
        myfile.write(s)

    def newlit(i, j, k):
        if N == 4 or N == 9:
            output(str(i) + str(j) + str(k) + " ")
        else:
            output((("0" + str(i)) if i >= 1 and i < 10 else str(i)) +
                   (("0" + str(j)) if j >= 1 and j < 10 else str(j)) +
                   (("0" + str(k)) if k >= 1 and k < 10 else str(k)) + " ")

    def newcl():
        output("0\n")

    def newcomment(s):
        output("")

    if N == 4:
        n = 2
    elif N == 9:
        n = 3
    elif N == 16:
        n = 4
    elif N == 25:
        n = 5
    else:
        exit("Only supports size 4, 9, 16 and 25 !")
    
    # At least one value per cell, line and column, respectively
    for i in range(N):
        for j in range(N):
            for k in range(N):
                newlit(i + 1, j + 1, k + 1)
                newlit(i + 1, k + 1, j + 1)
                newlit(k + 1, i + 1, j + 1)
            newcl()
        
    # At most one value per cell, line and column, respectively
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(k + 1, N):
                    newlit(- (i + 1), j + 1, k + 1)
                    newlit(- (i + 1), j + 1, l + 1)
                    newcl()

                    newlit(- (i + 1), k + 1, j + 1)
                    newlit(- (i + 1), l + 1, j + 1)
                    newcl()

                    newlit(- (k + 1), i + 1, j + 1)
                    newlit(- (l + 1), i + 1, j + 1)
                    newcl()
   
    # Minimum one time each number per square (n * n)
    for i in range(0, N, n):
        for j in range(0, N, n):
            for k in range(1, N + 1):
                for l in range(i, n + i):
                    for m in range(j, n + j):
                        newlit(l + 1, m + 1, k)
                newcl()   
    
    # Maximum one time each number per square (n * n)
    for i in range(0, N, n):
        for j in range(0, N, n):
            for k in range(1, N + 1):
                for l in range(i, n + i):
                    for m in range(j, n + j):
                        for o in range(i, n + i):
                            for p in range(j, n + j):
                                if l != o or m != p:
                                    newlit(-(l + 1), m + 1, k)
                                    newlit(-(o + 1), p + 1, k)
                                    newcl()
                                    

def sudoku_specific_constraints(myfile, sudoku):
    N = len(sudoku)

    def output(s):
        myfile.write(s)

    def newlit(i, j, k):
        if N == 4 or N == 9:
            output(str(i) + str(j) + str(k) + " ")
        else:
            output((("0" + str(i)) if i >= 1 and i < 10 else str(i)) +
                   (("0" + str(j)) if j >= 1 and j < 10 else str(j)) +
                   (("0" + str(k)) if k >= 1 and k < 10 else str(k)) + " ")

    def newcl():
        output("0\n")

    for i in range(N):
        for j in range(N):
            if sudoku[i][j] > 0:
                newlit(i + 1, j + 1, sudoku[i][j])
                newcl()

def sudoku_other_solution_constraint(myfile, sudoku):
    N = len(sudoku)

    def output(s):
        myfile.write(s)

    def newlit(i, j, k):
        if N == 4 or N == 9:
            output(str(i) + str(j) + str(k) + " ")
        else:
            output((("0" + str(i)) if i >= 1 and i < 10 else str(i)) +
                   (("0" + str(j)) if j >= 1 and j < 10 else str(j)) +
                   (("0" + str(k)) if k >= 1 and k < 10 else str(k)) + " ")

    def newcl():
        output("0\n")

    # Add the previous solution to our constraint
    for i in range(N):
        for j in range(N):
            newlit(- (i + 1), j + 1, sudoku[i][j])
    newcl()
                
def sudoku_solve(filename):
    command = "java -jar org.sat4j.core.jar sudoku.cnf"
    process = subprocess.Popen(command, shell=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate()

    for line in out.split(b'\n'):
        line = line.decode("utf-8")

        if line == "" or line[0] == 'c':
            continue

        if line[0] == 's':
            if line != 's SATISFIABLE':
                return []

            continue

        if line[0] == 'v':
            line = line[2:]
            units = line.split()

            if units.pop() != '0':
                exit("Strange output from SAT solver 1:" + line + "\n")

            units = [int(x) for x in units if int(x) >= 0]
            N = len(units)

            if N == 16:
                N = 4
            elif N == 81:
                N = 9
            elif N == 256:
                N = 16
            elif N == 625:
                N = 25
            else:
                exit("Strange output from SAT solver 2:" + line + "\n")

            sudoku = [[0 for i in range(N)] for j in range(N)]

            for number in units:
                if N == 4 or N == 9:
                    sudoku[number // 100 - 1][(number // 10 )% 10 - 1] = number % 10
                else: # Because lines, columns and cells can be 2-digit numbers and not only 1-digit 
                    sudoku[number // 10000 - 1][(number // 100) % 100 - 1] = number % 100
                                
            return sudoku

        exit("Strange output from SAT solver 3:" + line + "\n")
        return []

def sudoku_generate(size, excluded):
    if size == 4:
        n = 2
    elif size == 9:
        n = 3
    elif size == 16:
        n = 4
    elif size == 25:
        n = 5
    else:
        exit("Only supports to create sudoku of size 4, 9, 16 and 25 !")
    
    # Create an empty sudoku
    sudoku = [[0] * size for _ in range(size)]
    
    def not_in(sudoku, i, value, type):
        for j in range(size):
            if ((type == "row") and sudoku[i][j] == value):
                return False
            elif ((type == "column") and sudoku[j][i] == value):
                return False

        return True
    
    def not_in_square(sudoku, i, j, value):
        i = (i // n) * n
        j = (j // n) * n
        
        for k in range(i, n + i):
            for l in range(j, n + j):
                if sudoku[k][l] == value:
                    return False

        return True

    # Fill the sudoku
    while True:
        value = randint(1, size)
        
        if (excluded is not None) and (excluded == value):
            continue
        
        i = randint(0, size - 1)
        j = randint(0, size - 1)

        if ((not sudoku[i][j]) and not_in(sudoku, i, value, "row") and
            not_in(sudoku, j, value, "column") and not_in_square(sudoku, i, j, value)):
            sudoku[i][j] = value

            mysudoku = open("sudoku.cnf", 'w')

            mysudoku.write("p cnf "+ str(size) +str(size) +str(size) +
                           " " + str(sudoku_constraints_number(sudoku))+"\n")

            sudoku_generic_constraints(mysudoku, size)
            sudoku_specific_constraints(mysudoku, sudoku)

            mysudoku.close()
            
            solution = sudoku_solve("sudoku.cnf")

            if solution == []: # Adding this value made the sudoku unfeasible
                sudoku[i][j] = 0
            else:
                mysudoku = open("sudoku.cnf", 'a')
                sudoku_other_solution_constraint(mysudoku, solution)
                mysudoku.close()

                solution = sudoku_solve("sudoku.cnf")

                if solution == [] : # Solution found is unique
                    return sudoku
                else: # There is another solution
                    continue
        else:
            continue

    return []
