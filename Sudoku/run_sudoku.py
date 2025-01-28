import sys
from enum import Enum
from solve_and_create_sudoku import *

class Mode(Enum):
    SOLVE = 1
    UNIQUE = 2
    CREATE = 3
    CREATEMIN = 4

OPTIONS = {}
OPTIONS["-s"] = Mode.SOLVE
OPTIONS["-u"] = Mode.UNIQUE
OPTIONS["-c"] = Mode.CREATE
OPTIONS["-cm"] = Mode.CREATEMIN

if not sys.argv[1] in OPTIONS:
    sys.stdout.write("python sudokub.py <operation> <argument>\n")
    sys.stdout.write(" where <operation> can be -s, -u, -c, -cm\n")
    sys.stdout.write("python sudokub.py -s <input>.txt: solves the Sudoku in input, whatever its size\n")
    sys.stdout.write("python sudokub.py -u <input>.txt: check the uniqueness of solution for Sudoku in input, whatever its size\n")
    sys.stdout.write("python sudokub.py -c <size>: creates a Sudoku of appropriate <size>\n")
    sys.stdout.write("python sudokub.py -cm <size> <number to exclude>: creates a Sudoku of appropriate <size> using only <size>-1 numbers\n")
    sys.stdout.write(" <size> is either 4, 9, 16, or 25\n")
    exit("Bad arguments !\n")

mode = OPTIONS[sys.argv[1]]
if mode == Mode.SOLVE or mode == Mode.UNIQUE:
    if(len(sys.argv)) != 3:
        exit("Only the size must be encoded !")

    filename = str(sys.argv[2])
    sudoku = sudoku_read(filename)
    N = len(sudoku)

    myfile = open("sudoku.cnf", 'w')
    myfile.write("p cnf "+str(N)+str(N)+str(N)+" "+
                 str(sudoku_constraints_number(sudoku))+"\n")

    sudoku_generic_constraints(myfile, N)
    sudoku_specific_constraints(myfile, sudoku)

    myfile.close()

    sys.stdout.write("sudoku\n")
    sudoku_print(sys.stdout, sudoku)

    sudoku = sudoku_solve("sudoku.cnf") 

    sys.stdout.write("\nsolution\n")
    sudoku_print(sys.stdout, sudoku)

    if sudoku != [] and mode == Mode.UNIQUE:
        myfile = open("sudoku.cnf", 'a')
        sudoku_other_solution_constraint(myfile, sudoku)
        myfile.close()

        sudoku = sudoku_solve("sudoku.cnf")

        if sudoku == []:
            sys.stdout.write("\nsolution is unique\n")
        else:
            sys.stdout.write("\nother solution\n")
            sudoku_print(sys.stdout, sudoku)

elif mode == Mode.CREATE or mode == Mode.CREATEMIN:
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        exit("Invalid number of arguments !")

    size = int(sys.argv[2])
    excluded = None

    if mode == Mode.CREATEMIN:
        if len(sys.argv) != 4:
            exit("Please also encode the number to exclude !")
        excluded = int(sys.argv[3])

        if excluded < 1 or excluded > size:
            exit("Number to exclude must be strictly positive and smaller or equal to the size !")

    sudoku = sudoku_generate(size, excluded)
    
    # Save the generated Sudoku to a file
    sudoku_save_to_file("Generation/generated_sudoku.txt", sudoku)

    myfile = open("sudoku.cnf", 'w')
    myfile.write("p cnf "+str(size)+str(size)+str(size)+" "+
                 str(sudoku_constraints_number(sudoku))+"\n")

    sudoku_generic_constraints(myfile, size)
    sudoku_specific_constraints(myfile, sudoku)

    myfile.close()

    sys.stdout.write("generated sudoku\n")
    sudoku_print(sys.stdout, sudoku)

    sudoku = sudoku_solve("sudoku.cnf")

    sys.stdout.write("\nsolution of the generated sudoku\n")
    sudoku_print(sys.stdout, sudoku)

    if sudoku != []:
        sudoku_save_to_file("Generation/solution_generated_sudoku.txt", sudoku)

        myfile = open("sudoku.cnf", 'a')
        sudoku_other_solution_constraint(myfile, sudoku)
        myfile.close()

        sudoku = sudoku_solve("sudoku.cnf")

        if sudoku == []:
            sys.stdout.write("\nsolution is unique\n")
        else:
            sys.stdout.write("\nother solution\n")
            sudoku_print(sys.stdout, sudoku)
