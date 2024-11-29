import random
import numpy as np

def generate_random_assignment(variables):
    return {var: random.choice([True, False]) for var in variables}

def evaluate_formula(assignment, formula):
    return all(any(assignment.get(abs(lit), False) if lit > 0 else not assignment.get(abs(lit), False) for lit in clause) for clause in formula)

def count_satisfied_clauses(assignment, formula):
    return sum(1 for clause in formula if any(assignment.get(abs(lit), False) if lit > 0 else not assignment.get(abs(lit), False) for lit in clause))

def select_unsatisfied_clause(formula, assignment):
    unsatisfied_clauses = [clause for clause in formula if not any(assignment.get(abs(lit), False) if lit > 0 else not assignment.get(abs(lit), False) for lit in clause)]
    if unsatisfied_clauses:
        return random.choice(unsatisfied_clauses)
    return None

def select_variable_to_flip(assignment, formula, clause):
    best_var = None
    best_increase = -1
    current_satisfied = count_satisfied_clauses(assignment, formula)
    
    for lit in clause:
        var = abs(lit)

        assignment[var] = not assignment[var]
        flipped_satisfied = count_satisfied_clauses(assignment, formula)
        
        increase = flipped_satisfied - current_satisfied
        if increase > best_increase:
            best_increase = increase
            best_var = var

        assignment[var] = not assignment[var]

    return best_var

def walksat(formula, max_flips, p, max_tries):
    variables = set(abs(lit) for clause in formula for lit in clause)
    
    for _ in range(max_tries):
        assignment = generate_random_assignment(variables)
        
        for i in range(max_flips):
            if evaluate_formula(assignment, formula):
                return (i, True)

            unsatisfied_clause = select_unsatisfied_clause(formula, assignment)
            if unsatisfied_clause is None:
                continue

            elif random.random() < p:
                var_to_flip = abs(random.choice(unsatisfied_clause))
            else:
                var_to_flip = select_variable_to_flip(assignment, formula, unsatisfied_clause)
            
            if var_to_flip is not None:
                assignment[var_to_flip] = not assignment[var_to_flip]
        
    return (0, False)

def random_formula(num_vars, num_clauses, clause_length=3):
    return [[random.choice([-1, 1]) * random.randint(1, num_vars) for _ in range(clause_length)] for _ in range(num_clauses)]

nb_clauses = [50, 80, 100, 130, 150, 180, 200, 230, 250]
max_flips = 1000
max_tries = 10
walk_probability = 0.5

total_result = np.zeros(len(nb_clauses))
total_flips = np.zeros(len(nb_clauses))

for i in range(len(nb_clauses)):
    result = 0
    nb_flips = 0
    for _ in range(10):
        formula = random_formula(50, nb_clauses[i])
        flips, assignment = walksat(formula, max_flips, walk_probability, max_tries)
        if assignment: 
            result += 1
            nb_flips += flips

    total_result[i] = result

    if result == 1:
        total_flips[i] = nb_flips
        print(f"For a ratio of {nb_clauses[i] / 50}, {result} formula was solved, with {nb_flips} flips.")
    elif result != 0:
        total_flips[i] = (nb_flips / result)
        print(f"For a ratio of {nb_clauses[i] / 50}, {result} formula were solved, with an average of {total_flips[i]:.0f} flips.")
    else:
        print(f"For a ratio of {nb_clauses[i] / 50}, no formula were solved.")

print(total_result)
print(total_flips)

""" p = 0.0
For a ratio of 1.0, 10 formula were solved, with an average of 4 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 7 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 9 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 11 flips.
For a ratio of 3.0, 9 formula were solved, with an average of 13 flips.
For a ratio of 3.6, 8 formula were solved, with an average of 22 flips.
For a ratio of 4.0, 3 formula were solved, with an average of 15 flips.
For a ratio of 4.6, no formula were solved.
For a ratio of 5.0, no formula were solved.
[10. 10. 10. 10.  9.  8.  3.  0.  0.]
[ 4.5         6.9         8.6        10.9        13.22222222 21.625
 15.33333333  0.          0.        ]
"""

""" p = 0.1
For a ratio of 1.0, 10 formula were solved, with an average of 5 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 12 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 23 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 104 flips.
For a ratio of 3.0, 10 formula were solved, with an average of 164 flips.
For a ratio of 3.6, 8 formula were solved, with an average of 233 flips.
For a ratio of 4.0, 5 formula were solved, with an average of 297 flips.
For a ratio of 4.6, 2 formula were solved, with an average of 470 flips.
For a ratio of 5.0, no formula were solved.
[10. 10. 10. 10. 10.  8.  5.  2.  0.]
[  4.8    12.5    23.    104.2   164.2   233.125 297.2   470.5     0.   ]
"""

""" p = 0.2
For a ratio of 1.0, 10 formula were solved, with an average of 6 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 13 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 29 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 19 flips.
For a ratio of 3.0, 10 formula were solved, with an average of 243 flips.
For a ratio of 3.6, 10 formula were solved, with an average of 289 flips.
For a ratio of 4.0, 8 formula were solved, with an average of 225 flips.
For a ratio of 4.6, 1 formula was solved, with 986 flips.
For a ratio of 5.0, 1 formula was solved, with 178 flips.
[10. 10. 10. 10. 10. 10.  8.  1.  1.]
[  6.4    13.1    29.1    19.    243.1   289.4   224.625 986.    178.   ]
"""

""" p = 0.3
For a ratio of 1.0, 10 formula were solved, with an average of 5 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 10 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 13 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 68 flips.
For a ratio of 3.0, 10 formula were solved, with an average of 196 flips.
For a ratio of 3.6, 10 formula were solved, with an average of 235 flips.
For a ratio of 4.0, 5 formula were solved, with an average of 229 flips.
For a ratio of 4.6, 1 formula was solved, with 619 flips.
For a ratio of 5.0, no formula were solved.
[10. 10. 10. 10. 10. 10.  5.  1.  0.]
[  5.3   9.6  12.6  67.5 195.9 235.3 229.  619.    0. ]
"""

""" p  = 0.4
For a ratio of 1.0, 10 formula were solved, with an average of 6 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 10 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 18 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 84 flips.
For a ratio of 3.0, 10 formula were solved, with an average of 48 flips.
For a ratio of 3.6, 9 formula were solved, with an average of 329 flips.
For a ratio of 4.0, 8 formula were solved, with an average of 451 flips.
For a ratio of 4.6, 2 formula were solved, with an average of 280 flips.
For a ratio of 5.0, 1 formula was solved, with 255 flips.
[10. 10. 10. 10. 10.  9.  8.  2.  1.]
[  6.3         10.1         18.2         83.5         47.7
 329.44444444 450.625      280.5        255.        ]
"""

""" p = 0.5
For a ratio of 1.0, 10 formula were solved, with an average of 7 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 15 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 21 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 37 flips.
For a ratio of 3.0, 10 formula were solved, with an average of 60 flips.
For a ratio of 3.6, 10 formula were solved, with an average of 213 flips.
For a ratio of 4.0, 9 formula were solved, with an average of 236 flips.
For a ratio of 4.6, 4 formula were solved, with an average of 284 flips.
For a ratio of 5.0, no formula were solved.
[10. 10. 10. 10. 10. 10.  9.  4.  0.]
[  7.1         14.9         20.8         37.1         60.1
 213.         235.88888889 284.           0.        ]
"""

""" p = 0.6
For a ratio of 1.0, 10 formula were solved, with an average of 8 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 15 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 19 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 37 flips.
For a ratio of 3.0, 10 formula were solved, with an average of 66 flips.
For a ratio of 3.6, 10 formula were solved, with an average of 203 flips.
For a ratio of 4.0, 9 formula were solved, with an average of 500 flips.
For a ratio of 4.6, 3 formula were solved, with an average of 255 flips.
For a ratio of 5.0, no formula were solved.
[10. 10. 10. 10. 10. 10.  9.  3.  0.]
[  7.5         15.2         19.3         37.3         65.6
 203.4        500.11111111 255.33333333   0.        ]
"""

""" p = 0.7
For a ratio of 1.0, 10 formula were solved, with an average of 7 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 11 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 17 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 42 flips.
For a ratio of 3.0, 10 formula were solved, with an average of 68 flips.
For a ratio of 3.6, 9 formula were solved, with an average of 458 flips.
For a ratio of 4.0, 8 formula were solved, with an average of 304 flips.
For a ratio of 4.6, 1 formula was solved, with 639 flips.
For a ratio of 5.0, no formula were solved.
[10. 10. 10. 10. 10.  9.  8.  1.  0.]
[  7.3         10.8         17.2         42.3         68.3
 457.55555556 303.75       639.           0.        ]
"""

""" p = 0.8
For a ratio of 1.0, 10 formula were solved, with an average of 7 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 13 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 17 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 50 flips.
For a ratio of 3.0, 10 formula were solved, with an average of 89 flips.
For a ratio of 3.6, 10 formula were solved, with an average of 234 flips.
For a ratio of 4.0, 9 formula were solved, with an average of 399 flips.
For a ratio of 4.6, 2 formula were solved, with an average of 448 flips.
For a ratio of 5.0, 2 formula were solved, with an average of 324 flips.
[10. 10. 10. 10. 10. 10.  9.  2.  2.]
[  6.8         12.6         16.9         49.9         89.3
 234.3        399.22222222 447.5        324.5       ]
"""

""" p  = 0.9
For a ratio of 1.0, 10 formula were solved, with an average of 9 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 17 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 29 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 59 flips.
For a ratio of 3.0, 10 formula were solved, with an average of 183 flips.
For a ratio of 3.6, 10 formula were solved, with an average of 503 flips.
For a ratio of 4.0, 9 formula were solved, with an average of 470 flips.
For a ratio of 4.6, no formula were solved.
For a ratio of 5.0, no formula were solved.
[10. 10. 10. 10. 10. 10.  9.  0.  0.]
[  8.8         16.9         29.          58.7        183.2
 502.6        470.11111111   0.           0.        ]
"""

""" p = 1.0
For a ratio of 1.0, 10 formula were solved, with an average of 14 flips.
For a ratio of 1.6, 10 formula were solved, with an average of 25 flips.
For a ratio of 2.0, 10 formula were solved, with an average of 35 flips.
For a ratio of 2.6, 10 formula were solved, with an average of 84 flips.
For a ratio of 3.0, 10 formula were solved, with an average of 207 flips.
For a ratio of 3.6, 9 formula were solved, with an average of 321 flips.
For a ratio of 4.0, 5 formula were solved, with an average of 690 flips.
For a ratio of 4.6, no formula were solved.
For a ratio of 5.0, no formula were solved.
[10. 10. 10. 10. 10.  9.  5.  0.  0.]
[ 14.          24.8         35.2         84.1        207.3
 320.88888889 690.4          0.           0.        ]
"""
