import random
import numpy as np

def generate_random_assignment(variables):
    return {var: random.choice([True, False]) for var in variables}

def evaluate_formula(assignment, formula):
    return all(any(assignment.get(abs(lit), False) if lit > 0 else not assignment.get(abs(lit), False) for lit in clause) for clause in formula)

def count_satisfied_clauses(assignment, formula):
    return sum(1 for clause in formula if any(assignment.get(abs(lit), False) if lit > 0 else not assignment.get(abs(lit), False) for lit in clause))

def select_variable_to_flip(assignment, formula):
    best_var = None
    best_increase = -1
    current_satisfied = count_satisfied_clauses(assignment, formula)

    for var in assignment:
        # Flip variable
        assignment[var] = not assignment[var]
        flipped_satisfied = count_satisfied_clauses(assignment, formula)
        
        increase = flipped_satisfied - current_satisfied
        if increase > best_increase:
            best_increase = increase
            best_var = var
        
        # Unflip variable
        assignment[var] = not assignment[var]
    
    #print(best_increase, end=" ", flush=True)
    return best_var

def gsat(formula, max_flips, max_tries):
    variables = set(abs(lit) for clause in formula for lit in clause)

    for _ in range(max_tries):
        assignment = generate_random_assignment(variables)
        
        for i in range(max_flips):
            if evaluate_formula(assignment, formula):
                return (i, True)
            
            var_to_flip = select_variable_to_flip(assignment, formula)
            if var_to_flip is not None:
                assignment[var_to_flip] = not assignment[var_to_flip]
        
    return (0, False)


def random_formula(num_vars, num_clauses, clause_length=3):
    return [[random.choice([-1, 1]) * random.randint(1, num_vars) for _ in range(clause_length)] for _ in range(num_clauses)]


"""# Example usage with a CNF formula: (A or not B) and (not A or C) and (B or not C)
formula = [[1, -2], [-1, 3], [2, -3]]
max_flips = 100
max_tries = 10
result = gsat(formula, max_flips, max_tries)
print(result)

formula = random_formula(50, 150)
max_flips = 1000
max_tries = 10
result = gsat(formula, max_flips, max_tries)
print(result)

formula = random_formula(10, 50)
max_flips = 1000
max_tries = 10
result = gsat(formula, max_flips, max_tries)
print(result)"""

nb_clauses = [50, 80, 100, 130, 150, 180, 200, 230, 250]
max_flips = 1000
max_tries = 10
total_result = np.zeros(len(nb_clauses))
total_flips = np.zeros(len(nb_clauses))

for i in range(len(nb_clauses)):
    result = 0
    nb_flips = 0
    for _ in range(10):
        formula = random_formula(50, nb_clauses[i])
        flips, assignment = gsat(formula, max_flips, max_tries)
        if assignment: 
            result += 1
            nb_flips += flips
    total_result[i] = result
    total_flips[i] = nb_flips / 10
    print(f"Done for a ratio of {nb_clauses[i] / 50}")

print(total_result)
print(total_flips)

# [10. 10. 10. 10.  5.  2.  2.  0.  0.]
# [ 4.5  6.6  8.7 10.   5.1  2.7  2.4  0.   0. ]