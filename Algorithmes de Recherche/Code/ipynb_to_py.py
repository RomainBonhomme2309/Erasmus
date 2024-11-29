import json

# Load the .ipynb file
with open('Planning.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Extract code cells
code_cells = [
    cell['source'] for cell in notebook['cells']
    if cell['cell_type'] == 'code'
]

# Write to .py file
with open('Planning.py', 'w', encoding='utf-8') as f:
    for cell in code_cells:
        f.write('\n'.join(cell) + '\n\n')
