import papermill as pm
import os

param_values = [1,2,20]  # or use a list of dicts for multiple parameters
output_dir = "L_sweep"

os.makedirs(output_dir, exist_ok=True)

for val in param_values:
    output_nb = os.path.join(output_dir, f"notebook_{val}.ipynb")

    # Run the notebook with new parameter
    pm.execute_notebook(
        'analyze_linear_network.ipynb',  # input notebook
        output_nb,  # output notebook with execution results
        parameters=dict(param1=val)
    )

    # Convert to HTML
    os.system(f"jupyter nbconvert --to html '{output_nb}'")

    # Optional: Convert to PDF
    # os.system(f"jupyter nbconvert --to pdf '{output_nb}'")
