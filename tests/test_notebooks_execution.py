"""
Test that example Jupyter notebooks execute without errors.
"""
import json
import matplotlib
from pathlib import Path
import pytest


# Use non-interactive backend for plotting
matplotlib.use('Agg')


# Directory containing example notebooks
EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'


@pytest.mark.parametrize('nb_path', EXAMPLES_DIR.glob('*.ipynb'))
def test_notebook_execution(nb_path):
    """
    Execute all code cells in the notebook and ensure no exceptions occur.
    """
    # Load notebook JSON
    nb_json = json.loads(nb_path.read_text(encoding='utf-8'))
    # Extract code cells
    cells = [cell for cell in nb_json.get('cells', []) if cell.get('cell_type') == 'code']
    # Build a combined script
    script_lines = []
    # Prepend backend setting in case notebooks import matplotlib.pyplot
    script_lines.append('import matplotlib\n')
    script_lines.append('matplotlib.use(\'Agg\')\n')
    for cell in cells:
        for line in cell.get('source', []):
            # Skip IPython magic commands
            if line.lstrip().startswith('%'):
                continue
            # Ensure each source line ends with a newline
            script_lines.append(line.rstrip('\n') + '\n')
    script = ''.join(script_lines)
    # Execute in isolated namespace
    ns = {}
    exec(script, ns, ns)
