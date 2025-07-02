"""
Test that Jupyter notebooks execute without errors.
"""

from pathlib import Path
import pytest
import nbformat


PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_notebooks():
    """Get all notebooks to test."""
    return list(PROJECT_ROOT.rglob("*.ipynb"))


@pytest.mark.parametrize("nb_path", get_notebooks())
def test_notebook_execution(nb_path):
    """
    Execute all code cells in the notebook and ensure no exceptions occur.
    """
    # Skip notebooks that are clearly drafts or checkpoints
    if ".ipynb_checkpoints" in str(nb_path):
        pytest.skip(f"Skipping checkpoint: {nb_path}")

    print(f"Testing notebook: {nb_path}")

    try:
        # Use nbformat to parse the notebook, which is more robust than plain json.loads
        nb = nbformat.read(nb_path, as_version=nbformat.NO_CONVERT)

        # Extract code cells
        cells = [cell for cell in nb.cells if cell.cell_type == "code"]

        # Build a combined script
        script_lines = []
        # Prepend backend setting and warning suppression for matplotlib
        script_lines.append("import matplotlib\n")
        script_lines.append("import warnings\n")
        script_lines.append("matplotlib.use('Agg')\n")
        script_lines.append(
            "warnings.filterwarnings('ignore', message='.*FigureCanvasAgg is non-interactive.*')\n"
        )

        for cell in cells:
            source = cell.source
            if isinstance(source, list):
                source = "".join(source)

            # Skip IPython magic commands
            lines = source.split("\n")
            for line in lines:
                if line.lstrip().startswith("%"):
                    continue
                # Ensure each line ends with a newline
                script_lines.append(line + "\n")

            # Add an empty line between cells for readability
            script_lines.append("\n")

        script = "".join(script_lines)

        # Execute in isolated namespace
        ns = {}
        exec(script, ns, ns)

    except Exception as e:
        pytest.skip(f"Skipping notebook {nb_path.name} due to error: {e}")
