import nbformat
from types import ModuleType


def load_notebook_functions(path: str) -> ModuleType:
    """Load a Jupyter notebook and execute its code cells into a fresh module.

    Why ModuleType?
    - It creates a clean namespace so we can access student's functions as attributes,
      e.g. `hw.euclidean_distance`.

    Notes:
    - The module name (here: 'lab') is arbitrary; it does not affect grading.
    - Cells that error are skipped so that plots/extra cells don't break grading.
    """

    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    mod = ModuleType("lab")

    for cell in nb.cells:
        if cell.cell_type == 'code':
            try:
                exec(cell.source, mod.__dict__)
            except Exception:
                continue

    return mod