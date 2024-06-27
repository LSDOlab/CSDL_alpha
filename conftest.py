def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default="inline", help="Decides which backend to use for testing. Default is 'inline', other options are 'jax_sim' and 'py_sim'")
    parser.addoption("--build_inline", action="store_false", help="If on, sets inline=True in Recorder argument'")
    parser.addoption("--batched_derivs", action="store_true", help="If on, computes derivatives in batched mode'")

collect_ignore_glob = ["*__init__.py"]

from pathlib import Path
from pytest import Module

package_loc = Path(__file__).parent

additional_modules = list((package_loc / "csdl_alpha" / "src" / "operations").glob("*.py"))
additional_modules += list((package_loc / "csdl_alpha" / "src" / "operations" / "set_get").glob("*.py"))
additional_modules += list((package_loc / "csdl_alpha" / "src" / "operations" / "linalg").glob("*.py"))
additional_modules += list((package_loc / "csdl_alpha" / "src" / "operations" / "tensor").glob("*.py"))
additional_modules += list((package_loc / "csdl_alpha" / "src" / "operations" / "sparse").glob("*.py"))
additional_modules += list((package_loc / "csdl_alpha" / "src" / "operations" / "derivatives").glob("*.py"))
additional_modules += list((package_loc / "csdl_alpha" / "src" / "operations" / "special").glob("*.py"))

def pytest_collect_file(file_path, path, parent):
    if file_path in additional_modules:
        return Module.from_parent(path=file_path, parent=parent)
    else:
        return None