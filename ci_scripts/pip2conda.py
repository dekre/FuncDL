import logging
import subprocess
import sys
import os
from typing import List
from pathlib import Path
from pip._internal.network.session import PipSession
from pip._internal.req import parse_requirements
from conda.models.match_spec import MatchSpec
import conda.cli.python_api as conda_api
from typer import Typer, Option, echo

BASE_DIR = Path(__file__).parent.parent
PROJECT_DEFAULT_REQS = "3rdparty/requirements.txt"

logger = logging.getLogger("fundl")
app = Typer()

PIP2CONDA_MAP = {"opencv-python-headless": "opencv"}

PIP_ONLY = ["darwin-py"]


def __conda_install(specs=List[str], channels=List[str], base_prefix=str):
    channels_ = []
    for c in channels:
        channels_.append("-c")
        channels_.append(c)
    conda_api.run_command(
        conda_api.Commands.INSTALL, "--prefix", base_prefix, *channels_, *specs
    )


def __parse_pip_reqs(path_to_req: str) -> List[MatchSpec]:
    reqs = parse_requirements(path_to_req, session=PipSession())
    reqs = [MatchSpec(req.requirement) for req in reqs]
    return reqs


@app.command()
def parse_reqs(
    path_to_req: str = Option(None),
    channels: List[str] = ["conda-forge", "anaconda", "nodefaults"],
):
    if not path_to_req:
        path_to_req = str(BASE_DIR.joinpath(PROJECT_DEFAULT_REQS))
    reqs = __parse_pip_reqs(path_to_req)
    conda_env = {"channels": list(channels)}
    deps = list()
    pip_pckgs = list()
    for req in reqs:
        pkg_name = PIP2CONDA_MAP.get(req.name, req.name)
        pkg_version = req.version
        if pkg_name in PIP_ONLY:
            pip_pckgs.append(f"{pkg_name}=={pkg_version}")
            continue
        deps.append(f"{pkg_name}=={pkg_version}")
    conda_env["dependencies"] = deps
    echo(conda_env)
    return conda_env, pip_pckgs


@app.command()
def install(
    base_prefix: str = Option(None),
    channels: List[str] = ["conda-forge", "anaconda", "nodefaults"],
):
    conda_env, pip_pkgs = parse_reqs(channels)
    if not base_prefix:
        base_prefix = os.environ.get("CONDA_PREFIX")
    __conda_install(
        specs=conda_env["dependencies"],
        channels=conda_env["channels"],
        base_prefix=base_prefix,
    )
    if pip_pkgs:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + pip_pkgs)


if __name__ == "__main__":
    app()
