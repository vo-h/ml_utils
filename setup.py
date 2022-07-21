from setuptools import setup, find_packages
import pathlib
from datetime import date


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")
today = date.today().strftime(r"%Y.%m.%d")

requirements = (here / "requirements.txt").read_text(encoding="utf-8").split("\n")

setup(
    name="ml_utils",
    version=today,
    packages=find_packages(where="ml_utils"),
    install_requires=requirements,
    include_package_data=True,
)
