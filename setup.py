from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e."

def get_requirements(file_path:str) -> List[str] :
    requirements = []

    with open(file=file_path) as file_obj:
        requirements = file_obj.readlines()

    if(HYPHEN_E_DOT in requirements) :
        requirements.remove(HYPHEN_E_DOT)

    requirements = [req.replace("\n","") for req in requirements]
    return requirements

setup(
    name="Wafer Fault Detection",
    version='0.0.1',
    author='Himanshu',
    author_email='himanshupandey1036@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)