from setuptools import setup

with open("requirements.txt") as f :
    requirements = f.read().splitlines()

setup(
    name="Tengai",
    version="0.0.1",
    description="Just Ordinary Pakages",
    author="Iqbal Ramadhan Anniswa",
    packages=["Tengai","Tengai_demo"],
    python_requires=">=3.7",
    install_requires=requirements
)