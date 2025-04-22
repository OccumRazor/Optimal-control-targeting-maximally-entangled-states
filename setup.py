from setuptools import setup, find_packages

setup(
    name="QSPOC",
    version="0.1",
    description="Contains cheby propagator",
    author="Yitian Wang",
    author_email="wangyitian19@mails.ucas.edu.cn",
    url="https://github.com/OccumRazor/Optimal-control-targeting-maximally-entangled-states",
    packages=["src"],
    install_requires=["qutip", "matploblib", "numpy","qdyn","scipy","krotov"],
)
