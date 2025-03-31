from setuptools import find_packages, setup

setup(
    name="xgboost_covid",
    version="0.1.0",
    description="Genomic analysis of COVID-19 using XGBoost and other ML models",
    author="Felipe dos Santos Passarela",
    author_email="felipepassarela11@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "matplotlib",
        "shap==0.46.0",
        "imbalanced-learn",
        "umap-learn",
        "pyyaml",
        "scipy",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ]
)