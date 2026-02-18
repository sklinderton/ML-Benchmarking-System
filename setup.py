from setuptools import setup, find_packages

setup(
    name="mlbenchmark",
    version="1.0.0",
    description="Sistema de Benchmarking de Modelos de Machine Learning - BCD-7213 LEAD University",
    author="Melany Ramírez, Jason Barrantes, Junior Ramírez",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "imbalanced-learn>=0.11.0",
        "statsmodels>=0.14.0",
        "plotly>=5.18.0",
        "streamlit>=1.32.0",
    ],
    extras_require={
        "dl": ["tensorflow>=2.13.0"],
        "xgb": ["xgboost>=2.0.0"],
        "full": ["tensorflow>=2.13.0", "xgboost>=2.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
