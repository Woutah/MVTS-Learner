"""The setup script."""
from setuptools import setup, find_packages

setup(
	name = "MVTS-Learner",
	version= "0.0.1",
	packages=find_packages('.'),
    description=("GUI"),
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author="Wouter Stokman",
    url="https://github.com/Woutah/MVTS-Learner",
    include_package_data=True,
	entry_points={
        'console_scripts': [
            'MVTS-Learner=mvts_learner.main:main',
            'mvts-learner=mvts_learner.main:main',
            'mvtsl=mvts_learner.main:main'
		],
	},
    install_requires=[ #Generated using pipreqs
        'PySide6>=6.0.0', # Qt for Python, works for 6.5.1.1
        'pathos>=0.3.0', #Works for 0.3.0
        'setuptools>=65.0.0', #Works for 65.5.0
		'dill>=0.3.0', #Works for 0.3.6
		# 'multiprocess>=0.70.00', #Works for 0.70.14
		'numpydoc>=1.4.0', #Works for 1.5.0
		'pycryptodome>=3.10.0', #Works for 3.18.0

		#===== TODO: lower version reqs ==========
        'ipdb>=0.13.11',
		'numba>=0.56.4',
		'numpydoc>=1.5.0',
		'plotly>=5.12.0',
		'scikit_learn>=1.2.0',
		'tabulate>=0.9.0',
		'torch>=2.0.1',
		'tqdm>=4.64.1',
		'wandb>=0.13.9',
		'xgboost>=1.7.3',
		'xlrd>=2.0.1',
		'xlutils>=2.0.0',
		'xlwt>=1.3.0',
        'tensorboard>=2.13.0',

		#=================Other========================
        'pyside6-utils>=1.2.2', #Works for 1.2.1
        'configurun==0.2.5',


		#================= Not used===================
        'matplotlib==3.6.2',
	]
)
