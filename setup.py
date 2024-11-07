from setuptools import setup, find_packages

setup(
    name='Prometheus',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pandas',
        'matplotlib',
        'tqdm',
        'pyarrow'
    ],
    entry_points={
        'console_scripts': [
            'train_model=Train.train_somoformer:main',
            # 'squeeze_data=DataCollection.squezzing_data:main'
        ],
    },
    author='Noah Cylich',
    author_email='noahcylich@gmail.com',
    description='Prometheus Quant Research',
    long_description='proprietary',
    long_description_content_type='text/markdown',
    url='https://github.com/ncylich/Prometheus',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)