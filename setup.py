from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).parent

long_description = (root / "README.md").read_text(encoding="utf-8")
install_requires = [
    line.strip()
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("--")
]

setup(
    name='dl_backtrace',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    description='A python SDK for Deep Learning Backtrace',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords='lexsi labs deep learning backtrace, ML observability',
    license='Lexsi Labs Source Available License (LSAL) v1.0',
    url='https://xai.arya.ai/docs/introduction',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(where='.'),
    python_requires='>=3.8',
    install_requires=install_requires,
    package_data={'': ['*.md', '*.txt']},
)
