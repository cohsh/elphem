from setuptools import setup, find_packages

DESCRIPTION = 'Elphem: Calculating electron-phonon interactions with the empty lattice.'
NAME = 'elphem'
AUTHOR = 'Kohei Ishii'
AUTHOR_EMAIL = ''
URL = 'https://github.com/cohsh/elphem'
LICENSE = 'MIT'
DOWNLOAD_URL = URL
VERSION = '0.1.0'
PYTHON_REQUIRES = '>=3.10'
INSTALL_REQUIRES = ['numpy', 'scipy']

CLASSIFIERS=[
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12'
]

try:
    with open('README.md', 'r', encoding='utf-8') as fp:
        LONG_DESCRIPTION = fp.read()
except FileExistsError:
    LONG_DESCRIPTION = DESCRIPTION

LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    url=URL,
    download_url=URL,
    packages=find_packages(),
    classifiers=CLASSIFIERS,
    license=LICENSE,
    install_requires=INSTALL_REQUIRES
)