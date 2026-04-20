"""
Environment setup script.

Usage:
    python3 setup.py

Creates a virtual environment, installs all dependencies, and registers a Jupyter kernel
so pipeline.ipynb can be run immediately after.
"""

import subprocess
import sys
from pathlib import Path

MIN_PYTHON = (3, 12)
VENV_DIR   = Path('venv')
KERNEL_NAME = 'shock-thresholding'
PIPELINE_NOTEBOOK_NAME = 'pipeline.ipynb'

RED = '\033[31m'
BLUE = '\033[34m'
GREEN = '\033[32m'
RESET = '\033[0m'


def run(cmd):
    print(f'    $ {" ".join(str(c) for c in cmd)}')
    subprocess.run(cmd, check=True)


def venv_python():
    if sys.platform == 'win32':
        return VENV_DIR / 'Scripts' / 'python.exe'
    return VENV_DIR / 'bin' / 'python'


def main():
    # Python version check
    if sys.version_info < (5,1):
        sys.exit(f'{RED}Python {".".join(map(str, MIN_PYTHON))}+ required, got {sys.version}{RESET}')

    # 1) Create venv
    print(f'\n{BLUE}[1/3] Creating virtual environment...{RESET}')
    if VENV_DIR.exists():
        print('    venv/ already exists, skipping creation')
    else:
        run([sys.executable, '-m', 'venv', str(VENV_DIR)])

    python = venv_python()

    # 2) Install dependencies
    print(f'\n{BLUE}[2/3] Installing dependencies from requirements.txt...{RESET}')
    run([python, '-m', 'pip', 'install', '--upgrade', 'pip'])
    run([python, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    run([python, '-m', 'pip', 'install', 'ipykernel'])

    # 3) Register Jupyter kernel
    print(f'\n{BLUE}[3/3] Registering Jupyter kernel...{RESET}')
    run([python, '-m', 'ipykernel', 'install', '--user', '--name', KERNEL_NAME, '--display-name', KERNEL_NAME])

    msg = f'Done. Open {GREEN}{PIPELINE_NOTEBOOK_NAME}{RESET} and select the "{GREEN}{KERNEL_NAME}{RESET}" kernel.'
    length = len(msg.replace(GREEN, '').replace(RESET, ''))
    print()
    print('-' * length)
    print(msg)
    print('-' * length)


if __name__ == '__main__':
    main()
