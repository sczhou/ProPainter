# -*- coding: utf-8 -*-
"""Entry point kept for the upstream command line (`python inference_propainter.py ...`).

The implementation lives in `propainter/inference.py`; installed as a package it is also
available as `propainter-inference` or `python -m propainter.inference`.
"""
from propainter.inference import main

if __name__ == '__main__':
    main()
