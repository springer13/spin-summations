# Spin Summation Code Generator #

Spin summations are essentially ``summations over tensor transpositions'',
please refer to the publication below for additional details.

# Requirements
--------------

* Intel's ICC (>= v16.0, recommended) or g++ (>= v4.8, experimental) 
* Python (tested with v2.7.5 and v2.7.9)

# Getting Started
---------

Run 

    python tensorSum.py --help

to get a list of available arguments.

If you are interested in generating the production ready code you should run:

    python tensorSum.py

This will create 21 different directories (spinSummation1, spinSummation2, ...,
spinSummation21) corresponding to the spin summations outlined in the paper (see
below).

Use the following command line, if you want to run the benchmark:

    python tensorSum.py --benchmark --numThreads=<THREADS>

# Citation
-----------
In case you want to refer this code as part of a research paper, please cite the following
article [(pdf)](https://arxiv.org/abs/1705.06661):
```
@article{springer17,
   author      = {Paul Springer, Devin Matthews  and Paolo Bientinesi},
   title       = {{Spin {S}ummations: {A} {H}igh-{P}erformance {P}erspective}},
   archivePrefix = "arXiv",
   eprint = {1705.06661},
   primaryClass = "quant-ph",
   journal     = {CoRR},
   year        = {2017},
   issue_date  = {April 2017},
   url         = {https://arxiv.org/abs/1705.06661}
}
``` 
