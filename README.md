# Language Identification with Character Level Ngram Language Models


## Data

The training data can be found in [data/train](data/train) and the development data can be found in [data/dev](data/dev).

## Execution
Example usage: use the following command in the current directory.
Notice: If you are using a Windows machine, change all of the '\' (backslash) symbols to '/' (slash)! 

- For running unsmoothed language model:
`python3 src/main.py data/train/ data/dev/ output/results_dev_unsmoothed.csv --unsmoothed`

- For running laplace smoothed language model:
`python3 src/main.py data/train/ data/dev/ output/results_dev_laplace.csv --laplace`

- For running interpolated language model:
`python3 src/main.py data/train/ data/dev/ output/results_dev_interpolation.csv --interpolation`

For knowing which N was selected as the best fit and what was the accuracy of the LM:

- For the unsmoothed language model:
`python3 src/main.py data/train/ data/dev/ output/ --bestUnsmoothed`

- For the laplace smoothed language model:
`python3 src/main.py data/train/ data/dev/ output/ --bestLaplace`

- For the interpolated language model:
`python3 src/main.py data/train/ data/dev/ output/ --bestInterpolation`

