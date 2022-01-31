# Intro to NLP - Assignment 3

## Team
|Student name| CCID |
|------------|------|
|student 1   |  karimiab   |
|student 2   |  azamani1   |

## TODOs

In this file you **must**:
- [x] Fill out the team table above. Please note that CCID is **different** from your student number.
- [x] Acknowledge all resources consulted (discussions, texts, urls, etc.) while working on an assignment. Non-detailed oral discussion with others is permitted as long as any such discussion is summarized and acknowledged by all parties.
- [x] Provide clear installation and execution instructions that TAs must follow to execute your code.

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


## Data

The assignment's training data can be found in [data/train](data/train) and the development data can be found in [data/dev](data/dev).

## Acknowledgment
This assignment was done using python's standard libraries and instructors' lectures and the main textbook (J&M).

We also reviewed NLTK documentation about [Ngrams](https://www.nltk.org/api/nltk.lm.html).

We also had a discussion with one of the other students: Yousef Nademi regarding the smoothing methods.


