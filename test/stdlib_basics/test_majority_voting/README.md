# Test for OpenAI API served by VLLM

## Requirement

anaconda / miniconda / miniforge.

Make sure to run the test with multiple cores available (e.g. in a cloud instance / cluster job).
Although you may think 1 core is enough,
vllm could get stuck due to deadlock if so.

## Installation

Run the `install.sh` script, which needs to be done only once.
The script creates a new conda environment named "mellea_tbf" only for the purposes of testing or contributing to the think budget-forcing feature.

Run `./install.sh`

## Testing

``` shell
./run_test.sh
```
