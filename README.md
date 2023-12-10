Original code at https://github.com/markkraay/mnist-from-scratch!

## Inference and Parameters
In order to run the inference, simply run 'make'.
To enable or disable hashing during inference, set RUN_HASH in main.c to true or false respectively.
You can also set the number of inferences to do with NUMBER_RUNS. For the report, this value was 1.
To change the number of weights being hashed at one time, change WEIGHTS_PER_HASH in nn.c.
To run a no-op hash inference, uncomment line 155 in nn.c and comment out lines 157 to 166 in nn.c.

## Data Acquisition and Processing
For instruction and cycle count information, write 'perf stat make'.
For memory access information, write 'perf mem record make' followed by 'perf mem report'.
The arithmetic intensity was calculated in 'plots_and_data' in the 'noop' and 'fasthash' sheets for no-op and fasthash inference respectively.
   - The instruction count was divided by the sum of memory loads and memory stores for a given weight number (N).
Plots were created in the 'both' sheet.

Thank you!
