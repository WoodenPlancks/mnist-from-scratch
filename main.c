#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"
#include "util/asm_commands.h"

#include <stdint.h>

#define NUMBER_RUNS 10
#define NUMBER_IMGS 3000
#define RUN_HASH true


int main() {
	srand(time(NULL));

	//TRAINING
	// int number_imgs = 10000;
	// Img** imgs = csv_to_imgs("./data/mnist_test.csv", number_imgs);
	// NeuralNetwork* net = network_create(784, 300, 10, 0.1);
	// network_train_batch_imgs(net, imgs, number_imgs);
	// network_save(net, "testing_net");

	uint64_t run_times[NUMBER_RUNS] = {0};
	double scores[NUMBER_RUNS] = {0};
	// PREDICTING
	
	Img** imgs = csv_to_imgs("data/mnist_test.csv", NUMBER_IMGS);
	NeuralNetwork* net = network_load("testing_net");

	printf("Getting ready to run iterations!\n");

	for(int i=0; i<NUMBER_RUNS; i++)
	{
		printf("Running iteration %d...\n", i);
		uint64_t start_time = rdtsc();
		scores[i] = network_predict_imgs(net, imgs, NUMBER_IMGS, RUN_HASH);
		uint64_t end_time = rdtsc();
		run_times[i] = end_time - start_time;		
	}

	uint64_t sum_run_times = 0;
	double sum_scores = 0;

	for(int i=0; i<NUMBER_RUNS; i++)
	{
		sum_run_times += run_times[i];
		sum_scores += scores[i];
	}
	printf("\n");

	if(RUN_HASH)
	{
		printf("HASHED INFERENCE\n");
	}
	else
	{
		printf("NORMAL (NON-HASHED) INFERENCE\n");
	}
	
	printf("Average Accuracy (percent): \t %1.5f\n", sum_scores/NUMBER_RUNS);
	printf("Average Duration (cycles): \t %ld\n", sum_run_times/NUMBER_RUNS);
	printf("\n");


	imgs_free(imgs, NUMBER_IMGS);
	network_free(net);
	return 0;
}