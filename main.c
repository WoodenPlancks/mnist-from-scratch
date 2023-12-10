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

#define NUMBER_RUNS 1
#define NUMBER_IMGS 1000
#define RUN_HASH true


int main() {
	srand(time(NULL));

	//TRAINING
	// int number_imgs = 10000;
	// Img** imgs = csv_to_imgs("./data/mnist_test.csv", number_imgs);
	// NeuralNetwork* net = network_create(784, 300, 10, 0.1);
	// network_train_batch_imgs(net, imgs, number_imgs);
	// network_save(net, "testing_net");

	// PREDICTING
	Img** imgs = csv_to_imgs("data/mnist_test.csv", NUMBER_IMGS);
	NeuralNetwork* net = network_load("testing_net", RUN_HASH);

	for(int i=0; i<NUMBER_RUNS; i++)
	{
		printf("Running iteration %d...\n", i);
		network_predict_imgs(net, imgs, NUMBER_IMGS, RUN_HASH);		
	}


	if(RUN_HASH)
	{
		printf("HASHED INFERENCE\n");
	}
	else
	{
		printf("NORMAL (NON-HASHED) INFERENCE\n");
	}


	imgs_free(imgs, NUMBER_IMGS);
	network_free(net, RUN_HASH);
	return 0;
}