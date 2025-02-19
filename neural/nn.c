#include "nn.h"
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include "../matrix/ops.h"
#include "../neural/activations.h"
#include "../matrix/ops.h"
#include "../util/asm_commands.h"
#include <openssl/sha.h>

#define MAXCHAR 1000
#define WEIGHTS_PER_HASH 1
#define HASH_DIM_HIDDEN (net->input * net->hidden)/(WEIGHTS_PER_HASH)
#define HASH_DIM_OUTPUT (net->output * net->hidden)/(WEIGHTS_PER_HASH)
#define OUTPUT_DIM net->output
#define INPUT_DIM net->input
#define HIDDEN_DIM net->hidden

// 784, 300, 10
NeuralNetwork* network_create(int input, int hidden, int output, double lr) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	net->input = input;
	net->hidden = hidden;
	net->output = output;
	net->learning_rate = lr;
	Matrix* hidden_layer = matrix_create(hidden, input);
	Matrix* output_layer = matrix_create(output, hidden);
	matrix_randomize(hidden_layer, hidden);
	matrix_randomize(output_layer, output);
	net->hidden_weights = hidden_layer;
	net->output_weights = output_layer;

	return net;
}

void network_train(NeuralNetwork* net, Matrix* input, Matrix* output) {
	// Feed forward
	Matrix* hidden_inputs	= dot(net->hidden_weights, input);
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs);

	// Find errors
	Matrix* output_errors = subtract(output, final_outputs);
	Matrix* transposed_mat = transpose(net->output_weights);
	Matrix* hidden_errors = dot(transposed_mat, output_errors);
	matrix_free(transposed_mat);

	// Backpropogate
	// output_weights = add(
	//		 output_weights, 
	//     scale(
	// 			  net->lr, 
	//			  dot(
	// 		 			multiply(
	// 						output_errors, 
	//				  	sigmoidPrime(final_outputs)
	//					), 
	//					transpose(hidden_outputs)
	// 				)
	//		 )
	// )
	Matrix* sigmoid_primed_mat = sigmoidPrime(final_outputs);
	Matrix* multiplied_mat = multiply(output_errors, sigmoid_primed_mat);
	transposed_mat = transpose(hidden_outputs);
	Matrix* dot_mat = dot(multiplied_mat, transposed_mat);
	Matrix* scaled_mat = scale(net->learning_rate, dot_mat);
	Matrix* added_mat = add(net->output_weights, scaled_mat);

	matrix_free(net->output_weights); // Free the old weights before replacing
	net->output_weights = added_mat;

	matrix_free(sigmoid_primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	// hidden_weights = add(
	// 	 net->hidden_weights,
	// 	 scale (
	//			net->learning_rate
	//    	dot (
	//				multiply(
	//					hidden_errors,
	//					sigmoidPrime(hidden_outputs)	
	//				)
	//				transpose(inputs)
	//      )
	// 	 )
	// )
	// Reusing variables after freeing memory
	sigmoid_primed_mat = sigmoidPrime(hidden_outputs);
	multiplied_mat = multiply(hidden_errors, sigmoid_primed_mat);
	transposed_mat = transpose(input);
	dot_mat = dot(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->learning_rate, dot_mat);
	added_mat = add(net->hidden_weights, scaled_mat);
	matrix_free(net->hidden_weights); // Free the old hidden_weights before replacement
	net->hidden_weights = added_mat; 

	matrix_free(sigmoid_primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);

	// Free matrices
	matrix_free(hidden_inputs);
	matrix_free(hidden_outputs);
	matrix_free(final_inputs);
	matrix_free(final_outputs);
	matrix_free(output_errors);
	matrix_free(hidden_errors);
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size) {
	for (int i = 0; i < batch_size; i++) {
		if (i % 100 == 0) printf("Img No. %d\n", i);
		Img* cur_img = imgs[i];
		Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
		Matrix* output = matrix_create(10, 1);
		output->entries[cur_img->label][0] = 1; // Setting the result
		network_train(net, img_data, output);
		matrix_free(output);
		matrix_free(img_data);
	}
}

Matrix* network_predict_img(NeuralNetwork* net, Img* img, bool hash) {
	// printf("In network_predict_img...\n");
	Matrix* img_data = matrix_flatten(img->img_data, 0);
	Matrix* res = network_predict(net, img_data, hash);
	matrix_free(img_data);
	return res;
}

double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n, bool hash) {
	int n_correct = 0;
	// printf("In network_predict_imgs...\n");
	for (int i = 0; i < n; i++) {
		Matrix* prediction = network_predict_img(net, imgs[i], hash);
		if (matrix_argmax(prediction) == imgs[i]->label) {
			n_correct++;
		}
		matrix_free(prediction);
	}
	return 1.0 * n_correct / n;
}

// uint64_t quickhash64(char *str, uint64_t mix){}

uint64_t quickhash64(char *str, uint64_t mix)
{ // set 'mix' to some value other than zero if you want a tagged hash          
    uint64_t mulp = 2654435789;
    mix ^= 104395301;

    while(*str)
        mix += (*str++ * mulp) ^ (mix >> 23);

    return mix ^ (mix << 37);
}

bool check_weights(NeuralNetwork* net)
{
	
	for(int i=0; i<HASH_DIM_HIDDEN; i++)
	{
		// uint64_t stitched_weights = net->hidden_weights->entries[(i * WEIGHTS_PER_HASH) % HIDDEN_DIM][(i * WEIGHTS_PER_HASH)/HIDDEN_DIM];
		char* stitched_weights = (char*) calloc(WEIGHTS_PER_HASH*sizeof(double), sizeof(char));

		for(int k=0; (k<WEIGHTS_PER_HASH) & (k + (i * WEIGHTS_PER_HASH) < HIDDEN_DIM * INPUT_DIM); k++)
		{
			memcpy(stitched_weights + sizeof(double) * k, &net->hidden_weights->entries[((i + k) * WEIGHTS_PER_HASH) % HIDDEN_DIM][((i + k) * WEIGHTS_PER_HASH)/HIDDEN_DIM], sizeof(double));
		}
		

		if(net->hidden_hashes[i] != quickhash64(stitched_weights, 0))
		{
			return false;
		}
	}

	for(int i=0; i<HASH_DIM_OUTPUT; i++)
	{

		char* stitched_weights = (char*) calloc(WEIGHTS_PER_HASH*sizeof(double), sizeof(char));	

		for(int k=0; (k<WEIGHTS_PER_HASH) & (k + (i * WEIGHTS_PER_HASH) < HIDDEN_DIM * OUTPUT_DIM); k++)
		{
			memcpy(stitched_weights + sizeof(double) * k, &net->output_weights->entries[((i + k) * WEIGHTS_PER_HASH) % OUTPUT_DIM][((i + k) * WEIGHTS_PER_HASH)/OUTPUT_DIM], sizeof(double));
		}

		if(net->output_hashes[i] != quickhash64(stitched_weights, 0))
		{
			return false;
		}
	}
	
	return true;
}

Matrix* network_predict(NeuralNetwork* net, Matrix* input_data, bool hash)
{
	// Hash the weights
	if(hash)
	{
		if(!check_weights(net))
		{
			// printf("WEIGHTS ARE INVALID.\n");
		}
	}

	Matrix* hidden_inputs	= dot(net->hidden_weights, input_data);
	Matrix* hidden_outputs = apply(sigmoid, hidden_inputs);
	Matrix* final_inputs = dot(net->output_weights, hidden_outputs);
	Matrix* final_outputs = apply(sigmoid, final_inputs);
	Matrix* result = softmax(final_outputs);

	matrix_free(hidden_inputs);
	matrix_free(hidden_outputs);
	matrix_free(final_inputs);
	matrix_free(final_outputs);

	return result;
}

void network_save(NeuralNetwork* net, char* file_string) {
	mkdir(file_string, 0777);
	// Write the descriptor file
	chdir(file_string);
	FILE* descriptor = fopen("descriptor", "w");
	fprintf(descriptor, "%d\n", net->input);
	fprintf(descriptor, "%d\n", net->hidden);
	fprintf(descriptor, "%d\n", net->output);
	fclose(descriptor);
	matrix_save(net->hidden_weights, "hidden");
	matrix_save(net->output_weights, "output");
	printf("Successfully written to '%s'\n", file_string);
	chdir("-"); // Go back to the orignal directory
}

NeuralNetwork* network_load(char* file_string, bool hash) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	char entry[MAXCHAR];
	chdir(file_string);

	FILE* descriptor = fopen("descriptor", "r");
	fgets(entry, MAXCHAR, descriptor);
	net->input = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->hidden = atoi(entry);
	fgets(entry, MAXCHAR, descriptor);
	net->output = atoi(entry);
	fclose(descriptor);
	net->hidden_weights = matrix_load("hidden");
	net->output_weights = matrix_load("output");

	printf("Loading hash matrices...\n");

	if(hash)
	{
		net->hidden_hashes = (uint64_t*) calloc(HASH_DIM_HIDDEN, sizeof(uint64_t));
		net->output_hashes = (uint64_t*) calloc(HASH_DIM_OUTPUT, sizeof(uint64_t));

		for(int i=0; i<HASH_DIM_HIDDEN; i++)
		{
			char* stitched_weights = (char*) calloc(WEIGHTS_PER_HASH*sizeof(double), sizeof(char));

			for(int k=0; (k<WEIGHTS_PER_HASH) & (k + (i * WEIGHTS_PER_HASH) < HIDDEN_DIM * INPUT_DIM); k++)
			{
				// printf("%d Weight is %lf\n", k, net->hidden_weights->entries[((i + k) * WEIGHTS_PER_HASH) % HIDDEN_DIM][((i + k) * WEIGHTS_PER_HASH)/HIDDEN_DIM]);
				memcpy(stitched_weights + sizeof(double) * k, &net->hidden_weights->entries[((i + k) * WEIGHTS_PER_HASH) % HIDDEN_DIM][((i + k) * WEIGHTS_PER_HASH)/HIDDEN_DIM], sizeof(double));
			}

			// printf("memcpy result: %.64s, %d\n", stitched_weights, WEIGHTS_PER_HASH*sizeof(double));
			net->hidden_hashes[i] = quickhash64(stitched_weights, 0);

		}

		printf("Done hashing hidden weights.\n");

		for(int i=0; i<HASH_DIM_OUTPUT; i++)
		{
			char* stitched_weights = (char*) calloc(WEIGHTS_PER_HASH*sizeof(double), sizeof(char));

			for(int k=0; (k<WEIGHTS_PER_HASH) & (k + (i * WEIGHTS_PER_HASH) < HIDDEN_DIM * OUTPUT_DIM); k++)
			{
				memcpy(stitched_weights + sizeof(double) * k, &net->output_weights->entries[((i + k) * WEIGHTS_PER_HASH) % OUTPUT_DIM][((i + k) * WEIGHTS_PER_HASH)/OUTPUT_DIM], sizeof(double));
			}

			net->output_hashes[i] = quickhash64(stitched_weights, 0);
		}
	}

	printf("Successfully loaded network from '%s'\n", file_string);
	chdir("-"); // Go back to the original directory
	return net;
}

void network_print(NeuralNetwork* net) {
	printf("# of Inputs: %d\n", net->input);
	printf("# of Hidden: %d\n", net->hidden);
	printf("# of Output: %d\n", net->output);
	printf("Hidden Weights: \n");
	matrix_print(net->hidden_weights);
	printf("Output Weights: \n");
	matrix_print(net->output_weights);
}

void network_free(NeuralNetwork *net, bool hash) {
	matrix_free(net->hidden_weights);
	matrix_free(net->output_weights);
	if(hash)
	{
		free(net->hidden_hashes);
		free(net->output_hashes);
	}
	free(net);
	net = NULL;
}