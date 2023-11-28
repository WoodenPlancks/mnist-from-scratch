#pragma once

#include "../matrix/matrix.h"
#include "../util/img.h"
#include <stdbool.h>
#include <stdint.h>

typedef struct {
	int input;
	int hidden;
	int output;
	double learning_rate;
	Matrix* hidden_weights;
	Matrix* output_weights;
	Matrix* hidden_hashes;
	Matrix* output_hashes;
} NeuralNetwork;

NeuralNetwork* network_create(int input, int hidden, int output, double lr);
void network_train(NeuralNetwork* net, Matrix* input_data, Matrix* output_data);
void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size);
Matrix* network_predict_img(NeuralNetwork* net, Img* img, bool hash);
double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n, bool hash);
Matrix* network_predict(NeuralNetwork* net, Matrix* input_data, bool hash);
void network_save(NeuralNetwork* net, char* file_string);
NeuralNetwork* network_load(char* file_string, bool hash);
void network_print(NeuralNetwork* net);
void network_free(NeuralNetwork* net, bool hash);
bool check_weights(NeuralNetwork* net);
double hash_func(double weight);