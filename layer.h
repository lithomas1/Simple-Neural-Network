#include <string>
#include <vector>
//legacy code
struct Node {
	std::vector<float> weights = {};
	float activation = 0;
	float value = 0;
	float delta = 0;
	float bias = 0;
};
struct Layer {
	//TODO add bias
	int nodes;
	std::string activation;
	std::vector<float> input_vec;
	std::vector<std::vector<float>> weights;
	Layer(int num_nodes, std::string activation_fun, std::vector<float> input_vector, bool rand_init) {
		nodes = num_nodes;
		activation = activation_fun;
		input_vec = input_vector;
		int input_size = static_cast<int>(input_vec.size());
		if (rand_init == true) {
			for (int i = 0; i < nodes; i++) {
				std::vector<float> randvec(input_size);
				std::generate(randvec.begin(), randvec.end(), std::rand); //Random number generation
				weights.push_back(randvec);
			}
		}
		else {
			//initialize weights to zero otherwise
			std::vector<std::vector<float>> emptyvec(input_size, std::vector<float> (nodes,0));
			weights = emptyvec;
		}
	}
};