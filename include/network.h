#ifndef NETWORK_H
#define NETWORK_H

class neuron;

//////// neuron layers ////////
class NLayer {
  friend class network;
 public:
  NLayer(int neurons_number);
  ~NLayer();

  inline int get_neurons_number() const;
  inline neuron* get_neuron(int i) const;

 private:
  wchar_t layer_name[_MAX_PATH];     // layer name
  int m_neurons_number;              // number of neurons in layer
  std::vector<neuron*> neurons;      // vector of neuron objects
};

inline int NLayer::get_neurons_number() const {
  return m_neurons_number;
}

inline neuron* NLayer::get_neurons(int i) const {
  if (i > get_neurons_number() - 1|| i < 0) 
    return 0;
  return neurons[i];
}


////////// network of layers containing neurons //////////////
class network {
 public:
  network(int layers_number, int *neurons_per_layer);    //  number of layers and neurons per layer
  network(const wchar_t* fname);                         // custom config from the file
  ~network();

  void init_links(const float* avec = 0, const float* mvec = 0, int ifunc = 0, int hfunc = 1);
  void randomize_weights(unsigned int rseed = 0);  // <---------  make RND machine

  // train network with input vector and desired output vector
  bool train(const float* ivec, float* ovec, const float* dsrdvec, float error = 0.05);

  // run network with in vector[ivec] return in[ovec] network out
  void classify(const float* ivec, float* ovec);

  bool save(const wchar_t* fname) const;

  inline int status() const;
  inline int get_layers_number() const;
  inline NLayer* get_layer(int i) const;

 private:
  int m_status;                    // status -1::err, 0::OK, 1::rnd weights
  int m_layers_number;             // number of layers in network
  std::vector<NLayer*>layers;      // vector of layer

  float m_rule;      //learning rule---> 0.2
  float m_alpha;     // momentum 0.7

  void backprop_run(const float* dsrdvec);       
  void network_output(float* ovec) const;         // get network out
};

inline int network::status() const {
  return m_status;
}

inline int network::get_layers_number() const {
  return m_layers_number;
}

inline NLayer* network::get_layer(int i) const {
  if (i > get_layers_number() - 1 || i < 0)
    return 0;
  return layers[i];
}

#endif NETWORK_H

/*
network
1 2 3 4    layers, NLayer
|N|N|N|N| .. neurons N, neuron
|N|N|..
|N|..

 */
				       

