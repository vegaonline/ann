#include "neuron.h"
#include "network.h"

///////// network layer //////
NLayer::NLayer(int neurons_number):m_neurons_number(neurons_number) {
  for (int n = 0; n < m_neurons_number; n++) 
    neurons.push_back(new neuron());
}
NLayer::~NLayer() {
  for (int n = 0; n < m_neurons_number; n++)     // delete neurons from layer
    delete neurons[n];
}


///////// network ///////////
network::network(int layers_number, int* neurons_per_layer):m_status(0), m_rule(0.2f), m_alpha(0.7f) {
  m_layers_number = layers_number;
  for(int l = 0; l < layers_number; l++)
    layers.push_back(new NLayer(neurons_per_layer[l]));
}

network::network(const wchar_t* fname):m_status(-1), m_rule(0.2f), m_alpha(0.7f) {
  int res = 0;
  int nnum = 0, ifunc = 0, hfunc = 0;
  float wt = 0.0f;

  std::FILE *fp = _wfopen(fname, L"rt");
  if (fp) {
    if ((res = fwscanf(fp, L"%d", &m_layers_number))!=1) {
      fclose(fp);
      m_status = -1;
      return;
    }
    for (int l = 0; l < m_layers_number; l++) {
      if ((res = fwscanf(fp, L"%d", &nnum))!=1) {
	fclose(fp);
	m_status = -1;
	return;
      } else 
	layers.push_back(new NLayer(nnum));
    }
    if ((res = fwscanf(fp, L"%d %d", &ifunc, &hfunc))! = 2) {  // function for hidden/output layer
      ifunc = 0;   // default LINEAR for input
      hfunc = 1;   // default SIGMOID for hidden / output
    }

    std::vector<float> adds, mults;
    for(int n = 0; n < layers[0]->get_neurons_number(); n++) {
      float a, m;
      if (res = fwscanf(fp, L"%f %f", &a, &m)!= 2) {    // blank network file ?
	for (int n = 0; n < layers[0]->get_neurons_number(); n++) {
	  adds.push_back(0.0);                                    // default add = 0
	  mults.push_back(1.0);                                   // default mult = 1
	}
	break;
      }
      adds.push_back(a);
      mults.push_back(m);
    }

    init_links(&adds[0], &mults[0], ifunc, hfunc);
    for (int l = 1; l < m_layers_number; l++) {    // load all weights except input layer
      for (int n = 0; n < layers[l]->get_neurons_number(); n++) {   // num of neurons in layer
	for (int i = 0; i < layers[l]->neurons->get_input_links_number(); i++) { // num of inputs in neuron
	  if ((res = fwscanf(fp, L"%f", &wT)) != 1) {     // blank network file ?
	    fclose(fp);
	    m_status = 1;
	    randomize_weights((unsigned int) time(0));
	    return;
	  } else
	    layers[l]->neurons[n]->inputs[i]->wt = wt;
	}
      }
    }
    fclose(fp);
    m_status = 0;
  } else 
    m_status = -1;
}

network::~network() {
  for (int l = 0; l < m_layers_number; l++)   // delete layers
    delete layers[l];
}


////////// neuron weights //////////////////
void ANNetwork::randomize_weights(unsigned int rseed) { 
  int wt;
        
  srand(rseed);

  //input layer remains with w=1.0
  for (int l = 1; l < m_layers_number; l++) {
    for (int n = 0; n < layers[l]->get_neurons_number(); n++) {
      for (int i = 0; i < layers[l]->neurons[n]->get_input_links_number(); i++) {
	wt = 0xFFF & rand();
	wt -= 0x800;
	layers[l]->neurons[n]->inputs[i]->wt = (float)w / 2048.0f;
      }
    }
  }
}
	  
