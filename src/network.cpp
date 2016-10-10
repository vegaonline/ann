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
	layers[l]->neurons[n]->inputs[i]->wt = (float)wt / 2048.0f;
      }
    }
  }
}
	  
//////////////  init links /////////////////////
void network::init_links(const float* avec, const float* mvec, int ifunc, int hfunc) {
  NLayer* plr;        // current layer
  NLayer* pprevlr;    // previous layer
  neuron* pnrn;       // neuron pointer

  int l = 0;

  //////////// input layer  /////////////
  plr = layers[l++];
  swprintf(plr->layer_name, L"input layer");
  for (int n = 0; n < plr->get_neurons_number(); n++) {
    pnrn = plr->neurons[n];
    pnrn->function = ifunc;
    pnrn->add_input();       // one input link for every "input layer" neuron
    if (avec)
      pnrn->inputs[0]->iadd = avec[n];  // default add = 0
    if (mvec)
      pnrn->inputs[0]->wt = mvec[n];    // default wt = 0
    else 
      pnrn->inputs[0]->wt = 1.0f;       //default wt = 0 for every layer neuron
  }

  /////////// hidden layer :: 1 bias ////////////
    for (int i = 0; i < m_layers_number - 2; i++) { // 1 input [l-2 hidden] 1output
      pprevlr = plr;
      plr = layers[l++];
      swprintf(plr->layer_name, L"hidden layer %d", i + 1);

      for (int n = 0; n < plr->get_neurons_number(); n++) {
	pnrn = plr->neurons[n];
	pnrn->function = hfunc;
	pnrn->add_bias();

	for (int m = 0; m < pprevlr->get_neurons_number(); m++)
	  pnrn->add_input(pprevlr->neurons[m]);
      }
    }

    //////////// output layer :: 1 bias ///////////////////
    pprevlr = plr;
    plr = layers[l++];
    swprintf(plr->layer_name, L"output layer");
 
    for (int n = 0; n < plr->get_neurons_number(); n++) {
      pnrn = plr->neurons[n];
      pnrn->function = hfunc;
      pnrn->add_bias();

      for (int m = 0; m < pprevlr->get_neurons_number(); m++) 
	pnrn->add_input(pprevlr->neurons[m]);
    }
}



///////////// backpropagation training /////////////////////
void network::backprop_run(const float* dsrdvec) {
  float nrule = m_nrule;   // learning rule
  float alpha = m_alpha;   // momentum
  float delta, dw, oval;


// get deltas for " output layer"
  for (int n = 0; n < layers[m_layers_number - 1]->get_neurons_number(); n++) {
    oval = layers[m_layers_number - 1]->neurons[n]->oval;
    layers[m_layers_number - 1]->neurons[n]->delta = oval * (1.0f - oval) * (dsrdvec[n] - oval);
  }

  // get deltas for hidden layers
  for (int l = m_layers_number - 2; l > 0; l--) {
    for (int n = 0; n < layers[l]->get_neurons_number(); n++) {
      delta = 0.0f;
      for (int i = 0; i < layers[l]->neurons[n]->get_output_links_number(); i++)
	      delta+=layers[l]->neurons[n]->outputs[i]->wt * layers[l]->neurons[n]->outputs[i]->pinput_neuron->delta;
	    oval = layers[l]->neurons[n]->oval;
	    layers[l]->neurons[n]->delta = oval * (1 - oval) * delta;
    }
  }


///////// correct weights for every layer //////////////////
    for (int l = 1; l < m_layers_number; l++) {
      for (int n = 0; n < layers[l]->get_neurons_number(); n++) {
	for (int i = 0; i < layers[l]->neurons[n]->get_input_links_number(); i++) {
	  dw = nrule * layers[l]->neurons[n]->inputs[i]->ival * layers[l]->neurons[n]->delta;
	  dw += alpha * layers[l]->neurons[n]->inputs[i]->dwprv;
	  layers[l]->neurons[n]->inputs[i]->dwprv = dw;
	  layers[l]->neurons[n]->inputs[i]->wt += dw;  // correct weight
	}
      }
    }
}


bool network::train(const float* ivec, float* ovec, const float* dsrdvec, float error) { // 0.0 - 1.0 learning
  float dst = 0.0f;
  classify(ivec, ovec); // run network, compute inputs to output
  for (int n = 0; n < layers[m_layers_number - 1]->get_neurons_number(); n++) {
    dst = fabs(ovec[n] - dsrdvec[n]);
    if (dst > error) break;
  }
  if (dst > error) {
    backprop_run(dsrdvec);  // it was trained
    return true;
  } else                     // it wasnt trained
    return false;
}

      
/////////// run network /////////////
void network::classify(const float* ivec, float* ovec) {
  // input layer
  for (int i = 0; i < layers[0]->get_neurons_number(); i++) {
    layers[0]->neurons[i]->inputs[0]->ival = ivec[i];
    layers[0]->neurons[i]->input_fire();
  }

  // hidden and output layers
  for (int l =1; l < m_layers_number; l++)
    for (int n = 0; n < layers[l]->get_neurons_number(); n++)
      layers[l]->neurons[n]->fire();
  
  // produce ANN output
  network_output(ovec);
}

////////// network out ////////////
void network::network_output(float* ovec) const{
  for (int n = 0; n < layers[m_layers_number - 1]->get_neurons_number(); n++)
    ovec[n] = layers[m_layers_number - 1]->neurons[n]->oval;
}

//////////////// save network configuration /////////
bool network::save(const wchar_t* fname) const {
  FILE* fp = _wfopen(fname, L"wt");
 if (fp) {
   fwprintf(fp, L"%d\n", m_layers_number);
   for (int l = 0; l < m_layers_number; l++)
     fwprintf(fp, L"%d ", layers[l]->get_neurons_number());
   fwprintf(fp, L"\n\n");


   //input hidden/output layer neuron function 0-linear,1-sigmoidal
   fwprintf(fp, L"%d\n%d\n\n", layers[0]->neurons[0]->function, layers[1]->neurons[0]->function);

   for (int n = 0; n < layers[0]->get_neurons_number(); n++) {
     fwprintf(fp, L"%f ", layers[0]->neurons[n]->inputs[0]->iadd); //add term  0.0 default
     fwprintf(fp, L"%f\n", layers[0]->neurons[n]->inputs[0]->wt);    //multiply term  1.0 default
   }
   fwprintf(fp, L"\n");


   for (int l = 1; l < m_layers_number; l++) {  //save all weights except input layer
     for (int n = 0; n < layers[l]->get_neurons_number(); n++) {
       for (int i = 0; i < layers[l]->neurons[n]->get_input_links_number(); i++)
	 fwprintf(fp, L"%f\n", layers[l]->neurons[n]->inputs[i]->wt);
     }
     fwprintf(fp, L"\n");
   }
   fclose(fp);
   return true;
 } else
   return false; 
}

	  

	
