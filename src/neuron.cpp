#include "base.h"
#include "neuron.h"

/////// neuron link ///////
ANLinkk::ANLink(neuron* pinN, neuron* poutN,float in,  float wt, float add)::dwprev(0.0f) {
  pinput_neuron = pinN;
  poutput_neuron = poutN;
  ival = in;
  this->wt = wt;
  iadd = add;
}
ANLink::~ANLink() {
}

////// neuron /////////
neuron::neuron():oval(0), delta(0), function(LINEAR) {
}
neuron::~neuron() {
  for (int i = 0; i < get_input_links_number(); i++)    // delete input links
    delete inputs[i];
}


////////// add input link ////////
void neuron::add_bias() {
  inputs.push_back(new ANLink(this));
}
void neuron::add_input(neuron *poutN) {   // add input link
  ANLink* plink = new ANLink(this, poutN);
  inputs.push_back(plink);
  if (poutN)
    poutN->outputs.push_back(plink);
}
  
