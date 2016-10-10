#ifndef NEURON_H
#define NEURON_H

class neuron;

class ANLink {
    friend class neuron;
    friend class network;
  public:
    ANLink(neuron* pinN, neuron* poutN = 0, float in = 1.0f, float w = 0.0f, float add = 0.0f);
    ~ANLink();
    inline void set_add_term(float add);
    inline void set_weight(float weight);

  private:
    neuron* pinput_neuron;  // neuron with this->Link connected to N input
    neuron* poutput_neuron; // neuron with this->Link connected to N output

    float iadd;   // add term for input layer neurons
    float dwprev; // weight change from previous run

    float wt;   // weight
    float ival; // input value on the link, passed to *pinput_neuron
};

inline void ANLink::set_add_term(float add)
{
    iadd = add;
}

inline void ANLink::set_weight(float weight)
{
    wt = weight;
}

class neuron {
    friend class network;

    enum FUNCTION { LINEAR, SIGMOID };

    neuron();
    ~neuron();

    void add_bias();
    void add_input(neuron* poutN = 0); // add input link

    void fire();       // process my inputs and pass oval to N connected to my output
    void input_fire(); // push data from input to hidden layer

    inline void set_function(enum FUNCTION func);
    inline int get_input_links_number() const;
    inline int get_output_links_number() const;
    inline ANLink* get_input_link(int i) const;
    inline ANLink* get_output_link(int i) const;

 private:
    std::vector<ANLink*> inputs;  // bias link and input links. Each knows which connected to its I/P
    std::vector<ANLink*> outputs; // Each N knows which N connected to its output

    float delta;    // delta
    float oval;     // old value

    int function;   // Neuron function :: Liniar, Sigmoid etc..
};

inline void neuron::set_function(enum FUNCTION func) {
  function = func;
}
inline int neuron::get_input_links_number() const {
  return (int)inputs.size();
}
inline int neuron::get_output_links_number() const {
  return (int)outputs.size();
}
inline ANLink* neuron::get_input_link(int i) const {
  if (i > get_input_links_number() -1 || i < 0)
    return 0;
  return inputs[i];
}
inline ANLink* neuron::get_output_link(int i) const {
  if (i > get_output_links_number() -1 || i < 0)
    return 0;
  return outputs[i];
}

///////// fire ////////
inline void neuron::input_fire() {
  oval = (inputs[0]->ival + inputs[0]->iadd) * inputs[0]->wt;  // input layer normalization

  switch(function) {      // single input for input layer neuron
  default: 
  case LINEAR:
    break;
  case SIGMOID:
    oval = 1.0f / (1.0f + exp(float((-1.0f) * oval)));
    break;
  }

  for (int i = 0; i < get_output_links_number(); i++) // transfer output to links connected to output
    outputs[i]->ival=oval;
}

inline void neuron::fire() {
  oval = 0.0f;  // oval = SUM(in[] * wt[])
  for (int i = 0; i < get_input_links_number(); i++)
    oval += inputs[i]->ival * inputs[i]->wt;
  switch(function) {
  default:
  case LINEAR:
    break;
  case SIGMOID:
    oval = 1.0f / (1.0f + exp(float((-1.0f) * oval)));
    break;
  }
  for (int i = 0; i < get_output_links_number(); i++)  // transfer outputs to links connected to output
    outputs[i]->ival = oval;
}
  
#endif NEURON_H
