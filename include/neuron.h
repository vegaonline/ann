class neuron;

class ANLink {
    friend class neuron;
    // friend class ANNetwork;
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
    enum FUNCTION { LINEAR, SIGMOID };

    neuron();
    ~neuron();

    void add_bias();
    void add_input(neuron* poutn = 0); // add input link

    void fire();       // process my inputs and pass oval to N connected to my output
    void input_fire(); // push data from input to hidden layer

    inline void set_function(enum FUNCTION func);
    inline int get_input_links_number() const;
    inline int get_output_links_number() const;
    inline ANLink* get_input_link(int i) const;
    inline ANLink* get_output_link(int i) const;
}