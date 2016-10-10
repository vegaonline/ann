

#ifndef ANNeuron_h
#define ANNeuron_h



class ANeuron;

////////////////////////////////////ANN Link///////////////////////////////////////////////////
class ANLink
{
        friend class ANeuron;
        friend class ANNetwork;
public:
        ANLink(ANeuron *pinn, ANeuron *poutn = 0, float in = 1.0f, float w = 0.0f, float add = 0.0f);
        ~ANLink();        

        inline void set_add_term(float add);
        inline void set_weight(float weight);

private:
        ANeuron *pinput_neuron;     //neuron with this->Link connected to N input
        ANeuron *poutput_neuron;    //neuron with this->Link connected to N output

        float iadd;        //add term for input layer neurons
        float dwprv;       //wight change from previous run

        float w;           //weight
        float ival;        //input value on the link, passed to *pinput_neuron
};

inline void ANLink::set_add_term(float add)
{
        iadd = add;
}

inline void ANLink::set_weight(float weight)
{
        w = weight;
}
////////////////////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////ANN neuron///////////////////////////////////////////////////
class ANeuron
{      
        friend class ANNetwork;
public:
        enum FUNCTION {LINEAR, SIGMOID};

        ANeuron();
        ~ANeuron();        

        void add_bias();
        void add_input(ANeuron *poutn = 0);  //add input link               

        void input_fire();              //push data from input layer to hidden
        void fire();                    //process my inputs and pass oval to N connected to my output

        inline void set_function(enum FUNCTION func);
        inline int get_input_links_number() const;
        inline int get_output_links_number() const;
        inline ANLink *get_input_link(int i) const;
        inline ANLink *get_output_link(int i) const;

private:
        vector<ANLink *> inputs;    //bias link and input links  Every N knows what N connected to its inputs
        vector<ANLink *> outputs;   //Every N knows what N connected to its output

        float delta;                //delta
        float oval;                 //output value

        int function;               //Neuron function: LINIAR,SIGMOID,....

};

inline void ANeuron::set_function(enum FUNCTION func)
{
        function = func;
}
inline int ANeuron::get_input_links_number() const
{
        return (int)inputs.size();
}

inline int ANeuron::get_output_links_number() const
{
        return (int)outputs.size();
}

inline ANLink *ANeuron::get_input_link(int i) const
{
        if (i > get_input_links_number() - 1 || i < 0)
                return 0;
        return inputs[i];
}

inline ANLink *ANeuron::get_output_link(int i) const
{
        if (i > get_output_links_number() - 1 || i < 0)
                return 0;
        return outputs[i];
}
////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////fire////////////////////////////////////////////////////////
inline void ANeuron::input_fire()
{
        //input layer normalization
        oval = (inputs[0]->ival + inputs[0]->iadd) * inputs[0]->w;

        //single input for input layer neuron
        switch (function) {
        default:
        case LINEAR:
                break;        

        case SIGMOID:
                oval = 1.0f / (1.0f + exp(float((-1.0f) * oval)));
                break;
        }

        //transfer my output to links connected to my output
        for (int i = 0; i < get_output_links_number(); i++)
                outputs[i]->ival = oval;
}

inline void ANeuron::fire()
{
        //oval = SUM (in[]*w[])
        oval = 0.0f;

        //compute output for Neuron
        for (int i = 0; i < get_input_links_number(); i++)
                oval += inputs[i]->ival * inputs[i]->w;

        switch (function) {
        default:
        case LINEAR:
                break;

        case SIGMOID:
                oval = 1.0f / (1.0f + exp(float((-1.0f) * oval)));
                break;
        }

        //transfer my output to links connected to my output
        for (int i = 0; i < get_output_links_number(); i++)
                outputs[i]->ival = oval;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

#endif







