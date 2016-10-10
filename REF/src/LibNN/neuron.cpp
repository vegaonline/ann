


#include "stdAfx.h"
#include "neuron.h"



/*
                                                    ANN neuron link
                                                                                                                              */
//////////////////////////////////////////////////constructor/destructor////////////////////////////////////////////////////////
ANLink::ANLink(ANeuron *pinn, ANeuron *poutn, float in, float w, float add) : dwprv(0.0f)
{
        pinput_neuron = pinn;           //this neuron
        poutput_neuron  = poutn;        //out neuron

        ival = in;	                //input val
        this->w = w;                    //weight

        iadd = add;
}
ANLink::~ANLink()
{
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////








/*
                                                     ANN neuron
                                                                                                                              */
//////////////////////////////////////////////////constructor/destructor////////////////////////////////////////////////////////
ANeuron::ANeuron(): oval(0), delta(0), function(LINEAR)
{
}
ANeuron::~ANeuron()
{
        for (int i = 0; i < get_input_links_number(); i++)   //delete input links
                delete inputs[i];
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////add input link//////////////////////////////////////////////////
void ANeuron::add_bias()
{        
        inputs.push_back(new ANLink(this));
}
void ANeuron::add_input(ANeuron *poutn)       //add input link
{
        //poutn - N from previous layer
        ANLink *plnk = new ANLink(this, poutn); 
        inputs.push_back(plnk);
        if (poutn)                 
                poutn->outputs.push_back(plnk);        
}
///////////////////////////////////////////////////////////////////////////////////////////////////
