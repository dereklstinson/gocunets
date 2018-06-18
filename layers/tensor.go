//Package layers contains shared things between layers.  It also contains functions that will be supplimental to cudnn.
package layers

import gocudnn "github.com/dereklstinson/GoCudnn"

//IO is input and output of a convolution layer.  It is the communication between layers and networks
type IO struct {
	desc *gocudnn.TensorD
	mem  gocudnn.Memer
	dmem gocudnn.Memer
}

//TensorD returns the descriptor tensor
func (i *IO) TensorD() *gocudnn.TensorD {
	return i.desc
}

//Mem returns the main memery
func (i *IO) Mem() gocudnn.Memer {
	return i.mem
}

//DMem returns the error backprop memory for the tensor memory
func (i *IO) DMem() gocudnn.Memer {
	return i.dmem
}
