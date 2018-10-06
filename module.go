package gocunets

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type Module struct {
	index int64
	layer []*layer
	mem   []*layers.IO
}
type connection struct {
	sharedid int
	id       int
	recieved bool
	x        *layers.IO
}
type concatconnection struct {
	connection []connection
}
type ConcatModule struct {
	index int64
	layer []*layer
}
type SplitModule struct {
}

//ForwardProp does the forward prop for a prebuilt module
func (m *Module) ForwardProp(handle Handles, stream *gocudnn.Stream, wspace *gocudnn.Malloced, x, y *layers.IO) error {

	err := m.layer[0].forwardprop(handle, wspace, x, m.mem[0])
	if err != nil {
		return err
	}
	lnum := len(m.layer)
	for i := 1; i < lnum-1; i++ {
		err = m.layer[i].forwardprop(handle, wspace, m.mem[i-1], m.mem[i])
		if err != nil {
			return err
		}
	}
	//for m.layer[lnum-1] is the last layer in the slice and m.mem[lnum-2] should be the output from the layer before.
	err = m.layer[lnum-1].forwardprop(handle, wspace, m.mem[lnum-2], y)
	if err != nil {
		return err
	}

	return stream.Sync()
}

//BackProp does the backprop of a module
func (m *Module) BackProp(handle Handles, stream *gocudnn.Stream, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	lnum := len(m.layer)
	err := m.layer[lnum-1].backprop(handle, wspace, m.mem[lnum-2], y)

	if err != nil {
		return err
	}

	for i := lnum - 2; i > 1; i-- {
		err = m.layer[i].backprop(handle, wspace, m.mem[i-1], m.mem[i])
		if err != nil {
			return err
		}
	}

	err = m.layer[0].backprop(handle, wspace, x, m.mem[0])
	if err != nil {
		return err
	}

	return stream.Sync()
}

//UpdateWeights updates the weights of a module
func (m *Module) UpdateWeights(handle *Handles) error {
	for i := 0; i < len(m.layer); i++ {
		err := m.layer[i].updateWeights(handle)
		if err != nil {
			return err
		}
	}
	return nil
}
