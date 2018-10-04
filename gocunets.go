package gocunets

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

type IO struct {
	mem *layers.IO
}

type Handles struct {
	cudnn   *gocudnn.Handle
	xhandle *gocudnn.XHandle
}
type Network struct {
	modules []*Module
	io      []IO
}

/*
func (n *Network) ForwardProp(handle Handles, stream *gocudnn.Stream, wspace *gocudnn.Malloced, x, y []*IO) error {
	for i := range x {
		for j := range x[i].children {
			module := x[i].children[j]

			n.modules[module].ForwardProp(x)
		}

	}
	return nil
}
*/
func CreateHandles(dev *gocudnn.Device, trainingfolder string) (*Handles, error) {

	x := gocudnn.NewHandle()

	y, err := gocudnn.Xtra{}.MakeXHandle(trainingfolder, dev)
	if err != nil {
		return nil, err
	}
	return &Handles{
		cudnn:   x,
		xhandle: y,
	}, nil
}

/*
func MakeLayer(input interface{}) Layer {
	switch x := input.(type) {

	}
}
func (l *Layer) Setup() error {
	if l.cnn != nil {
		//return l.cnn.(handle.cudnn, wpace, x, y)
	}
	if l.fcnn != nil {
		return l.fcnn.Setup(handle.cudnn, x, y)
	}
	if l.activation != nil {
		return l.Setup.BackProp(handle.cudnn, x, y)
	}
	if l.softmax != nil {
		return l.Setup.BackProp(handle.cudnn, x, y)
	}
	if l.pool != nil {
		return l.pool.Set(handle.cudnn, x, y)
	}
	return errors.New("Layer Not Set Up")
}
*/
