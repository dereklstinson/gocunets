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

func (n *Network) ForwardProp(handle Handles, stream *gocudnn.Stream, wspace *gocudnn.Malloced, x, y IO) error {
	var err error
	err = n.modules[0].ForwardProp(handle, stream, wspace, x.mem, n.io[0].mem)
	if err != nil {
		return err
	}
	length := len(n.modules)
	for i := 1; i < length-1; i++ {
		err = n.modules[i].ForwardProp(handle, stream, wspace, n.io[i-1].mem, n.io[i].mem)
		if err != nil {
			return err
		}
	}
	return n.modules[length-1].ForwardProp(handle, stream, wspace, n.io[length-1].mem, y.mem)
}
func (n *Network) BackProp(handle Handles, stream *gocudnn.Stream, wspace *gocudnn.Malloced, x, y IO) error {
	var err error
	length := len(n.modules)
	err = n.modules[length-1].BackProp(handle, stream, wspace, n.io[length-1].mem, y.mem)

	if err != nil {
		return err
	}

	for i := length - 2; i > 0; i-- {
		err = n.modules[i].BackProp(handle, stream, wspace, n.io[i-1].mem, n.io[i].mem)
		if err != nil {
			return err
		}
	}
	return n.modules[0].BackProp(handle, stream, wspace, x.mem, n.io[0].mem)
}
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
