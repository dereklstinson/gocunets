package fcnn

import (
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//ForwardProp does the forward propigation
func (l *Layer) ForwardProp(handle *gocudnn.Handle, x, y *layers.IO) error {
	/*
		xfmt, xdtype, xdims, _ := x.Properties()
		yfmt, ydtype, ydims, _ := y.Properties()
		nfmt, ndtype, ndims, _ := l.neurons.Properties()

		fmt.Println("x:", xfmt, xdtype, xdims)
		fmt.Println("y:", yfmt, ydtype, ydims)
		fmt.Println("n: ", nfmt, ndtype, ndims)
	*/
	err := l.conv.FwdProp(handle, l.fwd.alpha1, x.T(), l.neurons.T(), nil, l.fwd.beta, y.T())
	if err != nil {
		return appenderror("FCNN FwdProp", err)
	}

	return y.T().AddTo(handle, l.bias.T(), 1.0, 1.0)
}

//BackPropFilterData does the backpropigation for both the data and filter
func (l *Layer) BackPropFilterData(handle *gocudnn.Handle, x, y *layers.IO) error {
	err := l.BackPropData(handle, x, y)
	if err != nil {
		return err
	}
	return l.BackPropFilter(handle, x, y)
}

//BackPropData does the backprop data operation
func (l *Layer) BackPropData(handle *gocudnn.Handle, x, y *layers.IO) error {
	return l.conv.BwdPropData(handle, l.bwdd.alpha1, l.neurons.T(), y.DeltaT(), nil, l.bwdd.beta, x.DeltaT())
}

//BackPropFilter does the back prop filter operation
func (l *Layer) BackPropFilter(handle *gocudnn.Handle, x, y *layers.IO) error {
	err := l.conv.BwdPropFilt(handle, l.bwdf.alpha1, x.T(), y.DeltaT(), nil, l.bwdf.beta, l.neurons.DeltaT())
	if err != nil {
		return err
	}
	return l.conv.BwdBias(handle, l.bwdf.alpha1, y.DeltaT(), l.bwdf.beta, l.bias.DeltaT())
}
