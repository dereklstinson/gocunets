package tconv

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Op  handles the type conversion operations
type Op struct {
	int8tofloatd *gocudnn.XInt8ToFloatD
}

//CreateInt8ToFloat creates a converter that takes int8 and converts it to float32.  Good for taking images that are stored as int8 on the gpu
//and convert them to float32
func CreateInt8ToFloat(handle *cudnn.Handler) (*Op, error) {
	x, err := gocudnn.Xtra{}.MakeIntToFloatD(handle.XHandle())
	if err != nil {
		return nil, err
	}
	return &Op{
		int8tofloatd: x,
	}, nil
}

//Int8ToFloat will take a float and make it int8
func (o *Op) Int8ToFloat(handle *cudnn.Handler, x *tensor.Volume, y *tensor.Volume, normalize bool) error {
	return o.int8tofloatd.Int8ToFloat(handle.XHandle(), x.TD(), x.Memer(), y.TD(), y.Memer(), normalize)
}
