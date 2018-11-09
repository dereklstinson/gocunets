package cudnn

import (
	"github.com/dereklstinson/GoCudnn"
)

//TensorFormat holds the tensor format flag
type TensorFormat gocudnn.TensorFormat

//Cu allows for easy change to gocudnn when used inside Ops
func (t TensorFormat) Cu() gocudnn.TensorFormat {
	return gocudnn.TensorFormat(t)
}

//TensorFormatFlag passes TensorFormat flags through methods
type TensorFormatFlag struct {
	c gocudnn.TensorFormatFlag
}

//NCWH dictates the layout of the tensor into NCHW format
func (t TensorFormatFlag) NCWH() TensorFormat {
	return TensorFormat(t.c.NCHW())
}

//NHWC dictates the layout of the tensor into NHWC format
func (t TensorFormatFlag) NHWC() TensorFormat {
	return TensorFormat(t.c.NHWC())
}

//NCHWvectC dictates the layout of the tensor into NCHWvectC format
func (t TensorFormatFlag) NCHWvectC() TensorFormat {
	return TensorFormat(t.c.NCHWvectC())
}
