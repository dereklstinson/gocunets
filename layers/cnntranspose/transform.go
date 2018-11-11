package cnntranspose

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Transform does the Trans
func Transform(handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	upscaleddims []int32, //UpscaledDims will be the dims of the input before the convolution
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	managedmem bool) (*Layer, error) {
	return build(handle, frmt, dtype, upscaleddims, filterdims, convmode, pad, stride, dilation, convtransposetrans, managedmem)
}

func (l *Layer) tranformforward(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	if l.thelper == nil {
		err := l.makehelper(x, l.hiddenmem)
		if err != nil {
			return err
		}
	}
	if utils.CompareInt32(x.T().TD().Dims(), l.thelper.Dims()) == false {
		l.thelper.Destroy()
		err := l.makehelper(x, l.hiddenmem)
		if err != nil {
			return err
		}
	}
	err := l.trans.TransformForward(handle, 1, 0, x.T(), l.hiddenmem.T(), l.thelper)
	if err != nil {
		return err
	}
	return l.conv.ForwardProp(handle, wspace, l.hiddenmem, y)

}
func (l *Layer) makehelper(x, y *layers.IO) error {
	var err error
	l.thelper, err = l.trans.MakeTransformHelper(x.T(), y.T())
	return err
}
func (l *Layer) transformBackPropFilterData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := l.conv.BackPropFilterData(handle, wspace, l.hiddenmem, y)
	if err != nil {
		return err
	}

	return l.trans.TransformBackward(handle, 1, 0, x.DeltaT(), l.hiddenmem.DeltaT(), l.thelper)

}
func (l *Layer) transformBackPropData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := l.conv.BackPropData(handle, wspace, l.hiddenmem, y)
	if err != nil {
		return err
	}

	return l.trans.TransformBackward(handle, 1, 0, x.DeltaT(), l.hiddenmem.DeltaT(), l.thelper)

}
