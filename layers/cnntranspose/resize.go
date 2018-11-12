package cnntranspose

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/layers"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Resize does the cnntranpose Resize Style
func Resize(handle *cudnn.Handler,
	frmt cudnn.TensorFormat,
	dtype cudnn.DataType,
	upscaleddims []int32, //UpscaledDims will be the dims of the input before the convolution
	filterdims []int32,
	convmode gocudnn.ConvolutionMode,
	pad,
	stride,
	dilation []int32,
	inputlayer bool,
	managedmem bool) (*Layer, error) {
	return build(handle, frmt, dtype, upscaleddims, filterdims, convmode, pad, stride, dilation, convtransposeresize, inputlayer, managedmem)
}

func (l *Layer) resizeforward(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := l.trans.ResizeForward(handle, x.T(), l.hiddenmem.T())
	if err != nil {
		return err
	}
	err = handle.Sync()
	if err != nil {
		return err
	}
	return l.conv.ForwardProp(handle, wspace, l.hiddenmem, y)

}
func (l *Layer) resizeBackPropData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {

	err := l.conv.BackPropData(handle, wspace, l.hiddenmem, y)
	if err != nil {
		return err
	}
	err = handle.Sync()
	if err != nil {
		return err
	}
	err = x.DeltaT().SetValues(handle, 0)
	return l.trans.ResizeBackward(handle, x.DeltaT(), l.hiddenmem.DeltaT())

}
func (l *Layer) resizeBackPropFilter(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	return l.conv.BackPropFilter(handle, wspace, l.hiddenmem, y)

}

func (l *Layer) resizeBackPropFilterData(handle *cudnn.Handler, wspace *gocudnn.Malloced, x, y *layers.IO) error {
	err := l.conv.BackPropFilterData(handle, wspace, l.hiddenmem, y)
	if err != nil {
		return err
	}
	err = handle.Sync()
	if err != nil {
		return err
	}
	err = x.DeltaT().SetValues(handle, 0)
	return l.trans.ResizeBackward(handle, x.DeltaT(), l.hiddenmem.DeltaT())

}
