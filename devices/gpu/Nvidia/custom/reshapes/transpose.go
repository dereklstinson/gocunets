package reshapes

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn"
	"github.com/dereklstinson/GoCuNets/devices/gpu/Nvidia/cudnn/tensor"
)

//TransposeChannelForward will take a nchw and change it to a nhwc and vice-versa. Will find the transpose of x and put it in y
func (o *Ops) TransposeChannelForward(handle *cudnn.Handler, x, y *tensor.Volume) error {
	xfrmt, _, xdims, err := x.Properties()
	if err != nil {
		return err
	}
	_, _, ydims, err := y.Properties()
	if err != nil {
		return err
	}
	//ydims[3]==xdims[1]||ydims[1]==xdims[3] one of these being false is ok, but if they are both false then uhh oh
	if ydims[0] != xdims[0] || !(ydims[3] == xdims[1] || ydims[1] == xdims[3]) {
		return errors.New("Dims are not matching up N for both tensors need to be the same and channel dims need to be switched")
	}
	var fflg cudnn.TensorFormatFlag
	switch xfrmt {
	case fflg.NCHW():
		return o.trans.Transpose(handle.XHandle(), o.nCHWtonHWC, x.TD(), x.Memer(), y.TD(), y.Memer())
	case fflg.NHWC():
		return o.trans.Transpose(handle.XHandle(), o.nHWCtonCHW, x.TD(), x.Memer(), y.TD(), y.Memer())
	}
	return errors.New("TransposeChannelXtoY - Passed Non supported tensor format")
}

//TransposeChannelBackward will take a nchw and change it to a nhwc and vice-versa. Will find the transpose of y and put it in x
func (o *Ops) TransposeChannelBackward(handle *cudnn.Handler, x, y *tensor.Volume) error {
	xfrmt, _, xdims, err := x.Properties()
	if err != nil {
		return err
	}
	_, _, ydims, err := y.Properties()
	if err != nil {
		return err
	}
	//ydims[3]==xdims[1]||ydims[1]==xdims[3] one of these being false is ok, but if they are both false then uhh oh
	if ydims[0] != xdims[0] || !(ydims[3] == xdims[1] || ydims[1] == xdims[3]) {
		return errors.New("Dims are not matching up N for both tensors need to be the same and channel dims need to be switched")
	}
	var fflg cudnn.TensorFormatFlag
	switch xfrmt {
	case fflg.NCHW():
		return o.trans.Transpose(handle.XHandle(), o.nCHWtonHWC, y.TD(), y.Memer(), x.TD(), x.Memer())
	case fflg.NHWC():
		return o.trans.Transpose(handle.XHandle(), o.nHWCtonCHW, y.TD(), y.Memer(), x.TD(), x.Memer())
	}
	return errors.New("TransposeChannelXtoY - Passed Non supported tensor format")
}

//TransposeChannel will take x and transpose it along the channel.
//The function works by creating a new volume and replacing x with it and deleting the old x.
func (o *Ops) TransposeChannel(handle *cudnn.Handler, x *tensor.Volume) error {
	xfrmt, _, _, err := x.Properties()
	if err != nil {
		return err
	}

	y, err := o.gettransposevol(handle, x)
	if err != nil {
		return err
	}

	var fflg cudnn.TensorFormatFlag
	switch xfrmt {
	case fflg.NCHW():
		err = o.trans.Transpose(handle.XHandle(), o.nCHWtonHWC, x.TD(), x.Memer(), y.TD(), y.Memer())
		//	x.Destroy()
		*x = *y
		//y.Destroy()
		return err
	case fflg.NHWC():
		err = o.trans.Transpose(handle.XHandle(), o.nHWCtonCHW, x.TD(), x.Memer(), y.TD(), y.Memer())
		//	x.Destroy()
		*x = *y
		//y.Destroy()
		return err
	}
	return errors.New("TransposeChannelXtoY - Passed Non supported tensor format")
}

//GetTransposeOutputProperties will get the volume of a transpose operation handled through this op
func (o *Ops) GetTransposeOutputProperties(handle *cudnn.Handler, x *tensor.Volume) (cudnn.TensorFormat, cudnn.DataType, []int32, []int32, error) {

	frmt, dtype, dims, perm, err := o.trans.GetChannelTransposeOutputProperties(x.TD())

	return cudnn.TensorFormat(frmt), cudnn.DataType(dtype), dims, perm, err

}

func (o *Ops) gettransposevol(handle *cudnn.Handler, x *tensor.Volume) (*tensor.Volume, error) {

	frmt, dtype, dims, _, err := o.trans.GetChannelTransposeOutputProperties(x.TD())
	if err != nil {
		return nil, err
	}

	return tensor.Build(handle, cudnn.TensorFormat(frmt), cudnn.DataType(dtype), dims)

}
