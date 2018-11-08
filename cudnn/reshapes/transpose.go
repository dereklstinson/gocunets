package reshapes

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//TransposeChannelForward will take a nchw and change it to a nhwc and vice-versa. Will find the transpose of x and put it in y
func (o *Ops) TransposeChannelForward(handle *gocudnn.XHandle, x, y *tensor.Volume) error {
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
	var fflg gocudnn.TensorFormatFlag
	switch xfrmt {
	case fflg.NCHW():
		return o.trans.Transpose(handle, o.nCHWtonHWC, x.TD(), x.Memer(), y.TD(), y.Memer())
	case fflg.NHWC():
		return o.trans.Transpose(handle, o.nHWCtonCHW, x.TD(), x.Memer(), y.TD(), y.Memer())
	}
	return errors.New("TransposeChannelXtoY - Passed Non supported tensor format")
}

//TransposeChannelBackward will take a nchw and change it to a nhwc and vice-versa. Will find the transpose of y and put it in x
func (o *Ops) TransposeChannelBackward(handle *gocudnn.XHandle, x, y *tensor.Volume) error {
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
	var fflg gocudnn.TensorFormatFlag
	switch xfrmt {
	case fflg.NCHW():
		return o.trans.Transpose(handle, o.nCHWtonHWC, y.TD(), y.Memer(), x.TD(), x.Memer())
	case fflg.NHWC():
		return o.trans.Transpose(handle, o.nHWCtonCHW, y.TD(), y.Memer(), x.TD(), x.Memer())
	}
	return errors.New("TransposeChannelXtoY - Passed Non supported tensor format")
}

//TransposeChannel will take x and transpose it along the channel.
//The function works by creating a new volume and replacing x with it and deleting the old x.
func (o *Ops) TransposeChannel(handle *gocudnn.XHandle, x *tensor.Volume) error {
	xfrmt, _, _, err := x.Properties()
	if err != nil {
		return err
	}

	y, err := o.gettransposevol(handle, x)
	if err != nil {
		return err
	}

	var fflg gocudnn.TensorFormatFlag
	switch xfrmt {
	case fflg.NCHW():
		err = o.trans.Transpose(handle, o.nCHWtonHWC, x.TD(), x.Memer(), y.TD(), y.Memer())
		x.Destroy()
		*x = *y
		//y.Destroy()
		return err
	case fflg.NHWC():
		err = o.trans.Transpose(handle, o.nHWCtonCHW, x.TD(), x.Memer(), y.TD(), y.Memer())
		x.Destroy()
		*x = *y
		//y.Destroy()
		return err
	}
	return errors.New("TransposeChannelXtoY - Passed Non supported tensor format")
}

//GetTransposeOutputProperties will get the volume of a transpose operation handled through this op
func (o *Ops) GetTransposeOutputProperties(handle *gocudnn.XHandle, x *tensor.Volume) (gocudnn.TensorFormat, gocudnn.DataType, []int32, []int32, bool, error) {
	xmal := x.Memer()
	if xmal != nil {
		var managed bool
		var flgloc gocudnn.LocationFlag
		if flgloc.Unified() == xmal.Stored() {
			managed = true
		}
		frmt, dtype, dims, perm, err := o.trans.GetChannelTransposeOutputProperties(x.TD())

		return frmt, dtype, dims, perm, managed, err

	}

	return 255, 255, nil, nil, false, errors.New("memory is nil")
}

func (o *Ops) gettransposevol(handle *gocudnn.XHandle, x *tensor.Volume) (*tensor.Volume, error) {
	xmal := x.Memer()
	if xmal != nil {
		var managed bool
		var flgloc gocudnn.LocationFlag
		if flgloc.Unified() == xmal.Stored() {
			managed = true
		}
		frmt, dtype, dims, _, err := o.trans.GetChannelTransposeOutputProperties(x.TD())
		if err != nil {
			return nil, err
		}

		return tensor.Build(frmt, dtype, dims, managed)

	}

	return nil, errors.New("memory is nil")
}
