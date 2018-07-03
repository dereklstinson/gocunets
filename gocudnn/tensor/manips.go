package tensor

import (
	"errors"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

//SetValues sets all the values in the tensor to whatever is passed. It does this by looking at the format that is held in the tensor descriptor and auto retypes it.
func (t *Tensor) SetValues(handle *gocudnn.Handle, input float64) error {
	dtype, _, _, err := t.tD.GetDescrptor()

	if err != nil {
		return err
	}
	switch dtype {
	case t.thelp.Flgs.Data.Double():
		err = t.thelp.Funcs.SetTensor(handle, t.tD, t.mem, gocudnn.CDouble(input))
	case t.thelp.Flgs.Data.Float():
		err = t.thelp.Funcs.SetTensor(handle, t.tD, t.mem, gocudnn.CFloat(input))
	case t.thelp.Flgs.Data.Int32():
		err = t.thelp.Funcs.SetTensor(handle, t.tD, t.mem, gocudnn.CInt(input))
	default:
		return errors.New("Not supported Format to make Set All Values")
	}
	if err != nil {
		return err
	}
	return nil
}

//ScaleValues values will scale the values to the scalar passed
func (t *Tensor) ScaleValues(h *gocudnn.Handle, alpha float64) error {
	dtype, _, _, err := t.tD.GetDescrptor()
	if err != nil {
		return err
	}
	switch dtype {
	case t.thelp.Flgs.Data.Double():
		return t.thelp.Funcs.ScaleTensor(h, t.tD, t.mem, gocudnn.CDouble(alpha))
	case t.thelp.Flgs.Data.Float():
		return t.thelp.Funcs.ScaleTensor(h, t.tD, t.mem, gocudnn.CFloat(alpha))
	case t.thelp.Flgs.Data.Int32():
		return t.thelp.Funcs.ScaleTensor(h, t.tD, t.mem, gocudnn.CInt(alpha))

	}
	return errors.New("Not supported Format to make zero")
}

//AddTo formula is  (t *Tensor)= alpha*(A)+beta*(t *Tensor)
//Dim max is 5. Number of dims need to be the same.  Dim size need to match or be equal to 1.
//In the later case the same value from the A tensor for the dims will be used to blend into (t *Tensor).
func (t *Tensor) AddTo(h *gocudnn.Handle, A *Tensor, alpha, beta float64) error {
	dtype, _, _, err := t.tD.GetDescrptor()
	if err != nil {
		return err
	}
	dtypeA, _, _, err := A.tD.GetDescrptor()
	if err != nil {
		return err
	}
	if dtype != dtypeA {
		return errors.New("Datatypes Don't Match for Scalar")
	}
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	switch dtype {
	case t.thelp.Flgs.Data.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.thelp.Flgs.Data.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.thelp.Flgs.Data.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	default:
		return errors.New("Not supported Format to make zero")
	}

	return t.thelp.Funcs.AddTensor(h, a, A.tD, A.mem, b, t.tD, t.mem)
}

//Transform tensor
/*
From the SDK Documentation:
This function copies the scaled data from one tensor to another tensor with a different layout.
Those descriptors need to have the same dimensions but not necessarily the same strides.
The input and output tensors must not overlap in any way (i.e., tensors cannot be transformed in place).
This function can be used to convert a tensor with an unsupported format to a supported one.

my guess in what this does is change the format like NCHW to NHWC of t,
This is probably an EX function
*/
/*
func (t *Tensor) Transform(h *gocudnn.Handle, A *Tensor, alpha, beta float64) error {
	dtypeA, dims, _, err := A.tD.GetDescrptor()
	if err != nil {
		return err
	}
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	switch dtypeA {
	case t.thelp.Flgs.Data.Double():
		a = gocudnn.CDouble(alpha)
		b = gocudnn.CDouble(beta)
	case t.thelp.Flgs.Data.Float():
		a = gocudnn.CFloat(alpha)
		b = gocudnn.CFloat(beta)
	case t.thelp.Flgs.Data.Int32():
		a = gocudnn.CInt(alpha)
		b = gocudnn.CInt(beta)
	default:
		return errors.New("Not supported Format to make zero")
	}
	return t.thelp.Funcs.TransformTensor(h, a, A.tD, A.mem, b, t.tD, t.mem)
}
*/
//func (t *Tensor) AddAll()
