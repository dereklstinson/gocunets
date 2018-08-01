package tensor

import (
	"errors"

	gocudnn "github.com/dereklstinson/GoCudnn"
)

/*
cudnnOpTensor from the cudnn sdk documentation
This function implements the equation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
given tensors A, B, and C and scaling factors alpha1, alpha2, and beta.
The op to use is indicated by the descriptor opTensorDesc. Currently-supported ops are listed by the cudnnOpTensorOp_t enum.
Each dimension of the input tensor A must match the corresponding dimension of the destination tensor C,
and each dimension of the input tensor B must match the corresponding dimension of the destination tensor C or must be equal to 1.
In the latter case, the same value from the input tensor B for those dimensions will be used to blend into the C tensor.
The data types of the input tensors A and B must match.
 If the data type of the destination tensor C is double, then the data type of the input tensors also must be double.
If the data type of the destination tensor C is double, then opTensorCompType in opTensorDesc must be double.
Else opTensorCompType must be float.
If the input tensor B is the same tensor as the destination tensor C, then the input tensor A also must be the same tensor as the destination tensor C.
*/

//OpAdd does addition Operation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
//Or vol= op(alpha1 *A, alpha2 *B)+(beta *vol)
func (t *Volume) OpAdd(h *gocudnn.Handle, A, B *Volume, alpha1, alpha2, beta float64) error {

	_, dtypet, _, err := t.Properties()
	if err != nil {
		return err
	}
	_, dtypeA, _, err := A.Properties()
	if err != nil {
		return err
	}
	_, dtypeB, _, err := B.Properties()
	if err != nil {
		return err
	}
	if dtypeA != dtypeB || dtypeA != dtypet {
		return errors.New("Data Types don't match")
	}
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	var c gocudnn.CScalar
	switch dtypet {
	case t.thelp.Flgs.Data.Double():
		a = gocudnn.CDouble(alpha1)
		b = gocudnn.CDouble(alpha2)
		c = gocudnn.CDouble(beta)
	case t.thelp.Flgs.Data.Float():
		a = gocudnn.CFloat(alpha1)
		b = gocudnn.CFloat(alpha2)
		c = gocudnn.CFloat(beta)
	case t.thelp.Flgs.Data.Int32():
		a = gocudnn.CInt(alpha1)
		b = gocudnn.CInt(alpha2)
		c = gocudnn.CInt(beta)
	case t.thelp.Flgs.Data.Int8():
		a = gocudnn.CInt8(alpha1)
		b = gocudnn.CInt8(alpha2)
		c = gocudnn.CInt8(beta)
	case t.thelp.Flgs.Data.UInt8():
		a = gocudnn.CUInt8(alpha1)
		b = gocudnn.CUInt8(alpha2)
		c = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Add(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err != nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h, opdesc, a, A.tD, A.mem, b, B.tD, B.mem, c, t.tD, t.mem)
}

//OpMult does a multiplication Operation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
func (t *Volume) OpMult(h *gocudnn.Handle, A, B *Volume, alpha1, alpha2, beta float64) error {

	_, dtypet, _, err := t.Properties()
	if err != nil {
		return err
	}
	_, dtypeA, _, err := A.Properties()
	if err != nil {
		return err
	}
	_, dtypeB, _, err := B.Properties()
	if err != nil {
		return err
	}
	if dtypeA != dtypeB || dtypeA != dtypet {
		return errors.New("Data Types don't match")
	}
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	var c gocudnn.CScalar
	switch dtypet {
	case t.thelp.Flgs.Data.Double():
		a = gocudnn.CDouble(alpha1)
		b = gocudnn.CDouble(alpha2)
		c = gocudnn.CDouble(beta)
	case t.thelp.Flgs.Data.Float():
		a = gocudnn.CFloat(alpha1)
		b = gocudnn.CFloat(alpha2)
		c = gocudnn.CFloat(beta)
	case t.thelp.Flgs.Data.Int32():
		a = gocudnn.CInt(alpha1)
		b = gocudnn.CInt(alpha2)
		c = gocudnn.CInt(beta)
	case t.thelp.Flgs.Data.Int8():
		a = gocudnn.CInt8(alpha1)
		b = gocudnn.CInt8(alpha2)
		c = gocudnn.CInt8(beta)
	case t.thelp.Flgs.Data.UInt8():
		a = gocudnn.CUInt8(alpha1)
		b = gocudnn.CUInt8(alpha2)
		c = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Mul(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err != nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h, opdesc, a, A.tD, A.mem, b, B.tD, B.mem, c, t.tD, t.mem)
}

//OpNot does negation Operation performed on only the A  C = op ( alpha1[0] * A) + beta[0] * C,
func (t *Volume) OpNot(h *gocudnn.Handle, A *Volume, alpha1, beta float64) error {

	_, dtypet, _, err := t.Properties()
	if err != nil {
		return err
	}
	_, dtypeA, _, err := A.Properties()
	if err != nil {
		return err
	}

	if dtypeA != dtypet {
		return errors.New("Data Types don't match")
	}
	var a gocudnn.CScalar
	//var b gocudnn.CScalar
	var c gocudnn.CScalar
	switch dtypet {
	case t.thelp.Flgs.Data.Double():
		a = gocudnn.CDouble(alpha1)
		//	b = gocudnn.CDouble(alpha2)
		c = gocudnn.CDouble(beta)
	case t.thelp.Flgs.Data.Float():
		a = gocudnn.CFloat(alpha1)
		//	b = gocudnn.CFloat(alpha2)
		c = gocudnn.CFloat(beta)
	case t.thelp.Flgs.Data.Int32():
		a = gocudnn.CInt(alpha1)
		//	b = gocudnn.CInt(alpha2)
		c = gocudnn.CInt(beta)
	case t.thelp.Flgs.Data.Int8():
		a = gocudnn.CInt8(alpha1)
		//	b = gocudnn.CInt8(alpha2)
		c = gocudnn.CInt8(beta)
	case t.thelp.Flgs.Data.UInt8():
		a = gocudnn.CUInt8(alpha1)
		//	b = gocudnn.CUInt8(alpha2)
		c = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Not(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err != nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h, opdesc, a, A.tD, A.mem, nil, nil, nil, c, t.tD, t.mem)
}

//OpMax does max comparison Operation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
func (t *Volume) OpMax(h *gocudnn.Handle, A, B *Volume, alpha1, alpha2, beta float64) error {

	_, dtypet, _, err := t.Properties()
	if err != nil {
		return err
	}
	_, dtypeA, _, err := A.Properties()
	if err != nil {
		return err
	}
	_, dtypeB, _, err := B.Properties()
	if err != nil {
		return err
	}
	if dtypeA != dtypeB || dtypeA != dtypet {
		return errors.New("Data Types don't match")
	}
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	var c gocudnn.CScalar
	switch dtypet {
	case t.thelp.Flgs.Data.Double():
		a = gocudnn.CDouble(alpha1)
		b = gocudnn.CDouble(alpha2)
		c = gocudnn.CDouble(beta)
	case t.thelp.Flgs.Data.Float():
		a = gocudnn.CFloat(alpha1)
		b = gocudnn.CFloat(alpha2)
		c = gocudnn.CFloat(beta)
	case t.thelp.Flgs.Data.Int32():
		a = gocudnn.CInt(alpha1)
		b = gocudnn.CInt(alpha2)
		c = gocudnn.CInt(beta)
	case t.thelp.Flgs.Data.Int8():
		a = gocudnn.CInt8(alpha1)
		b = gocudnn.CInt8(alpha2)
		c = gocudnn.CInt8(beta)
	case t.thelp.Flgs.Data.UInt8():
		a = gocudnn.CUInt8(alpha1)
		b = gocudnn.CUInt8(alpha2)
		c = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Max(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err == nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h, opdesc, a, A.tD, A.mem, b, B.tD, B.mem, c, t.tD, t.mem)
}

//OpMin does min comparison Operation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
func (t *Volume) OpMin(h *gocudnn.Handle, A, B *Volume, alpha1, alpha2, beta float64) error {

	_, dtypet, _, err := t.Properties()
	if err != nil {
		return err
	}
	_, dtypeA, _, err := A.Properties()
	if err != nil {
		return err
	}
	_, dtypeB, _, err := B.Properties()
	if err != nil {
		return err
	}
	if dtypeA != dtypeB || dtypeA != dtypet {
		return errors.New("Data Types don't match")
	}
	var a gocudnn.CScalar
	var b gocudnn.CScalar
	var c gocudnn.CScalar
	switch dtypet {
	case t.thelp.Flgs.Data.Double():
		a = gocudnn.CDouble(alpha1)
		b = gocudnn.CDouble(alpha2)
		c = gocudnn.CDouble(beta)
	case t.thelp.Flgs.Data.Float():
		a = gocudnn.CFloat(alpha1)
		b = gocudnn.CFloat(alpha2)
		c = gocudnn.CFloat(beta)
	case t.thelp.Flgs.Data.Int32():
		a = gocudnn.CInt(alpha1)
		b = gocudnn.CInt(alpha2)
		c = gocudnn.CInt(beta)
	case t.thelp.Flgs.Data.Int8():
		a = gocudnn.CInt8(alpha1)
		b = gocudnn.CInt8(alpha2)
		c = gocudnn.CInt8(beta)
	case t.thelp.Flgs.Data.UInt8():
		a = gocudnn.CUInt8(alpha1)
		b = gocudnn.CUInt8(alpha2)
		c = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Min(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err == nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h, opdesc, a, A.tD, A.mem, b, B.tD, B.mem, c, t.tD, t.mem)
}

//OpSqrt does squareroot Operation C = op ( alpha1[0] * A ) + beta[0] * C,
func (t *Volume) OpSqrt(h *gocudnn.Handle,
	A *Volume,
	// B *Tensor,
	alpha1,
	//alpha2,
	beta float64) error {

	_, dtypet, _, err := t.Properties()
	if err != nil {
		return err
	}
	_, dtypeA, _, err := A.Properties()
	if err != nil {
		return err
	}
	/*
		_, dtypeB, _, err := B.Properties()
		if err != nil {
			return err
		}
	*/
	if dtypeA != dtypet {
		return errors.New("Data Types don't match")
	}
	var a gocudnn.CScalar
	///=	var b gocudnn.CScalar
	var c gocudnn.CScalar
	switch dtypet {
	case t.thelp.Flgs.Data.Double():
		a = gocudnn.CDouble(alpha1)
		//		b = gocudnn.CDouble(alpha2)
		c = gocudnn.CDouble(beta)
	case t.thelp.Flgs.Data.Float():
		a = gocudnn.CFloat(alpha1)
		//		b = gocudnn.CFloat(alpha2)
		c = gocudnn.CFloat(beta)
	case t.thelp.Flgs.Data.Int32():
		a = gocudnn.CInt(alpha1)
		//	b = gocudnn.CInt(alpha2)
		c = gocudnn.CInt(beta)
	case t.thelp.Flgs.Data.Int8():
		a = gocudnn.CInt8(alpha1)
		//	b = gocudnn.CInt8(alpha2)
		c = gocudnn.CInt8(beta)
	case t.thelp.Flgs.Data.UInt8():
		a = gocudnn.CUInt8(alpha1)
		//		b = gocudnn.CUInt8(alpha2)
		c = gocudnn.CUInt8(beta)

	default:
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Sqrt(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err == nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h,
		opdesc,
		a,
		A.tD,
		A.mem,
		nil,
		nil,
		nil,
		c, t.tD, t.mem)
}
