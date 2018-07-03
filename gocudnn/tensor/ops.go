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

//OpAdd does and addition Operation
func (t *Tensor) OpAdd(h *gocudnn.Handle, A, B *Tensor, alpha1, alpha2, beta float64) error {

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
	default:
		return errors.New("Not supported Format to make zero")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Add(), dtypet, t.propnan)

	t.ophelp.Funcs.OpTensor(h, opdesc, a, A.tD, A.mem, b, B.tD, B.mem, c, t.tD, t.mem)
	return opdesc.DestroyDescriptor()
}
