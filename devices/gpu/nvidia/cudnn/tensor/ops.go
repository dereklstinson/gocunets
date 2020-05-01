package tensor

import (
	"errors"

	"github.com/dereklstinson/gocunets/devices/gpu/nvidia/cudnn"
	gocudnn "github.com/dereklstinson/gocudnn"
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
type tensops struct {
	add  optensorop
	mult optensorop
	min  optensorop
	max  optensorop
	sqrt optensorop
	not  optensorop
	flg  gocudnn.OpTensorOp
}
type optensorop struct {
	mode gocudnn.OpTensorOp
	desc *gocudnn.OPTensorD
}

func (o *optensorop) isset() bool {
	if o.desc == nil {
		return false
	}
	return true
}

//OpAdd adds the op into t
func (t *Volume) OpAdd(h *cudnn.Handler, A, B *Volume, alpha1, alpha2, beta float64) error {

	if !t.op.add.isset() {
		_, dtypet, _, err := t.Properties()
		if err != nil {
			return err
		}
		t.op.add.desc, err = gocudnn.CreateOpTensorDescriptor()
		err = t.op.add.desc.Set(t.op.flg.Add(), dtypet, t.propnan)
		if err != nil {
			return errorappend("NewOpTensorDescriptor: ", err)
		}
		return t.op.add.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, alpha2, B.current.tD, B, beta, t.current.tD, t)
	}

	return t.op.add.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, alpha2, B.current.tD, B, beta, t.current.tD, t)
}
func errorappend(comment string, err error) error {
	return errors.New(comment + ": " + err.Error())
}

//OpMult does a multiplication Operation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
func (t *Volume) OpMult(h *cudnn.Handler, A, B *Volume, alpha1, alpha2, beta float64) error {

	if !t.op.mult.isset() {
		_, dtypet, _, err := t.Properties()
		if err != nil {
			return err
		}

		t.op.mult.desc, err = gocudnn.CreateOpTensorDescriptor()
		err = t.op.mult.desc.Set(t.op.flg.Mul(), dtypet, t.propnan)
		if err != nil {
			return errorappend("NewOpTensorDescriptor: ", err)
		}
		if err != nil {
			return errorappend("NewOpTensorDescriptor: ", err)
		}
		return t.op.mult.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, alpha2, B.current.tD, B, beta, t.current.tD, t)
	}
	return t.op.mult.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, alpha2, B.current.tD, B, beta, t.current.tD, t)

}

//OpNot does negation Operation performed on only the A  C = op ( alpha1[0] * A) + beta[0] * C,
func (t *Volume) OpNot(h *cudnn.Handler, A *Volume, alpha1, beta float64) error {

	if !t.op.not.isset() {
		_, dtypet, _, err := t.Properties()
		if err != nil {
			return err
		}
		t.op.not.desc, err = gocudnn.CreateOpTensorDescriptor()
		if err != nil {
			return errorappend("NewOpTensorDescriptor: ", err)
		}
		err = t.op.not.desc.Set(t.op.flg.Not(), dtypet, t.propnan)

		if err != nil {
			return err
		}
		return t.op.not.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, 0, nil, nil, beta, t.current.tD, t)
	}

	return t.op.not.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, 0, nil, nil, beta, t.current.tD, t)
}

//OpMax does max comparison Operation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
func (t *Volume) OpMax(h *cudnn.Handler, A, B *Volume, alpha1, alpha2, beta float64) error {

	if !t.op.max.isset() {
		_, dtypet, _, err := t.Properties()
		if err != nil {
			return err
		}

		t.op.max.desc, err = gocudnn.CreateOpTensorDescriptor()
		if err != nil {
			return errorappend("NewOpTensorDescriptor: ", err)
		}
		err = t.op.max.desc.Set(t.op.flg.Max(), dtypet, t.propnan)

		if err != nil {
			return errorappend("NewOpTensorDescriptor: ", err)
		}
		return t.op.max.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, alpha2, B.current.tD, B, beta, t.current.tD, t)
	}
	return t.op.max.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, alpha2, B.current.tD, B, beta, t.current.tD, t)

}

//OpMin does min comparison Operation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
func (t *Volume) OpMin(h *cudnn.Handler, A, B *Volume, alpha1, alpha2, beta float64) error {

	if !t.op.min.isset() {
		_, dtypet, _, err := t.Properties()
		if err != nil {
			return err
		}

		t.op.min.desc, err = gocudnn.CreateOpTensorDescriptor()
		if err != nil {
			return errorappend("NewOpTensorDescriptor: ", err)
		}
		err = t.op.min.desc.Set(t.op.flg.Min(), dtypet, t.propnan)

		if err != nil {
			return errorappend("NewOpTensorDescriptor: ", err)
		}
		return t.op.min.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, alpha2, B.current.tD, B, beta, t.current.tD, t)
	}
	return t.op.min.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, alpha2, B.current.tD, B, beta, t.current.tD, t)
}

//OpSqrt does squareroot Operation C = op ( alpha1[0] * A ) + beta[0] * C,
func (t *Volume) OpSqrt(h *cudnn.Handler, A *Volume, alpha1, beta float64) error {

	if !t.op.sqrt.isset() {
		_, dtypet, _, err := t.Properties()
		if err != nil {
			return err
		}

		t.op.sqrt.desc, err = gocudnn.CreateOpTensorDescriptor()
		if err != nil {
			return errorappend("NewOpTensorDescriptor: ", err)
		}
		err = t.op.sqrt.desc.Set(t.op.flg.Sqrt(), dtypet, t.propnan)

		if err != nil {
			return errorappend("NewOpTensorDescriptor: ", err)
		}
		return t.op.sqrt.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, 0, nil, nil, beta, t.current.tD, t)
	}
	return t.op.sqrt.desc.OpTensor(h.Cudnn(), alpha1, A.current.tD, A, 0, nil, nil, beta, t.current.tD, t)
}
