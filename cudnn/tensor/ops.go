package tensor

import (
	"errors"

	"github.com/dereklstinson/GoCuNets/cudnn"
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
func (t *Volume) OpAdd(h *cudnn.Handler, A, B *Volume, alpha1, alpha2, beta float64) error {

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
	a := gocudnn.CScalarByDataType(dtypet, alpha1)
	b := gocudnn.CScalarByDataType(dtypet, alpha2)
	c := gocudnn.CScalarByDataType(dtypet, beta)
	if a == nil || b == nil || c == nil {
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Add(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err != nil {
		return errorappend("NewOpTensorDescriptor: ", err)
	}
	//fmt.Println(t.propnan, A.propnan, B.propnan)
	//fmt.Println(fmtt, fmtA, fmtB)
	//fmt.Println(t.mem.Ptr())
	return t.ophelp.Funcs.OpTensor(h.Cudnn(), opdesc, a, A.tD, A.memgpu, b, B.tD, B.memgpu, c, t.tD, t.memgpu)
}
func errorappend(comment string, err error) error {
	return errors.New(comment + ": " + err.Error())
}

//OpMult does a multiplication Operation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
func (t *Volume) OpMult(h *cudnn.Handler, A, B *Volume, alpha1, alpha2, beta float64) error {

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

	a := gocudnn.CScalarByDataType(dtypet, alpha1)
	b := gocudnn.CScalarByDataType(dtypet, alpha2)
	c := gocudnn.CScalarByDataType(dtypet, beta)
	if a == nil || b == nil || c == nil {
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Mul(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err != nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h.Cudnn(), opdesc, a, A.tD, A.memgpu, b, B.tD, B.memgpu, c, t.tD, t.memgpu)
}

//OpNot does negation Operation performed on only the A  C = op ( alpha1[0] * A) + beta[0] * C,
func (t *Volume) OpNot(h *cudnn.Handler, A *Volume, alpha1, beta float64) error {

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
	a := gocudnn.CScalarByDataType(dtypet, alpha1)
	c := gocudnn.CScalarByDataType(dtypet, beta)
	if a == nil || c == nil {
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Not(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err != nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h.Cudnn(), opdesc, a, A.tD, A.memgpu, nil, nil, nil, c, t.tD, t.memgpu)
}

//OpMax does max comparison Operation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
func (t *Volume) OpMax(h *cudnn.Handler, A, B *Volume, alpha1, alpha2, beta float64) error {

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
	a := gocudnn.CScalarByDataType(dtypet, alpha1)
	b := gocudnn.CScalarByDataType(dtypet, alpha2)
	c := gocudnn.CScalarByDataType(dtypet, beta)
	if a == nil || b == nil || c == nil {
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Max(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err == nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h.Cudnn(), opdesc, a, A.tD, A.memgpu, b, B.tD, B.memgpu, c, t.tD, t.memgpu)
}

//OpMin does min comparison Operation C = op ( alpha1[0] * A, alpha2[0] * B ) + beta[0] * C,
func (t *Volume) OpMin(h *cudnn.Handler, A, B *Volume, alpha1, alpha2, beta float64) error {

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
	a := gocudnn.CScalarByDataType(dtypet, alpha1)
	b := gocudnn.CScalarByDataType(dtypet, alpha2)
	c := gocudnn.CScalarByDataType(dtypet, beta)
	if a == nil || b == nil || c == nil {
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Min(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err == nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h.Cudnn(), opdesc, a, A.tD, A.memgpu, b, B.tD, B.memgpu, c, t.tD, t.memgpu)
}

//OpSqrt does squareroot Operation C = op ( alpha1[0] * A ) + beta[0] * C,
func (t *Volume) OpSqrt(h *cudnn.Handler, A *Volume, alpha1, beta float64) error {

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
	a := gocudnn.CScalarByDataType(dtypet, alpha1)

	c := gocudnn.CScalarByDataType(dtypet, beta)
	if a == nil || c == nil {
		return errors.New("Not supported Format")
	}
	opdesc, err := t.ophelp.NewOpTensorDescriptor(t.ophelp.Flgs.Sqrt(), dtypet, t.propnan)
	defer opdesc.DestroyDescriptor()
	if err == nil {
		return err
	}

	return t.ophelp.Funcs.OpTensor(h.Cudnn(),
		opdesc,
		a,
		A.tD,
		A.memgpu,
		nil,
		nil,
		nil,
		c, t.tD, t.memgpu)
}
