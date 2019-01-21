package reshapes

import (
	"github.com/dereklstinson/GoCuNets/cudnn"
	"github.com/dereklstinson/GoCuNets/cudnn/tensor"
	"github.com/dereklstinson/GoCuNets/utils"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//TransformForward fills tensor y with x to the best of its ability.
func (o *Ops) TransformForward(handle *cudnn.Handler, alpha, beta float64, x, y *tensor.Volume, yhelper *TransFormHelper) error {
	dtype := x.TD().DataType()
	a := gocudnn.CScalarByDataType(dtype, alpha)
	b := gocudnn.CScalarByDataType(dtype, beta)
	/*
		z, zz, zzz, zzzz := x.TDStrided().GetDescrptor()
		w, ww, www, wwww := x.TD().GetDescrptor()
		g, gg, ggg, gggg := y.TDStrided().GetDescrptor()
		h, hh, hhh, hhhh := y.TD().GetDescrptor()
		fmt.Println(z, zz, zzz, zzzz)
		fmt.Println(w, ww, www, wwww)
		fmt.Println(h, hh, hhh, hhhh)
		fmt.Println(g, gg, ggg, gggg)
	*/
	return gocudnn.Tensor{}.TransformTensor(handle.Cudnn(), a, x.TDStrided(), x.Memer(), b, yhelper.desc, y.Memer())
}

//TransformBackward fills tensor x with the values of y to the best of its ability
func (o *Ops) TransformBackward(handle *cudnn.Handler, alpha, beta float64, x, y *tensor.Volume, yhelper *TransFormHelper) error {
	dtype := x.TD().DataType()
	a := gocudnn.CScalarByDataType(dtype, alpha)
	b := gocudnn.CScalarByDataType(dtype, beta)
	/*
		z, zz, zzz, zzzz := x.TDStrided().GetDescrptor()
		w, ww, www, wwww := x.TD().GetDescrptor()
		g, gg, ggg, gggg := y.TDStrided().GetDescrptor()
		h, hh, hhh, hhhh := y.TD().GetDescrptor()
		fmt.Println(z, zz, zzz, zzzz)
		fmt.Println(w, ww, www, wwww)
		fmt.Println(h, hh, hhh, hhhh)
		fmt.Println(g, gg, ggg, gggg)
	*/
	return gocudnn.Tensor{}.TransformTensor(handle.Cudnn(), a, yhelper.desc, y.Memer(), b, x.TDStrided(), x.Memer())
}

//TransFormHelper helps with the transform
type TransFormHelper struct {
	desc   *gocudnn.TensorD
	dims   []int32
	stride []int32
}

//Destroy destroys the malloced memory made for the tensordescriptor inside the struct.
func (t *TransFormHelper) Destroy() error {
	return t.desc.DestroyDescriptor()
}

//Dims returns the dims for the descriptor
func (t *TransFormHelper) Dims() []int32 {
	return t.dims
}

//MakeTransformHelper will find the strides needed for dims of src to fit dest.  Put it into a strided tensor descriptor and return it to use with transform tensor
//It is best if  destdims[i]= n*srcdims[i]. where n is an int
func (o *Ops) MakeTransformHelper(src, dest *tensor.Volume) (*TransFormHelper, error) {
	sdims := src.TD().Dims()
	dtype := src.TD().DataType()
	destdims := dest.TD().Dims()
	strides := utils.FindFittingStride(sdims, destdims)
	stridedtensordescriptor, err := gocudnn.Tensor{}.NewTensor4dDescriptorEx(dtype, sdims, strides)
	if err != nil {
		return nil, err
	}
	return &TransFormHelper{
		desc:   stridedtensordescriptor,
		dims:   sdims,
		stride: strides,
	}, nil

}
