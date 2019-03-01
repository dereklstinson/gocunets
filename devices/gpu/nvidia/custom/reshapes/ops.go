package reshapes

import (
	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/cudnn"
	"github.com/dereklstinson/GoCudnn/xtra"
)

//Ops holds kernel functions.  More will be in the making
type Ops struct {
	s2b *xtra.XShapetoBatchD
	//trans      *xtra.XTransposeD
	resize     *xtra.XResizeD
	nCHWtonHWC []int32
	nHWCtonCHW []int32
}

//Stage stages the ops so that the operations can run
func Stage(handle *cudnn.Handler) (*Ops, error) {
	h := handle.XHandle()
	/*
		trans, err := xtra.CreateTransposeDesc(h)
		if err != nil {
			return nil, err
		}
	*/
	s2b, err := xtra.CreateShapetoBatchDesc(h)
	if err != nil {
		return nil, err
	}
	resize, err := xtra.CreateResizeDesc(h, false)
	return &Ops{
		s2b: s2b,
		//	trans:      trans,
		resize:     resize,
		nCHWtonHWC: []int32{0, 2, 3, 1},
		nHWCtonCHW: []int32{0, 3, 1, 2},
	}, nil

}
