package reshapes

import (
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//Ops holds kernel functions.  More will be in the making
type Ops struct {
	s2b        *gocudnn.XShapetoBatchD
	trans      *gocudnn.XTransposeD
	nCHWtonHWC []int32
	nHWCtonCHW []int32
}

//Stage stages the ops so that the operations can run
func Stage(h *gocudnn.XHandle) (*Ops, error) {
	trans, err := gocudnn.Xtra{}.CreateTransposeDesc(h)
	if err != nil {
		return nil, err
	}
	s2b, err := gocudnn.Xtra{}.CreateShapetoBatchDesc(h)
	if err != nil {
		return nil, err
	}
	return &Ops{
		s2b:        s2b,
		trans:      trans,
		nCHWtonHWC: []int32{0, 2, 3, 1},
		nHWCtonCHW: []int32{0, 3, 1, 2},
	}, nil

}
