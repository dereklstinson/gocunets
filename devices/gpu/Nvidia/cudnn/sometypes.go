package cudnn

import (
	gocudnn "github.com/dereklstinson/GoCudnn"
)

//SizeT is used often in cuda and this is a type that is used internally that will interface with gocudnn
//that interfaces with cudnn (c version)
type SizeT gocudnn.SizeT

//Cu used to interface with gocudnn if having to use gocudnn functions
func (s SizeT) Cu() gocudnn.SizeT {
	return gocudnn.SizeT(s)
}

/*
//Memer is wrapper interface for gocudnn.Memer
type Memer interface {
	gocudnn.Memer
}
*/
