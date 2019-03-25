package pooling

import gocudnn "github.com/dereklstinson/GoCudnn"

//Settings contains all the info needed to build a pooing layer
type Settings struct {
	Mode    gocudnn.PoolingMode `json:"mode,omitempty"`
	Nan     gocudnn.NANProp     `json:"nan,omitempty"`
	Managed bool                `json:"managed,omitempty"`
	Window  []int32             `json:"window,omitempty"`
	Padding []int32             `json:"padding,omitempty"`
	Stride  []int32             `json:"stride,omitempty"`
}
