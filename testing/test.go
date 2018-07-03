package testing

import "github.com/dereklstinson/GoCuNets/gocudnn/tensor/convolution"

func Conv() {
	x := convolution.Flags()
	x.Mode.CrossCorrelation()
}
