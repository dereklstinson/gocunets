package main

import (
	"fmt"

	"github.com/dereklstinson/GoCudnn/npp"

	"github.com/dereklstinson/GoCuNets/devices/gpu/nvidia/nvutil"
)

func main() {
	var src npp.Size
	src.Set(108, 108)
	var dst npp.Size
	dst.Set(32, 32)
	srcr, dstr, err := nvutil.FindSrcROIandDstROI(src, dst, 4, 4)
	if err != nil {
		panic(err)
	}
	fmt.Println("Src:", srcr)
	fmt.Println("Dst:", dstr)

}
