//Package nvutil are functions that use the other nvidia packages and allows them to be used with each other
package nvutil

import (
	"fmt"
	"testing"

	"github.com/dereklstinson/GoCudnn/npp"
)

func TestFindSrcROIandDstROI(t *testing.T) {
	var src npp.Size
	src.Set(108, 108)
	var dst npp.Size
	dst.Set(32, 32)
	srcr, dstr, err := FindSrcROIandDstROI(src, dst, 16, 16)
	if err != nil {
		panic(err)
	}
	fmt.Println(srcr, dstr)
}
