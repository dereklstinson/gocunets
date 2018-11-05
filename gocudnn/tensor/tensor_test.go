package tensor

import "testing"

func TestTensor(t *testing.T) {
	//	tflgs := Flags()

}

/*
n, c, h, w := int32(1), int32(3), int32(4), int32(2)
sharedims := []int32{n, c, h, w}
//tensor dims a 1,4,4,2... slide is 32,8,2,1
chw := c * h * w
hw := h * w
ostride := []int32{chw, hw, w, 1}
xDesc, err := tensor.NewTensor4dDescriptorEx(float, sharedims, ostride)
if err != nil {
	t.Error(err)
}

x, y, z := int32(1), int32(4), int32(4)
xyz := x * y * z
yz := y * z
stride := []int32{ostride[0] * xyz, ostride[1] * xyz, ostride[2] * yz, ostride[3] * z}
outputdims := []int32{(stride[0] * sharedims[0]) / (chw * xyz), (sharedims[1] * stride[1]) / (yz * hw), (sharedims[2] * stride[2]) / (w * z), sharedims[3] * stride[3]}
//tensor dims a 1,4,4,2...

*/
