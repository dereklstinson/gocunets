package gocunets

import (
	"testing"
)

func TestTensorFormat_NCHW(t *testing.T) {
	var flag TensorFormat
	var tester TensorFormat
	if flag.NCHW() != (TensorFormat)(tester.NCHW()) {
		t.Error("Flag NCHW not working", flag, tester)
	}

	if flag.NHWC() != (TensorFormat)(tester.NHWC()) {
		t.Error("Flag NHWC not working", flag, tester)
	}

	if flag.NCHWvectC() != (TensorFormat)(tester.NCHWvectC()) {
		t.Error("Flag NCHWvectC not working", flag, tester)
	}

	if flag.Strided() != (TensorFormat)(tester.Strided()) {
		t.Error("Flag Strided not working", flag, tester)
	}

}
