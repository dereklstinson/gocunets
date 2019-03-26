package utils

import (
	"testing"
)

func TestFindMaxOutput(t *testing.T) {
	maxoutput, err := FindMaxOutput([]int32{10, 3, 32, 32}, []int32{30, 3, 10, 10}, true)
	if err != nil {
		t.Error(err)
	}
	maxout := []int32{10, 30, 41, 41}
	for i := range maxoutput.Output {
		if maxoutput.Output[i] != maxout[i] {
			t.Error(maxoutput.Output[i], maxout[i], "Values not same")
		}
	}
}
