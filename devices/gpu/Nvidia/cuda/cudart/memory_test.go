package cudart

import (
	"reflect"
	"testing"

	"github.com/dereklstinson/GoCuNets/devices"
	gocudnn "github.com/dereklstinson/GoCudnn"
)

/*

//CudaSlice is cuda memory.
type CudaSlice struct {
	mem       *gocudnn.Malloced
	dtype     devices.Type
	device    bool
	length    uint
	capacity  uint

	memcpyflg gocudnn.MemcpyKindFlag
}
*/
func TestMake(t *testing.T) {
	type args struct {
		x    interface{}
		args []uint
	}
	tests := []struct {
		name    string
		args    args
		want    *CudaSlice
		wantErr bool
	}{
		{"uint8", args{[]uint8{}, []uint{20, 25}}, &CudaSlice{&gocudnn.Malloced{}, devices.Uint8, true, 20, 25, gocudnn.MemcpyKindFlag{}}, false},
		{"uint8", args{[]uint8{}, []uint{25, 20}}, nil, true},
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Make(tt.args.x, tt.args.args...)
			if (err != nil) != tt.wantErr {
				t.Errorf("Make() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

		})
	}
}

func TestCudaSlice_Set(t *testing.T) {
	gocudnn.Cuda{}.LockHostThread()
	type args struct {
		val    interface{}
		offset uint
	}
	size := 4
	offsets := []uint{0, 1, 2, 3}
	x := make([]interface{}, size)
	x[0] = []uint32{8, 7, 6}
	x[1] = []int32{1, 2, 3}
	x[2] = []float32{3, 4, 5}
	x[3] = devices.MakeFloat16Slice([]float32{5, 6, 7})
	y := make([]interface{}, size)
	y[0] = []uint32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	y[1] = []int32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	y[2] = []float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	y[3] = devices.MakeFloat16Slice([]float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	z := make([]interface{}, size)
	z[0] = []uint32{8, 7, 6, 0, 0, 0, 0, 0, 0, 0, 8, 7, 6}
	z[1] = []int32{0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 1, 2, 3}
	z[2] = []float32{0, 0, 3, 4, 5, 0, 0, 0, 0, 0, 3, 4, 5}
	z[3] = devices.MakeFloat16Slice([]float32{0, 0, 0, 5, 6, 7, 0, 0, 0, 0, 5, 6, 7})
	slices := make([]*CudaSlice, size)
	var err error
	for i := range slices {
		slices[i], err = Make(x[i], 10)
		if err != nil {
			t.Error(err)
		}
	}
	arguments := make([]args, 4)
	for i := 0; i < 4; i++ {
		arguments[i].val = x[i]
		arguments[i].offset = offsets[i]

	}
	tests := []struct {
		name   string
		fields *CudaSlice
		args   args
	}{
		{"zero", slices[0], arguments[0]},
		{"one", slices[1], arguments[1]},
		{"two", slices[2], arguments[2]},
		{"three", slices[3], arguments[3]},
	}

	for i, tt := range tests {

		c := tt.fields
		err = c.Set(tt.args.val, tt.args.offset)
		if err != nil {
			t.Error(err)
		}
		c, err = c.Append(x[i])
		if err != nil {
			t.Error(err)
		}
		err = c.Get(y[i], 0)
		if err != nil {
			t.Error(err)
		}
		if !reflect.DeepEqual(y[i], z[i]) {
			t.Errorf("Make() = %v, want %v", y[i], z[i])

		}

	}

}
