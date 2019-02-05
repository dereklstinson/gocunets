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
			got, err := Make(tt.args.x, tt.args.args...)
			if (err != nil) != tt.wantErr {
				t.Errorf("Make() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Make() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCudaSlice_Set(t *testing.T) {

	type args struct {
		val    interface{}
		offset uint
	}
	size := 4
	offsets := []uint{0, 1, 2, 3}
	x := make([]interface{}, size)
	x[0] = []uint{}
	x[1] = []int{}
	x[2] = []float32{}
	x[3] = []devices.Float16{}
	slices := make([]*CudaSlice, size)
	var err error
	for i := range slices {
		slices[i], err = Make(x[i], 10)
		if err != nil {
			t.Error(err)
		}
	}
	for i := 0; i < 10; i++ {

	}
	tests := []struct {
		name   string
		fields *CudaSlice
		args   args
	}{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.fields
			c.Set(tt.args.val, tt.args.offset)
		})
	}
}
