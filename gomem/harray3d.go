package gomem

import "errors"

//First types are going to be 3d arrays.
//Maybe later will add 4d arrays.

//HArray3d is a host stored array
//HArray3d contains an array of float32 and x,y,z cordinates
//Data is stored in x major forray.  Everything is made public
//to reduce function call overhead.  Manipulation functions will be available to
//for the sake of ease, though.
type HArray3d struct {
	Data    []float32
	X, Y, Z int
}

//NewHArray3d returns a 3dHray with the size of x*y*z.
//with x,y,z being mapped to what is passed for x,y,z.
func NewHArray3d(x, y, z int) HArray3d {
	var newhray HArray3d
	newhray.X = x
	newhray.Y = y
	newhray.Z = z
	newhray.Data = make([]float32, x*y*z)
	return newhray
}

//Addto is a element by element add.
func (ray *HArray3d) Addto(input *HArray3d) error {
	if len(ray.Data) != len(input.Data) {
		return errors.New("Addto: Arrays Not Same Size")

	}
	for i := 0; i < len(ray.Data); i++ {
		ray.Data[i] += input.Data[i]
	}
	return nil
}

//PointerAt returns a pointer at the location specified
func (ray *HArray3d) PointerAt(x, y, z int) *float32 {
	return &ray.Data[(ray.Y*ray.Z*x)+(ray.Z*y)+(z)]
}

//ValueAt returns the value at the location specified
func (ray *HArray3d) ValueAt(x, y, z int) float32 {
	return ray.Data[(ray.Y*ray.Z*x)+(ray.Z*y)+(z)]
}

//Insert will insert a value at x,y,z
func (ray *HArray3d) Insert(x, y, z int, value float32) {
	ray.Data[(ray.Y*ray.Z*x)+(ray.Z*y)+(z)] = value
}

//Clone returns a cloned copy of the ray that calls it.
func (ray *HArray3d) Clone() HArray3d {
	var clone HArray3d
	clone.X = ray.X
	clone.Y = ray.Y
	clone.Z = ray.Z
	clone.Data = make([]float32, len(ray.Data))
	for i := 0; i < len(ray.Data); i++ {
		clone.Data[i] = ray.Data[i]
	}
	return clone
}

//CloneEmpty returns a cloned in size but zero value of the ray that calls it.
func (ray *HArray3d) CloneEmpty() HArray3d {
	var clone HArray3d
	clone.X = ray.X
	clone.Y = ray.Y
	clone.Z = ray.Z
	clone.Data = make([]float32, len(ray.Data))

	return clone
}

//OverWrite will overwrite the values from another HArray3d must be same size
func (ray *HArray3d) OverWrite(input HArray3d) error {
	if len(ray.Data) != len(input.Data) {
		return errors.New("OVerWrite: Arrays not the same size")
	}
	for i := 0; i < len(ray.Data); i++ {
		ray.Data[i] = input.Data[i]
	}
	return nil
}

//SetAll sets all the values of CUarray3d to whatever is passed in input
func (ray *HArray3d) SetAll(input float32) {
	for i := 0; i < len(ray.Data); i++ {
		ray.Data[i] = input
	}
}

//Append appends to the z coordinates from an input. Supports HArray3d, HArray2d and []float32
//Be careful with []float32, because it tack on the float as long as it is in multiples of ray.X*ray.Y
func (ray *HArray3d) Append(input interface{}) error {
	switch input := input.(type) {
	case HArray2d:
		return ray.appendHArray2d(input)
	case HArray3d:
		return ray.appendHArray3d(input)
	case []float32:
		return ray.appendhrayfloat32(input)
	default:
		return errors.New("Only Supports [][][]float32, [][]float32, HArray2d, and HArray3d")
	}

}
