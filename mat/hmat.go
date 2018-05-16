package mat

import "errors"

//First types are going to be 3d matrix arrays.
//Maybe later will add 4d matrix arrays.

//Hmat3d is a host stored matrix.
//Hmat3d contains an array of float32 and x,y,z cordinates
//Data is stored in x major format.  Everything is made public
//to reduce function call overhead.  Manipulation functions will be available to
//for the sake of ease, though.
type Hmat3d struct {
	Data    []float32
	X, Y, Z int
}

//NewHmat3d returns a 3dHmat with the size of x*y*z.
//with x,y,z being mapped to what is passed for x,y,z.
func NewHmat3d(x, y, z int) Hmat3d {
	var newhmat Hmat3d
	newhmat.X = x
	newhmat.Y = y
	newhmat.Z = z
	newhmat.Data = make([]float32, x*y*z)
	return newhmat
}
func (mat *Hmat3d) copydata() []float32 {
	copy := make([]float32, len(mat.Data))
	for i := 0; i < len(copy); i++ {
		copy[i] = mat.Data[i]
	}
	return copy
}

//PointerAt returns a pointer at the location specified
func (mat *Hmat3d) PointerAt(x, y, z int) *float32 {
	return &mat.Data[(mat.Y*mat.Z*x)+(mat.Z*y)+(z)]
}

//ValueAt returns the value at the location specified
func (mat *Hmat3d) ValueAt(x, y, z int) float32 {
	return mat.Data[(mat.Y*mat.Z*x)+(mat.Z*y)+(z)]
}

//Insert will insert a value at x,y,z
func (mat *Hmat3d) Insert(x, y, z int, value float32) {
	mat.Data[(mat.Y*mat.Z*x)+(mat.Z*y)+(z)] = value
}

//Clone returns a cloned copy of the mat that calls it.
func (mat *Hmat3d) Clone() Hmat3d {
	var clone Hmat3d
	clone.X = mat.X
	clone.Y = mat.Y
	clone.Z = mat.Z
	clone.Data = make([]float32, len(mat.Data))
	for i := 0; i < len(mat.Data); i++ {
		clone.Data[i] = mat.Data[i]
	}
	return clone
}

//Append appends to the z coordinates from an input. Supports Hmat3d, Hmat2d and []float32
//Be careful with []float32, because it tack on the float as long as it is in multiples of mat.X*mat.Y
func (mat *Hmat3d) Append(input interface{}) error {
	switch input := input.(type) {
	case Hmat2d:
		return mat.appendhmat2d(input)
	case Hmat3d:
		return mat.appendhmat3d(input)
	case []float32:
		return mat.appendhmatfloat32(input)
	default:
		return errors.New("Only Supports [][][]float32, [][]float32, Hmat2d, and Hmat3d")
	}

}

//Hmat2d contains an array of float32 and x,y cordinates
//Data is stored in x major format.  Everything is made public
//to reduce function call overhead.  Manipulation functions will be available to
//for the sake of ease, though.
type Hmat2d struct {
	Data []float32
	X, Y int
}

//NewHmat2d returns a 3dHmat with the size of x*y.
//with x,y being mapped to what is passed for x,y.
func NewHmat2d(x, y int) Hmat2d {
	var newhmat Hmat2d
	newhmat.X = x
	newhmat.Y = y
	newhmat.Data = make([]float32, x*y)
	return newhmat
}

//PointerAt returns a pointer at the location specified
func (mat *Hmat2d) PointerAt(x, y int) *float32 {
	return &mat.Data[(mat.Y*x)+(y)]
}

//ValueAt returns the value at the location specified
func (mat *Hmat2d) ValueAt(x, y int) float32 {
	return mat.Data[(mat.Y*x)+(y)]
}

//Insert will insert a value at x,y,z
func (mat *Hmat2d) Insert(x, y int, value float32) {
	mat.Data[(mat.Y*x)+(y)] = value
}

//Clone returns a cloned copy of the mat that calls it.
func (mat *Hmat2d) Clone() Hmat2d {
	var clone Hmat2d
	clone.X = mat.X
	clone.Y = mat.Y
	clone.Data = make([]float32, len(mat.Data))
	for i := 0; i < len(mat.Data); i++ {
		clone.Data[i] = mat.Data[i]
	}
	return clone
}
