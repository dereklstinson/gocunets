package gomem

//probably going to delete this

//HArray2d contains an array of float32 and x,y cordinates
//Data is stored in x major forray.  Everything is made public
//to reduce function call overhead.  Manipulation functions will be available to
//for the sake of ease, though.
type HArray2d struct {
	Data []float32
	X, Y int
}

//NewHArray2d returns a 3dHray with the size of x*y.
//with x,y being mapped to what is passed for x,y.
func NewHArray2d(x, y int) HArray2d {
	var newhray HArray2d
	newhray.X = x
	newhray.Y = y
	newhray.Data = make([]float32, x*y)
	return newhray
}

//PointerAt returns a pointer at the location specified
func (ray *HArray2d) PointerAt(x, y int) *float32 {
	return &ray.Data[(ray.Y*x)+(y)]
}

//ValueAt returns the value at the location specified
func (ray *HArray2d) ValueAt(x, y int) float32 {
	return ray.Data[(ray.Y*x)+(y)]
}

//Insert will insert a value at x,y,z
func (ray *HArray2d) Insert(x, y int, value float32) {
	ray.Data[(ray.Y*x)+(y)] = value
}

//Clone returns a cloned copy of the ray that calls it.
func (ray *HArray2d) Clone() HArray2d {
	var clone HArray2d
	clone.X = ray.X
	clone.Y = ray.Y
	clone.Data = make([]float32, len(ray.Data))
	for i := 0; i < len(ray.Data); i++ {
		clone.Data[i] = ray.Data[i]
	}
	return clone
}

//CloneEmpty returns a cloned in size but zero value of the ray that calls it.
func (ray *HArray2d) CloneEmpty() HArray2d {
	var clone HArray2d
	clone.X = ray.X
	clone.Y = ray.Y
	clone.Data = make([]float32, len(ray.Data))

	return clone
}

//SetAll sets all the values of CUarray3d to whatever is passed in input
func (ray *HArray2d) SetAll(input float32) {
	for i := 0; i < len(ray.Data); i++ {
		ray.Data[i] = input
	}
}
