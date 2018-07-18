package gomem

//Harray3dTestArray1 Used for testing purposes
func Harray3dTestArray1() HArray3d {
	var testarray = HArray3d{Data: []float32{
		1.0, 2.0, 3.0, 4.0,
		5.0, 4.0, 3.0, 2.0,
		2.0, 4.0, 6.0, 8.0,
		9.0, 7.0, 5.0, 3.0,

		4.0, 4.0, 3.0, 3.0,
		2.0, 2.0, 6.0, 6.0,
		1.0, 1.0, 3.0, 3.0,
		8.0, 8.0, 4.0, 4.0,

		1.0, 1.0, 1.0, 1.0,
		2.0, 2.0, 2.0, 2.0,
		4.0, 4.0, 4.0, 3.0,
		1.0, 1.0, 1.0, 1.0,
	},
		X: 4,
		Y: 4,
		Z: 3,
	}
	return testarray
}

//Harray3dTestArray2 Used for testing purposes
func Harray3dTestArray2() HArray3d {
	var testarray = HArray3d{Data: []float32{
		1.0, 2.0, 3.0,
		5.0, 4.0, 3.0,
		2.0, 4.0, 6.0,

		4.0, 4.0, 3.0,
		2.0, 2.0, 6.0,
		1.0, 1.0, 3.0,

		1.0, 1.0, 1.0,
		2.0, 2.0, 2.0,
		4.0, 4.0, 4.0,
	},
		X: 3,
		Y: 3,
		Z: 3,
	}
	return testarray
}

//Harray3dTestArray2 Used for testing purposes
func Harray3dTestArray3() HArray3d {
	var testarray = HArray3d{Data: []float32{
		2.0, 3.0, 3.0,
		1.0, 3.0, 3.0,
		4.0, 7.0, 2.0,

		4.0, 1.0, 4.0,
		2.0, 4.0, 6.0,
		1.0, 3.0, 3.0,

		1.0, 1.0, 2.0,
		2.0, 1.0, 2.0,
		3.0, 4.0, 6.0,
	},
		X: 3,
		Y: 3,
		Z: 3,
	}
	return testarray
}
