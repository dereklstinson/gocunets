package utils

import "strconv"

//Dims is a QOL func that can be used to organize code by renmaing it and using it in funcs
//all it does is just return an array of the arguements passed in it.
//example: slide:=utils.Dims     Then when building a cnn layer you can do slide(2,2)
func Dims(args ...int32) []int32 {
	return args
}

//CheckError is a func that checks the error (right now it will panic if error but later it could check errors more thoughtfully )
func CheckError(err error) {
	if err != nil {
		panic(err)
	}
}

//NumbertoString will add zeros to the left of numbers. So it will file in order
func NumbertoString(x, outof int) string {
	zeros := "0"
	var flag bool
	for i := 1; i < outof; i *= 10 {

		if x%i == x {
			if flag == true {
				zeros = zeros + "0"
			}
			flag = true
		}
	}
	return zeros + strconv.Itoa(x)

}
