package utils

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
