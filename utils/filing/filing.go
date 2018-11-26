package filing

import (
	"image"
	"image/jpeg"
	"io"
	"os"
	"strings"
)

//Encoder is used to encode
type Encoder interface {
	Encode(w io.Writer) error
}

//WritetoHD will write the Encoder to the hard drive
func WritetoHD(dir, fname string, e Encoder) error {
	err1 := os.MkdirAll(dir, os.ModePerm)
	if err1 != nil {
		panic(err1)
	}

	newfile, err := os.Create(dir + fname)
	defer newfile.Close()
	if err != nil {
		return err
	}
	err = e.Encode(newfile)
	return err
}

//WriteImage will take an image.Image and encode it to a jpg
func WriteImage(dir string, fname string, newimage image.Image) error {
	if newimage == nil {
		panic("image.Image passed is nil")
	}
	if strings.Contains(dir, fname) == true {
		dir = strings.TrimSuffix(dir, fname)

	}
	err1 := os.MkdirAll(dir, os.ModePerm)
	if err1 != nil {
		panic(err1)
	}

	if strings.Contains(fname, ".jpg") == true {
		newfile, err := os.Create(dir + fname)
		if err != nil {
			return err
		}

		jpeg.Encode(newfile, newimage, nil)
		newfile.Close()
		return nil
	}
	newfile, err := os.Create(dir + fname + ".jpg")

	if err != nil {
		return err
	}
	jpeg.Encode(newfile, newimage, nil)
	newfile.Close()
	return nil
}
