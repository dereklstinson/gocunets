package filing

import (
	"image"
	"image/jpeg"
	"os"
	"strings"
)

//WriteImage will take an image.Image and encode it to a jpg
func WriteImage(dir string, fname string, newimage image.Image) error {
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
