package tensor

import (
	"image"
)

//ToOneImage will spread out the image.Images into one image.Image.  It will only work if all the [][]image.Image are the same size
//Channels will be in the x direction, and neurons will be in the y direction. X and Y are the amount of separation in pixels in the given dim between images.
//The separator will be in the color black!
func ToOneImage(layer [][]image.Image, X, Y int) image.Image {
	//These should all be the same size
	//	fmt.Println("Going To One Image")
	bigY := len(layer)
	bigX := len(layer[0])
	littlex := layer[0][0].Bounds().Max.X
	littley := layer[0][0].Bounds().Max.Y
	xtraY := (bigY - 1) * Y //this probably could be gotten rid of and the MaxY and MaxX equations could be changed below. I just didn't want to get out a pencil and paper
	xtraX := (bigX - 1) * X //this probably could be gotten rid of and the MaxY and MaxX equations could be changed below. I just didn't want to get out a pencil and paper
	MaxY := (littley * bigY) + xtraY
	MaxX := (littlex * bigX) + xtraX
	if MaxY > 3*MaxX {
		BigImage := image.NewRGBA(image.Rect(0, 0, MaxY, MaxX)) //switched

		for i := range layer {
			for j := range layer[i] {
				if layer[i][j] == nil {
					panic("layer ij is nil")
				}
				img := layer[i][j]
				x := img.Bounds().Max.X
				y := img.Bounds().Max.Y
				for h := 0; h < y+Y; h++ {
					for w := 0; w < x+X; w++ {
						bigi := i*(littley+Y) + h
						bigj := j*(littlex+X) + w
						//	fmt.Println(i, j, h, w)
						if w < x && h < y {
							//		fmt.Println(h, w)
							BigImage.Set(bigi, bigj, img.At(w, h)) //Switched
						}

					}
				}
			}
		}
		return BigImage
	}

	BigImage := image.NewRGBA(image.Rect(0, 0, MaxX, MaxY))

	for i := range layer {
		for j := range layer[i] {
			if layer[i][j] == nil {
				panic("layer ij is nil")
			}
			img := layer[i][j]
			x := img.Bounds().Max.X
			y := img.Bounds().Max.Y
			for h := 0; h < y+Y; h++ {
				for w := 0; w < x+X; w++ {
					bigi := i*(littley+Y) + h
					bigj := j*(littlex+X) + w
					//	fmt.Println(i, j, h, w)
					if w < x && h < y {
						//		fmt.Println(h, w)
						BigImage.Set(bigj, bigi, img.At(w, h))
					}

				}
			}
		}
	}
	return BigImage
}
