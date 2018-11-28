package ui

import "image"

type WindowHandler struct {
	img     *ImageHandlerV2
	para    *ParagraphHandlerV2
	blength int
}

func MakeWindowHandler(bufflength int) *WindowHandler {

	return &WindowHandler{
		img:     MakeImageHandlerV2(bufflength),
		para:    MakeParagraphHandlerV2(bufflength),
		blength: bufflength,
	}
}
func (w *WindowHandler) UpdateWindow(img image.Image, format string, a ...interface{}) {

	go func() {
		var buf int
		select {
		case buf = <-w.img.Buffer():
		default:
			buf = 0
		}
		if buf < w.blength {
			w.img.Image(img)
		}
	}()

	go func() {
		var buf int
		select {
		case buf = <-w.para.Buffer():
		default:
			buf = 0
		}
		if buf < w.blength {
			w.para.Paragraph(format, a...)
		}

	}()

}
