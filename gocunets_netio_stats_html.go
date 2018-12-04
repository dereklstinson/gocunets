package gocunets

import (
	"fmt"
	"html/template"

	"github.com/dereklstinson/GoCuNets/ui/uihelper"
)

func MakeHTMLTemplates(input []*LayerIOStats) []template.HTML {
	section := make([]template.HTML, len(input))
	for i := range input {
		section[i] = input[i].makedivsection()
	}
	return section
}
func (l *LayerIOStats) makedivsection() template.HTML {
	section := make([]template.HTML, 0)
	section = append(section, uihelper.Header("3", l.Name))
	section = append(section, uihelper.Header("4", l.Weights.Name))
	section = append(section,
		uihelper.Paragraphsusingbreaks(
			fmt.Sprintf("%s=%f", "Min", l.Weights.Min),
			fmt.Sprintf("%s=%f", "Max", l.Weights.Max),
			fmt.Sprintf("%s=%f", "Avg", l.Weights.Avg),
			fmt.Sprintf("%s=%f", "Norm1", l.Weights.Norm1),
			fmt.Sprintf("%s=%f", "Norm2", l.Weights.Norm2),
		))
	if !l.IO {

		section = append(section, uihelper.Header("4", l.Bias.Name))
		section = append(section,
			uihelper.Paragraphsusingbreaks(
				fmt.Sprintf("%s=%f", "Min", l.Bias.Min),
				fmt.Sprintf("%s=%f", "Max", l.Bias.Max),
				fmt.Sprintf("%s=%f", "Avg", l.Bias.Avg),
				fmt.Sprintf("%s=%f", "Norm1", l.Bias.Norm1),
				fmt.Sprintf("%s=%f", "Norm2", l.Bias.Norm2),
			))
	}
	return uihelper.DivRegular(combine(section))
}

func combine(input []template.HTML) (section template.HTML) {
	for i := range input {
		section = section + input[i]
	}
	return section
}
