package server

import (
	"fmt"
	"html/template"
)

func Replacecomment(comment string) template.HTML {
	return template.HTML(fmt.Sprintf("<!---%s--->", comment))
}
func Canvas(id string) template.HTML {
	return template.HTML(fmt.Sprintf("<canvas id=\"%s\"></canvas>", id))
}
func Image(id, src string) template.HTML {
	return template.HTML(fmt.Sprintf("<img id=\"%s\" src=\"%s\"></img>", id, src))
}
func Divurl(id, src string) template.HTML {
	return template.HTML(fmt.Sprintf("<div id=\"%s\" src=\"%s\"></div>", id, src))
}
func Divcolumnwrap(class string, innerHTML template.HTML) template.HTML {
	return template.HTML(fmt.Sprintf("<div class=\"%s\">%s</div>", class, innerHTML))
}
func Paragraph(innerHTML string) template.HTML {
	return template.HTML(fmt.Sprintf("<p>%s</p>", innerHTML))
}
func ParagraphID(id string, filler string) template.HTML {
	return template.HTML(fmt.Sprintf("<p id=\"%s\">%s</p>", id, filler))
}
func Jquerygetjson(url string, callback template.JS) template.JS {
	return template.JS(fmt.Sprintf("$.getJSON(\"%s\",%s)", url, callback))
}
func Header(num, innerHTML string) template.HTML {
	return template.HTML(fmt.Sprintf("<h%s>%s</h%s>", num, innerHTML, num))
}
