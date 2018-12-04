package uihelper

import (
	"fmt"
	"html/template"
)

//Replacecomment allows for a comment to be placed
func Replacecomment(comment string) template.HTML {
	return template.HTML(fmt.Sprintf("<!---%s--->", comment))
}

//Canvas makes a canvas
func Canvas(id string) template.HTML {
	return template.HTML(fmt.Sprintf("<canvas id=\"%s\"></canvas>", id))
}

//Image does an image
func Image(id, src string) template.HTML {
	return template.HTML(fmt.Sprintf("<img id=\"%s\" src=\"%s\"></img>", id, src))
}

//DivURL sets up a div with a src and id
func DivURL(id, src string) template.HTML {
	return template.HTML(fmt.Sprintf("<div id=\"%s\" src=\"%s\"></div>", id, src))
}

//Divcolumnwrap wraps up an template.HTML with a div
func Divcolumnwrap(class string, innerHTML template.HTML) template.HTML {
	return template.HTML(fmt.Sprintf("<div class=\"%s\">%s</div>", class, innerHTML))
}

//DivRegular is a div wraper
func DivRegular(innerHTML template.HTML) template.HTML {
	return template.HTML(fmt.Sprintf("<div>%s</div>", innerHTML))
}

//Paragraph wraps a paragraph
func Paragraph(innerHTML string) template.HTML {
	return template.HTML(fmt.Sprintf("<p>%s</p>", innerHTML))
}

//Paragraphsusingbreaks takes a bunch of strings a wraps in in a paragraph separated by line breaks
func Paragraphsusingbreaks(paragraphs ...string) template.HTML {
	paragraphstring := ""
	for i := range paragraphs {
		paragraphstring = paragraphstring + fmt.Sprintf("%s<br>", paragraphs[i])
	}
	return Paragraph(paragraphstring)
}

//ParagraphID passes a paragraph that takes an ID. It also allows for some inner html
func ParagraphID(id string, filler string) template.HTML {
	return template.HTML(fmt.Sprintf("<p id=\"%s\">%s</p>", id, filler))
}

//Jquerygetjson sets up a getjson.
func Jquerygetjson(url string, callback template.JS) template.JS {
	return template.JS(fmt.Sprintf("$.getJSON(\"%s\",%s)", url, callback))
}

//Header allows for a header to be made
func Header(num, innerHTML string) template.HTML {
	return template.HTML(fmt.Sprintf("<h%s>%s</h%s>", num, innerHTML, num))
}
