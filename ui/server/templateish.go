package server

import (
	"fmt"
	"html/template"
)

func Replacecomment(comment string) template.HTML {
	return template.HTML(fmt.Sprintf("<!---%s--->", comment))
}
func Canvas(id string) template.HTML {
	return template.HTML(fmt.Sprintf("<canvas id=\"%s\"></canvas>\\n", id))
}
func Image(id, src string) template.HTML {
	return template.HTML(fmt.Sprintf("<img id=\"%s\" src=\"%s\"></img>\\n", id, src))
}
func Divurl(id, src string) template.HTML {
	return template.HTML(fmt.Sprintf("<div id=\"%s\" src=\"%s\"></div>\\n", id, src))
}
func Divcolumnwrap(class, innerHTML string) template.HTML {
	return template.HTML(fmt.Sprintf("<div class=\"%s\">\\n%s</div>\\n", class, innerHTML))
}
func Jquerygetjson(url, callback string) template.JS {
	return template.JS(fmt.Sprintf("$.getJSON(\"%s\",%s)", url, callback))
}
func Header2(innerHTML string) template.HTML {
	return template.HTML(fmt.Sprintf("	<h2>%s</h2>", innerHTML))
}
