package ui

func webTemp() string {
	return thewebpagetemplatest2
}

const thewebpagetemplatest2 = `<html lang="en">
<head>
<title>GoCuNets Output</title>
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
* {
	box-sizing: border-box;
}
img {
	width: 100%;
	height: auto;
}
body {
  margin: 0;
}

/* Style the header */
.header {
	background-color: #f1f1f1;
	padding: 20px;
	text-align: center;
}

/* Style the top navigation bar */
.topnav {
	overflow: hidden;
	background-color: #333;
}

/* Style the topnav links */
.topnav a {
	float: left;
	display: block;
	color: #f2f2f2;
	text-align: center;
	padding: 14px 16px;
	text-decoration: none;
}

/* Change color on hover */
.topnav a:hover {
	background-color: #ddd;
	color: black;
}

/* Create three equal columns that floats next to each other */
.column {
	float: left;
	width: 25%;
	padding: 4px;
}

/* Clear floats after the columns */
.row:after {
	content: "";
	display: table;
	clear: both;
}

/* Responsive layout - makes the three columns stack on top of each other instead of next to each other */
@media screen and (max-width:600px) {
	.column {
		width: 100%;
	}
}
</style>
<script>

</script>
<body>


<div class="header">
  <h1>On Demand Layout</h1>
  <p>Too Much Stuff</p>
</div>

<div class="topnav">
  <a href="#">Link</a>
  <a href="#">Link</a>
  <a href="#">Link</a>
  <a href="#">Link</a>
  <a href="#">Link</a>
  <a href="#">Link</a>
</div>
{{range .}}
{{ .NewRow}}
  <div class="column">
  	<h2>{{.Header}}</h2>
		<img id="{{.ID}}" src="{{.URL}}">
		<p>{{.Paragraph}}</p>
		<script>
			var {{.MyVar}} = setInterval({{.Func}},{{.Rate}})
		function {{.Func}}(){
			var d = new Date();
			var theurl = "{{.URL}}"+d.toLocaleTimeString();
			 document.getElementById("{{.ID}}").src= theurl;
		}
        </script>
</div>
{{ .EndRow}}
{{end}}


	
	
</body>
</html>
`
