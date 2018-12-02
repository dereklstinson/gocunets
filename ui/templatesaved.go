package ui

func webTemp() string {
	return thewebpagetemplatest2
}

const thewebpagetemplatest2 = `<html lang="en">
<head>
<script src="https://ajax.aspnetcdn.com/ajax/jQuery/jquery-3.3.1.min.js">
</script>
<title>GoCuNets Output</title>
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
* {
	box-sizing: border-box;
}
.zoom {
    padding: 0px;
    transition: transform .2s;
    width: auto;
    height: auto;
    margin: 0 auto;
}

.zoom:hover {
    -ms-transform: scale(2); /* IE 9 */
    -webkit-transform: scale(2); /* Safari 3-8 */
    transform: scale(2); 
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
.column16_66 {
	float: left;
	width: 16.66%;
	padding: 4px;
}
.column20 {
	float: left;
	width: 20%;
	padding: 4px;
}
.column25 {
	float: left;
	width: 25%;
	padding: 4px;
}
.column33 {
	float: left;
	width: 33.33%;
	padding: 4px;
}
.column50 {
	float: left;
	width 50%;
	padding: 4px;
}
.column100 {
	float: left;
	width 100%;
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

<body>
<script>



</script>

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

<h1>HardWare Info</h1> 
{{range .HardwareCharts}}
{{ .NewRow}}
  <div class="{{.ColWid}}">
	  <h2>{{.Header}}</h2>
	   <div><img id="{{.ID}}" src="{{.URL}}"></div>
		<script>
			var {{.MyVar}} = setInterval({{.Func}},{{.Rate}});
		function {{.Func}}(){
			var d = new Date();
			var theurl = "{{.URL}}"+d.toLocaleTimeString();
			document.getElementById("{{.ID}}").src= theurl;
			
		
		}
		</script>

</div>
{{ .EndRow}}
{{end}}


<h1>Network Input Output Info</h1>
{{range .DivOutputs}}
{{ .NewRow}}
  <div class="{{.ColWid}}">
	  <h2>{{.Header}}</h2>
	   <div><img id="{{.ID}}" src="{{.URL}}"></div>
		<div id = "{{.PID}}" src="{{.PURL}}"></div>
		<script>
			var {{.MyVar}} = setInterval({{.Func}},{{.Rate}});
		function {{.Func}}(){
			var d = new Date();
			var theurl = "{{.URL}}"+d.toLocaleTimeString();
			document.getElementById("{{.ID}}").src= theurl;
			var thepurl = "{{.PURL}}"+d.toLocaleTimeString();
			 $.get(thepurl,function(data,status){
				 if (status=="success"){
					document.getElementById("{{.PID}}").innerHTML= data;
				 }
				
			 });
		
		}
		</script>

</div>
{{ .EndRow}}
{{end}}

<h1>Network Stats</h1>
{{range .MinMaxes}}
{{ .NewRow}}
  <div class="{{.ColWid}}">
	  <h2>{{.Header}}</h2>
		<div id = "{{.PID}}" src="{{.PURL}}"></div>
		<script>
			var {{.MyVar}} = setInterval({{.Func}},{{.Rate}});
		function {{.Func}}(){
			var d = new Date();
			var thepurl = "{{.PURL}}"+d.toLocaleTimeString();
		
		
			 $.get(thepurl,function(data,status){
				 if (status=="success"){
					document.getElementById("{{.PID}}").innerHTML= data;
				 }
				
			 });
		
		}
		</script>

</div>
{{ .EndRow}}
{{end}}

	
	
</body>
</html>
`
