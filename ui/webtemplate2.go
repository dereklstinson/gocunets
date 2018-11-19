package ui

func thewebpagetemplate2() string {
	return thewebpagetemplatest2
}

const thewebpagetemplatest2 = `
<html lang="en">
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
	width: 33.33%;
	padding: 15px;
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
<div>

</div>
<div class="row">
  <div class="column">
  	<h2>Loss Graph</h2>
		<img id="image1" src="{{.Url1}}">
		<p>This is the loss over epoc.</p>
</div>
  <div class="column">
	<h2>Auto Encoder Input Image</h2>
	<img id="image2" src="{{.Url2}}">
	<p>Input image for autoencoder.</p>
	
  </div>
  <div class="column">
	<h2>Auto Encoder Output Image</h2>
	<img id="image3" src="{{.Url3}}">
	<p>Output image for autoencoder.</p>
	
  </div>
</div>
<div class="row">
		<div class="column">
			<h2>Gan Output</h2>
			<img id="image4" src="{{.Url4}}">
			<p>This is the output from the gan.</p>
	</div>
		<div class="column">
		<h2>Input Image</h2>
		<img id="image5" src="{{.Url5}}">
		<p>Selve Serving Description</p>
		
		</div>
		<div class="column">
		<h2>Output Image</h2>
		<img id="image6" src="{{.Url6}}">
		<p>Selve Serving Description</p>
		
		</div>
	</div>
<script>
	var mywaittime =5000;
		var myVar1 = setInterval(myimage1,mywaittime)
		function myimage1(){
			var d = new Date();
			var theurl = "{{.Url1}}"+d.toLocaleTimeString();
			 document.getElementById("image1").src= theurl;
			}
			var myVar2 = setInterval(myimage2,mywaittime)
		function myimage2(){
			var d = new Date();
			var theurl = "{{.Url2}}"+d.toLocaleTimeString();
			 document.getElementById("image2").src= theurl;
			}
			var myVar3 = setInterval(myimage3,mywaittime)
		function myimage3(){
			var d = new Date();
			var theurl = "{{.Url3}}"+d.toLocaleTimeString();
			 document.getElementById("image3").src= theurl;
			}

			var myVar4 = setInterval(myimage4,mywaittime)
		function myimage4(){
			var d = new Date();
			var theurl = "{{.Url4}}"+d.toLocaleTimeString();
			 document.getElementById("image4").src= theurl;
			}
			var myVar5 = setInterval(myimage5,mywaittime)
		function myimage5(){
			var d = new Date();
			var theurl = "{{.Url5}}"+d.toLocaleTimeString();
			 document.getElementById("image5").src= theurl;
			}
			var myVar6 = setInterval(myimage6,mywaittime)
		function myimage6(){
			var d = new Date();
			var theurl = "{{.Url6}}"+d.toLocaleTimeString();
			 document.getElementById("image6").src= theurl;
			}
			</script>
		</head>
</script>
		</head>
	
</body>
</html>`