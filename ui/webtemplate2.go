package ui

func thewebpagetemplate2() string {
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
	<script>
	  
		//checkout the site below for more info
		//https://developer.mozilla.org/en-US/docs/Learn/JavaScript/Client-side_web_APIs/Fetching_data

		  //  var buttonpress = document.querySelector('button');
		
					
		function updateDisplay(elid,ehh) {
		  document.getElementById(elid).src=ehh+ + new Date().getTime();
	  
		}

		   
</script>
</head>
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

<div class="row">
  <div class="column">
	<h2>Loss</h2>
	<img id="image1" src="">
	<p>This shows loss</p>
	<script>
		//updateDisplay("image1","{{.Url1}}")
		setTimeout( updateDisplay("image1","{{.Url1}}", 1000));
	</script>

   
</div>
  <div class="column">
	<h2>Input Image</h2>
	<img id="image2" src="{{.Url2}}">
	<p>Selve Serving Description</p>
	<button type="button"
	onclick="updateDisplay('image2','{{.Url2}}')">Update</button>
  </div>
  <div class="column">
	<h2>Output Image</h2>
	<img id="image3" src="{{.Url3}}">
	<p>Selve Serving Description</p>
	<button type="button"
	onclick="updateDisplay('image3','{{.Url3}}')">Update</button>
  </div>
</div>

</body>
</html>`
