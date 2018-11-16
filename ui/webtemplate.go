package ui

func thewebpagetemplate() string {
	return thewebpagetemplatest
}

const thewebpagetemplatest = `<html lang="en">
<head>
<title>GoCuNets Output</title>
<meta charset="utf-8">
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
    <h2>{{.LossHeader}}</h2>
    <img src="data:image/jpg;base64,{{.LSrc}}">
    <p>{{.LossInfo}}</p>
  </div>
  <div class="column">
    <h2>{{.Name}}</h2>
    <img src="data:image/jpg;base64,{{.Src}}">
    <p>{{.Info}}</p>
  </div>
  <div class="column">
    <h2>{{.Name1}}</h2>
    <img src="data:image/jpg;base64,{{.Src1}}">
    <p>{{.Info1}}</p>
  </div>
</div>

</body>
</html>
`

//<img src="{{.LSrc}}" width ="{{.LWidth}}" height= "{{.LHeight}}">
// <img src="{{.Src}}" width ="{{.Width}}" height= "{{.Height}}">
// <img src="{{.Src2}}" width ="{{.Width1}}" height= "{{.Height1}}">
