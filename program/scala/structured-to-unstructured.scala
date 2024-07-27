val directory = new File(dir)
var data = (directory.exists() && directory.isDirectory) ?
    directory.listFiles()
    .filter(_.isFile)
    .map(_.getName)
    .toList
    : List.empty[String]

data = data.mkString("###")