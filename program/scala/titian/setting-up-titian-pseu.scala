val conf = new SparkConf()
var lineage = true

logFile = "src/dataset.csv"
conf.setMaster("local[1]")
lineage = true

val sc = new SparkContext(conf)
val lc = new LineageContext(sc)