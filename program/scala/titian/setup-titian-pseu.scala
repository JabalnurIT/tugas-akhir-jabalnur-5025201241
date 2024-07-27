val conf = new SparkConf()
var lineage = true
var logFile = "hdfs://scai01.cs.ucla.edu:9000/clash/datasets/WB/"
if (args.size < 2) {
    logFile = "src/dataset.csv"
    conf.setMaster("local[1]")
    lineage = true
} else {
    lineage = args(0).toBoolean
    logFile += args(1)
    conf.setMaster("spark://SCAI01.CS.UCLA.EDU:7077")
}

val sc = new SparkContext(conf)
val lc = new LineageContext(sc)