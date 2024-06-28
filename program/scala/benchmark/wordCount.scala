package wordcount

import org.apache.spark.lineage.LineageContext
import org.apache.spark.lineage.LineageContext._
import org.apache.spark.{SparkConf, SparkContext}

object wordCount {
  def main(args: Array[String]) {
    val conf = new SparkConf()
    var lineage = true
    var logFile = "hdfs://scai01.cs.ucla.edu:9000/clash/datasets/WB/"
    if (args.size < 2) {
      logFile = "src/IncorrectFromModel/WordCount/new_incorrect_dataset_5000.csv"
      conf.setMaster("local[1]")
      lineage = true
    } else {
      lineage = args(0).toBoolean
      logFile += args(1)
      conf.setMaster("spark://SCAI01.CS.UCLA.EDU:7077")
    }
    conf.setAppName("WordCount-" + lineage + "-" + logFile)

    val sc = new SparkContext(conf)
    val lc = new LineageContext(sc)


    lc.setCaptureLineage(true)

    // Job
    val file = lc.textFile(logFile)

    val pairs = file.map(_.split("\\s"))
      .flatMap(s => s)
      .filter(s => wordCount.wrongWord(s))
      .map { s => (s, 1) }


    pairs.reduceByKey((a, b) => a + b)
      .take(10)
      .foreach(println)

    pairs.collect.foreach(println)

    lc.setCaptureLineage(false)

    sc.stop()
  }

  def wrongWord(word: String): Boolean = {
    word == "sentence"
  }
}