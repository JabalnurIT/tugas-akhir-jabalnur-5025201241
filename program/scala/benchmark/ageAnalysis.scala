package ageAnalysis
import org.apache.spark.lineage.LineageContext
import org.apache.spark.{SparkConf, SparkContext}

object ageAnalysis {
  def main(args: Array[String]) {
    val conf = new SparkConf()
    var lineage = true
    var logFile = "hdfs://scai01.cs.ucla.edu:9000/clash/datasets/WB/"
    if (args.size < 2) {
      logFile = "src/IncorrectFromModel/AgeAnalysis/new_incorrect_dataset_47.csv"
      conf.setMaster("local[1]")
      lineage = true
    } else {
      lineage = args(0).toBoolean
      logFile += args(1)
      conf.setMaster("spark://SCAI01.CS.UCLA.EDU:7077")
    }
    conf.setAppName("AgeAnalysis-" + lineage + "-" + logFile)

    val sc = new SparkContext(conf)
    val lc = new LineageContext(sc)


    lc.setCaptureLineage(true)

    // Job
    val ages = lc.textFile(logFile).map(_.split(","))

    val mapped = ages.map {
      cols => (cols(0), cols(1).toInt, cols(2).toInt)
    }
    val filtered = mapped.filter { s =>
      s._1 == "90024"
    }
    val mapped2 = filtered.filter(s => ageAnalysis.failure(s._2))
    .map {
      s =>
        if (s._2 >= 40 & s._2 <= 65) {
          ("40-65", s._3)
        } else if (s._2 >= 20 & s._2 < 40) {
          ("20-39", s._3)
        } else if (s._2 < 20) {
          ("0-19", s._3)
        } else if (s._2 > 65 & s._2 <= 100) {
          (">65", s._3)
        } else {
          (">100", s._3)
        }
    }

    println("This is mapped")
    mapped2.collect.foreach(println)

    lc.setCaptureLineage(false)

    sc.stop()
  }

  def failure(age: Int): Boolean = {
    age > 50
  }

}