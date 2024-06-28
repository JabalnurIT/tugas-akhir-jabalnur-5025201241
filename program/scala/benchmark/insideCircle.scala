package insideCircle
import org.apache.spark.lineage.LineageContext
import org.apache.spark.{SparkConf, SparkContext}

object insideCircle {

  def main(args: Array[String]) {
    val conf = new SparkConf()
    var lineage = true
    var logFile = "hdfs://scai01.cs.ucla.edu:9000/clash/datasets/WB/"
    if (args.size < 2) {
      logFile = "src/IncorrectFromModel/InsideCircle/new_incorrect_dataset_63.csv"
      conf.setMaster("local[1]")
      lineage = true
    } else {
      lineage = args(0).toBoolean
      logFile += args(1)
      conf.setMaster("spark://SCAI01.CS.UCLA.EDU:7077")
    }
    conf.setAppName("InsideCircle-" + lineage + "-" + logFile)

    val sc = new SparkContext(conf)
    val lc = new LineageContext(sc)


    lc.setCaptureLineage(true)

    // Job
    val sizes = lc.textFile(logFile, 3).map(_.split(","))

    val mapped = sizes.map {
      cols => (cols(0).toInt, cols(1).toInt, cols(2).toInt)
    }

    val mapped2 = mapped.filter(s => insideCircle.failure(s._1, s._2, s._3))
      .map {
        s => (s._1, s._2, s._3)
      }

    println("This is mapped")
    mapped2.collect.foreach(println)

    lc.setCaptureLineage(false)

    sc.stop()
  }

  def failure(x: Int, y: Int, z: Int): Boolean = {
    x * x + y * y >= z * z
  }
}