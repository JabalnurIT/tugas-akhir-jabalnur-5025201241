package commuteTypeFull
import org.apache.spark.lineage.LineageContext
import org.apache.spark.{SparkConf, SparkContext}

object commuteTypeFull {
  def main(args: Array[String]) {
    val conf = new SparkConf()
    var lineage = true
    var logFile1 = "hdfs://scai01.cs.ucla.edu:9000/clash/datasets/WB/"
    var logFile2 = "hdfs://scai01.cs.ucla.edu:9000/clash/datasets/WB/"
    if (args.size < 2) {
      logFile1 = "src/resources/dataCommuteTypeFull/trips"
      logFile2 = "src/IncorrectFromModel/Customers/new_incorrect_dataset_11.csv"
      conf.setMaster("local[1]")
      lineage = true
    } else {
      lineage = args(0).toBoolean
      logFile1 += args(1)
      logFile2 += args(2)
      conf.setMaster("spark://SCAI01.CS.UCLA.EDU:7077")
    }
    conf.setAppName("CommuteTypeFull-" + lineage)

    val sc = new SparkContext(conf)
    val lc = new LineageContext(sc)

    lc.setCaptureLineage(true)

    // Job
    val trips = lc.textFile(logFile1).map(s => s.split(","))
      .map { cols =>
        (cols(1), cols(3).toInt / cols(4).toInt)
      }

    val locations = lc.textFile(logFile2).map(s => s.filter(_ != '\"').split(","))
      .map { cols =>
        (cols(0).toInt, cols(3))
      }
      .filter {
      s => commuteTypeFull.wrongInput(s._1)
    }

    locations.collect().foreach(println)

    lc.setCaptureLineage(false)

    sc.stop()
  }

  def wrongInput(id: Int): Boolean = {
    id <= 1000
  }

  def add(a: Int, b: Int): Int = {
    a + b
  }

}