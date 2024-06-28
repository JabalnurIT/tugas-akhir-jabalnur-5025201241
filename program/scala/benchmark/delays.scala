package delays

import org.apache.spark.lineage.LineageContext
import org.apache.spark.{SparkConf, SparkContext}

object delays {
  def main(args: Array[String]) {
    val conf = new SparkConf().setMaster(if (args.length > 2) args(2) else "local[1]")
    conf.setAppName("Bus Delays")
    val sc = SparkContext.getOrCreate(conf)
    val lc = new LineageContext(sc)

    lc.setCaptureLineage(true)

    val station1 = lc.textFile("src/IncorrectFromModel/Delays/new_incorrect_dataset_1254.csv")
      .map(_.split(','))
      .filter(r => delays.failure(r(1).toDouble, r(2).toDouble))
      .map(r => (r(0), (r(1), r(2), r(3))))

    station1.collect.foreach(println)

    lc.setCaptureLineage(false)

    sc.stop()


  }

  def buckets(v: Int): Int = {
    v / 1800 // groups of 30 min delays
  }

  def failure (a: Double, d: Double): Boolean = {
    a < d
  }

}