package deliveryFaults

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.lineage.LineageContext

object deliveryFaults {
  def main(args: Array[String]) {
    val conf = new SparkConf()
    var lineage = true
    var logFile = "hdfs://scai01.cs.ucla.edu:9000/clash/datasets/WB/"
    if (args.size < 2) {
      logFile = "src/IncorrectFromModel/DeliveryFaults/new_incorrect_dataset_780.csv"
      conf.setMaster("local[1]")
      lineage = true
    } else {
      lineage = args(0).toBoolean
      logFile += args(1)
      conf.setMaster("spark://SCAI01.CS.UCLA.EDU:7077")
    }
    conf.setAppName("DeliveryFaults-" + lineage + "-" + logFile)

    val sc = new SparkContext(conf)
    val lc = new LineageContext(sc)


    lc.setCaptureLineage(true)

    val deliveries = lc.textFile(logFile)
      .map(_.split(','))
      .filter(r => deliveryFaults.failure(r(3).toInt))
      .map(r => (r(0), (r(1), r(2), r(3).toFloat)))

    val same_deliveries = deliveries.groupByKey()
    val triplets = same_deliveries.filter(_._2.size > 2)
    val bad_triplets = triplets.filter(tup => tripletRating(tup) < 2.0f)
    bad_triplets
      .map {
        case (_, iter) =>
          iter.foldLeft("")({
            case (acc, (_, vendor, _)) =>
              s"$acc,$vendor"
          })
      }

    deliveries.collect.foreach(println)

    lc.setCaptureLineage(false)

    sc.stop()
  }

  def tripletRating (tup: (String, Iterable[(String, String, Float)] ) ): Float = {
    val (_, iter) = tup
    iter.foldLeft (0.0f) {
      case (acc, (_, _, rating) ) => rating + acc
    } / iter.size
  }

  def failure (rating: Int): Boolean = {
    rating > 5
  }

}