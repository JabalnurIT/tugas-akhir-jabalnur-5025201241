package customers
import org.apache.spark.lineage.LineageContext
import org.apache.spark.{SparkConf, SparkContext}

object customers {
  def main(args: Array[String]) {
    val conf = new SparkConf()
    var lineage = true
    var customers_data = "hdfs://scai01.cs.ucla.edu:9000/clash/datasets/WB/"
    var orders_data = "hdfs://scai01.cs.ucla.edu:9000/clash/datasets/WB/"
    if (args.size < 2) {
      customers_data = "src/resources/customers/customers"
      orders_data = "src/IncorrectFromModel/Customers/new_incorrect_dataset_522.csv"
      conf.setMaster("local[1]")
      lineage = true
    } else {
      lineage = args(0).toBoolean
      customers_data += args(1)
      orders_data += args(2)
      conf.setMaster("spark://SCAI01.CS.UCLA.EDU:7077")
    }
    conf.setAppName("Customers-" + lineage)

    val sc = new SparkContext(conf)
    val lc = new LineageContext(sc)

    lc.setCaptureLineage(true)

    val customers = lc.textFile(customers_data).map(_.split(","))
    val orders = lc.textFile(orders_data).map(_.split(","))

    val o = orders
      .map {
        case Array(_, cid, date, iid) => (cid, (iid, date.toDouble))
      }
      .filter{
        s=>
          s._1.toLong < 100
      }

    o.collect.foreach(println)

    val c = customers
      .map {
        row =>
          (row(0), row(1))
      }

    lc.setCaptureLineage(false)

  }

}