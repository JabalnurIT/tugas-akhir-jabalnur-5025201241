import scala.language.postfixOps
import akka.actor.{ActorRef, ActorSystem, Props}
import akka.event.slf4j.Logger
import akka.http.javadsl.TimeoutAccess
import akka.http.scaladsl.Http
import akka.http.scaladsl.TimeoutAccess
import akka.http.scaladsl.settings.{ClientConnectionSettings, ConnectionPoolSettings}
import com.typesafe.config.ConfigFactory
import spray.json.DefaultJsonProtocol.{IntJsonFormat, StringJsonFormat, mapFormat}
import spray.json.{JsString, JsValue, JsonParser, enrichAny}

import java.io.PrintWriter
import scala.math.{ceil, pow}
import java.io.File
import akka.http.scaladsl.model.{ContentTypes, HttpEntity, HttpMethods, HttpRequest}

import akka.http.scaladsl.unmarshalling.Unmarshal
import scala.concurrent.Await
import scala.concurrent.duration._

import scala.io.Source

import java.time.LocalDateTime
import java.time.{Duration => DurationTime}

object Main {

  private val format = "yyyyMMdd_HHmmss"

  private val appName = "main_app"
  private val logger = Logger(appName)



  implicit val system: ActorSystem = ActorSystem("system")
  import system.dispatcher


  val settings: ConnectionPoolSettings = ConnectionPoolSettings(system)
    .withConnectionSettings(ClientConnectionSettings(system)
      .withIdleTimeout(Duration.Inf))
  //  implicit val baseURL: String = "http://18.136.194.164:8000"
  implicit val baseURL: String = "http://127.0.0.1:8000"


  implicit val basePath: String = "src/assets"

  private def makeDirsFromList(dir: String, dirs: List[String]): Unit = {
    dirs.foreach { d =>
      val newDir = new File(dir + "/" + d)
      newDir.mkdir()
    }
  }

  private def readAllFilesInDir(dir: String): List[String] = {
    val directory = new File(dir)
    if (directory.exists() && directory.isDirectory) {
      directory.listFiles()
        .filter(_.isFile)
        .map(_.getName)
        .toList
    } else {
      List.empty[String]
    }
  }

  private def readAllDirsInDir(dir: String): List[String] = {
    val directory = new File(dir)
    if (directory.exists() && directory.isDirectory) {
      directory.listFiles()
        .filter(_.isDirectory)
        .map(_.getName)
        .toList
    } else {
      List.empty[String]
    }
  }

  private def fileToList(filePath: String): List[String] = {
    val source = Source.fromFile(filePath)
    var listDataset = List[String]()
    try {
      listDataset = source.getLines().toList
    } catch {
      case e: Exception =>
        logger.error(s"${e.getMessage}")
    }
    finally {
      source.close()
    }

    listDataset
  }

  private def listToFile(filePath: String, list: List[String]): Unit = {
    val writer = new PrintWriter(filePath)
    try {
      list.foreach(item => writer.println(item))
      logger.info("Data saved to CSV file successfully.")
    } catch  {
      case e: Exception =>
        logger.error(s"${e.getMessage}")
    }
    finally {
      writer.close()
    }
  }

  private def getModelListRequest(): (String, List[String]) = {
    val requestModelList = HttpRequest(
      method = HttpMethods.POST,
      uri = s"$baseURL/model-list"
    )
    val futureRes = for {
      resp <- Http().singleRequest(requestModelList, settings = settings)
      res <- Unmarshal(resp.entity).to[String]
    } yield res

    val res = Await.result(futureRes, Duration.Inf)
    val jsonRes: JsValue = JsonParser(res)

    val currentModel = jsonRes.asJsObject
      .fields("current")
      .toString()
      .replace("\"","")
    val modelList = jsonRes.asJsObject
      .fields("models")
      .toString()
      .replaceAll("\\[|\\]|\"","")
      .split(",")
      .toList

    (currentModel, modelList)
  }

  private def useModelRequest(model: String): String = {
    val requestModelList = HttpRequest(
      method = HttpMethods.POST,
      uri = s"$baseURL/use-model",
      entity = HttpEntity(
        ContentTypes.`application/json`,
        Map("model" -> model).toJson.toString()
      )
    )
    val futureRes = for {
      resp <- Http().singleRequest(requestModelList, settings = settings)
      res <- Unmarshal(resp.entity).to[String]
    } yield res

    val res = Await.result(futureRes, Duration.Inf)
    val jsonRes: JsValue = JsonParser(res)

    val status = jsonRes.asJsObject
      .fields("status")
      .toString()
      .replace("\"", "")

    status
  }

  private def retrainRequest(text: String): String = {
    val requestRetrain = HttpRequest(
      method = HttpMethods.POST,
      uri = s"$baseURL/retrain",
      entity = HttpEntity(
        ContentTypes.`application/json`,
        Map("texts" -> text).toJson.toString()
      )
    )
    val futureRes = for {
      resp <- Http().singleRequest(requestRetrain, settings = settings)
      res <- Unmarshal(resp.entity).to[String]
    } yield res

    val res = Await.result(futureRes, Duration.Inf)
    val jsonRes: JsValue = JsonParser(res)

    val status = jsonRes.asJsObject
      .fields("status")
      .toString()
      .replace("\"", "")

    status
  }
  private def generateInputRequest(count: Int): List[String] = {
    val requestGenerateInput = HttpRequest(
      method = HttpMethods.POST,
      uri = s"$baseURL/generate",
      entity = HttpEntity(
        ContentTypes.`application/json`,
        Map("num_text" -> count).toJson.toString()
      )
    )
    val futureRes = for {
      resp <- Http().singleRequest(requestGenerateInput, settings = settings)
      res <- Unmarshal(resp.entity).to[String]
    } yield res

    val res = Await.result(futureRes, Duration.Inf)
    val jsonRes: JsValue = JsonParser(res)

    val dataset = jsonRes.asJsObject
      .fields("dataset")
      .toString()
      .replace("\"", "")

    val listDataset = dataset.split("###").toList
    listDataset
  }

  private def resetModelRequest(): String = {
    val requestRetrain = HttpRequest(
      method = HttpMethods.POST,
      uri = s"$baseURL/reset",
    )
    val futureRes = for {
      resp <- Http().singleRequest(requestRetrain, settings = settings)
      res <- Unmarshal(resp.entity).to[String]
    } yield res

    val res = Await.result(futureRes, Duration.Inf)
    val jsonRes: JsValue = JsonParser(res)

    val status = jsonRes.asJsObject
      .fields("status")
      .toString()
      .replace("\"", "")

    status
  }

  private def deleteModelRequest(): String = {
    val requestRetrain = HttpRequest(
      method = HttpMethods.POST,
      uri = s"$baseURL/delete",
    )
    val futureRes = for {
      resp <- Http().singleRequest(requestRetrain, settings = settings)
      res <- Unmarshal(resp.entity).to[String]
    } yield res

    val res = Await.result(futureRes, Duration.Inf)
    val jsonRes: JsValue = JsonParser(res)

    val status = jsonRes.asJsObject
      .fields("status")
      .toString()
      .replace("\"", "")

    status
  }

  def main(args: Array[String]): Unit = {

    val percentageOutput = List(1,2,5,10,25,50)

    appName = "app_name"
    logger.info(appName)
    var description = List[String]()
    if(readAllFilesInDir(s"src/assets/$appName/output/").sorted == List[String]()){
    val useModelStatus = useModelRequest(appName)
    val (currentModel, modelList) = getModelListRequest()
    logger.info(s"current model: $currentModel")
    logger.info(s"current model: $modelList")
    if (useModelStatus != "Success"){
        logger.error(useModelStatus)
        system.terminate()
    }
    println(s"use model status: $useModelStatus")

    val allData = fileToList(s"$basePath/$appName/input/incorrect_data.csv")
    println(s"Total Data: ${allData.length}")
    for (i <- 0 until math.min(5, allData.length)) {
        println(allData(i))
    }
    println("-- Training --")
    description = description :+ "retrain time"
    var startTime = LocalDateTime.now()
    val retrainStatus = retrainRequest(allData.mkString("###"))
    if (retrainStatus != "Success") {
        logger.error(retrainStatus)
        system.terminate()
    }
    var stopTime = LocalDateTime.now()
    val retrainTime = DurationTime.between(startTime,stopTime)
    description = description :+ retrainTime.toString

    println(s"retrain status: $retrainStatus")

    for (i <- 0 until 6) {
        val numRow: Int = ceil(percentageOutput(i).toDouble / 100 * allData.length).toInt
        println(s"Num of row: $numRow")
        startTime = LocalDateTime.now()
        val newDataset = generateInputRequest(numRow)
        stopTime = LocalDateTime.now()
        val generateInputTime = DurationTime.between(startTime,stopTime)
        description = description :+ s"generated time of $numRow row"
        description = description :+ generateInputTime.toString
        newDataset.foreach(println)
        listToFile(s"$basePath/$appName/output/new_incorrect_dataset_$numRow.csv",newDataset)
        println()
    }
    listToFile(s"$basePath/$appName/output/description.csv",description)


    val resetModelStatus = resetModelRequest()
    if (resetModelStatus != "Success") {
        logger.error(resetModelStatus)
        system.terminate()
    }
    println(s"reset model status: $resetModelStatus")

    val deleteModelStatus = deleteModelRequest()
    if (deleteModelStatus != "Success") {
        logger.error(deleteModelStatus)
        system.terminate()
    }
    println(s"delete model status: $deleteModelStatus")
    }

    system.terminate()

  }
}