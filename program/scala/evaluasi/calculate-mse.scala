var result = Array(0,0,0,0)
val resultData = fileToList(s"$basePath/$appName/input/incorrect_data.csv")
resultData.foreach({row =>
    val array = row.split(",")
    if (array(0) == "null" && array(1).toInt > 50 ){
        row :+ "mid"
        result(1) += 1
    } else if(array(0) == "null"){
        row :+"left"
        result(0) += 1
    }else if (array(1).toInt >50){
        row :+ "right"
        result(2) += 1
    }
})

result = result.map(_ / result.min.toDouble)


var resultFisum = Array(0,0,0,0)
val fisumData = fileToList(s"$basePath/$appName/output/backup2/new_incorrect_dataset_2953.csv")
fisumData.foreach({row =>
    val array = row.split(",")
    if (array(0) == "null" && array(1).toInt > 50 ){
        row :+ "mid"
        resultFisum(1) += 1
    } else if(array(0) == "null"){
        row :+"left"
        resultFisum(0) += 1
    }else if (array(1).toInt >50){
        row :+ "right"
        resultFisum(2) += 1
    }
})

resultFisum = resultFisum.map(_ / resultFisum.min.toDouble)

val differences = result.zip(resultFisum).map { case (a, b) => pow(a - b, 2) }
    
val MSE = differences.sum / differences.length

println("MSE: "+MSE)
  