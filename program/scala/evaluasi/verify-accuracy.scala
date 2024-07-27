val resultFISUM = fileToList(s"$basePath/$appName/input/incorrect_data.csv")
val resultFISUMFromTitian = fileToList(s"$basePath/$appName/input/incorrect_data_from_titian.csv")

val result = Array.ofDim[String](resultFISUM.length, 2)

resultFISUM.foreach({row =>
    if (resultFISUMFromTitian.contains(row)) {
        result(resultFISUM.indexOf(row))(0) = row
        result(resultFISUM.indexOf(row))(1) = "faulty"
    } else {
        result(resultFISUM.indexOf(row))(0) = row
        result(resultFISUM.indexOf(row))(1) = "not faulty"
    }
})

result.foreach(println)