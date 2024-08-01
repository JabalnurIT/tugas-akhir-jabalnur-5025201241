var linRdd = mapped2.getLineage()
linRdd = linRdd.goBackAll()
linRdd.show.saveAsTextFile("src/output/faulty-data")