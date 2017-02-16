import org.apache.spark.{SparkContext, SparkConf}
import utils.AnsjAnalyzer

/**
  * Created by li on 2016/11/18.
  */
object test {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("textVectors").setMaster("local")
    val sc = new SparkContext(conf)

    val stopWordsPath = "/Users/li/workshop/DataSet/stop_words_CN"
    val stopWords = sc.textFile(stopWordsPath).collect()


    val predictSetPath = "/Users/li/workshop/DataSet/111.txt"
    //    val predictSet = DataPrepare.readData(predictSetPath)

    val predictSet = sc.wholeTextFiles(predictSetPath)
      .map { row => {

        DataPrepare.removeStopWords(AnsjAnalyzer.cutTag(row._2, 0), stopWords)
      }}.collect().flatMap(x => x)

    println(predictSet.mkString("|"))

  }

}
