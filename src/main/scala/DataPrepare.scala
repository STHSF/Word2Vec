import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.{SparkConf, SparkContext}
import utils.AnsjAnalyzer

import scala.io.Source

/**
  * Created by li on 2016/10/14.
  */
object DataPrepare {

  /**
    * 读文件
    *
    * @param filePath 文本保存的位置
    * @return
    */
  def readData(filePath: String): Array[String] = {

    val doc = Source.fromFile(filePath).getLines().toArray

    doc
  }

  /**
    * 分词
    *
    * @param doc
    * @return
    */
  def docCut(doc: (Double, String)): (Double, Array[String]) = {

    val docSeg = AnsjAnalyzer.cutTag(doc._2, 0)

    (doc._1, docSeg)
  }

  /**
    * 创建带标签的文本
    *
    * @param label 标签
    * @param docSeg 分词结果
    * @return
    */
  def singleLabeledDoc(label: Double, docSeg: Array[String]): String = {

    val docSegString = docSeg.map(row => row.replace("　","")).mkString(",")

    label + "\t" +  docSegString
  }

  /**
    * 去除分词结果中的标点符号和停用词
    *
    * @param content 分词结果
    * @param stopWords 停用词
    * @return 返回一个元素为String的Array
    */
  def removeStopWords(content: Array[String], stopWords:Array[String]): Array[String] = {

    if (content != null) {

      var result = content.toBuffer
      stopWords.foreach(stopWord => {

        if (result.contains(stopWord)){
          result = result.filterNot(_ == stopWord)
        }

      })

      result.toArray
    } else {
      null
    }
  }

  /**
    * 测试代码
    */
  def dataPrepareTest(): Unit = {

    val conf = new SparkConf().setMaster("local").setAppName("DataPrepare")
    val sc = new SparkContext(conf)

    val filePath = "/Users/li/workshop/DataSet/111.txt"

    val word2vecModelPath = "/Users/li/workshop/DataSet/word2vec/result/2016-07-18-15-word2VectorModel"
    val model = Word2VecModel.load(sc, word2vecModelPath)
//
//    val data = readData(filePath)
//
//    val splitData = docCut((1.0, data))
//
//    val labeledP = singleLabeledDoc(1.0, splitData)
//
//    val tmp = labeledP.split("\t")
//
//    println(tmp(1).split(","))
  }


  def main(args: Array[String]) {

    dataPrepareTest()

  }

}
