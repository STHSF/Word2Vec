import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.{SparkConf, SparkContext}
import utils.JSONUtil

/**
  * Created by li on 2016/10/17.
  */
object ClassifyPredict {





  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("textVectors").setMaster("local")
    val sc = new SparkContext(conf)

    val jsonPath = "/Users/li/workshop/NaturalLanguageProcessing/src/main/scala/meachinelearning/word2vec/twc/W2VJsonConf.json"

    JSONUtil.initConfig(jsonPath)

    // load word2vec model
    val word2vecModelPath = JSONUtil.getValue("w2v", "w2vmodelPath")
    val modelSize = JSONUtil.getValue("w2v", "w2vmodelSize").toInt
    val isModel = JSONUtil.getValue("w2v", "isModel").toBoolean
    val w2vModel = Word2VecModel.load(sc, word2vecModelPath)

    // load classify model
    val classifyModelPath = JSONUtil.getValue("classify", "classifymodelpath")
    val classifyModel = SVMModel.load(sc, classifyModelPath)

    // 构建测试集labeledpoint格式
    val predictSetPath = "/Users/li/workshop/DataSet/trainingSets/test"
    // val predictSetPath = "/Users/li/workshop/DataSet/111.txt"


    val predictSet = DataPrepare.readData(predictSetPath)
      .map { row => val temp = row.split("\t")
        (temp(0).toDouble, temp(1).split(","))
      }
    val startTime = System.currentTimeMillis()
    // 对于单篇没有分词的文章
//    val splitData = DataPrepare.docCut((1.0, predictSet))
//    val doVec = DataPrepare.singleLabeledDoc(1.0, splitData)
//    val predictData = TextVectors.singleTextVectorsWithWeight(doVec, w2vModel, modelSize, isModel)

    val predictData = predictSet.map{row => {

      TextVectors.textVectorsWithWeight(row, w2vModel, modelSize, isModel)
    }}

    /** 对测试数据集使用训练模型进行分类预测 */
    // classifyModel.clearThreshold()
    val predictionAndLabel = predictData.map{ point => {
      val predictionFeature = classifyModel.predict(point.features)

      (predictionFeature, point.label)
    }}
    predictionAndLabel.foreach(println)

    val stopTime = System.currentTimeMillis() - startTime
    println(s"耗时: $stopTime")

    sc.stop()
  }
}
