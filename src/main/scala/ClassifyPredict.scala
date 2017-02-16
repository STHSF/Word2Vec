import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}
import utils.{AnsjAnalyzer, JSONUtil}

/**
  * Created by li on 2016/10/17.
  * 分类预测模型程序主体，与ClassifyModel相对应。
  */

object ClassifyPredict {

  def predict: Unit ={

  }


  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("textVectors").setMaster("local")
    val sc = new SparkContext(conf)

    // load word2vec model
    val word2vecModelPath = JSONUtil.getValue("w2v", "w2vmodelPath")
    val modelSize = JSONUtil.getValue("w2v", "w2vmodelSize").toInt
    val isModel = JSONUtil.getValue("w2v", "isModel").toBoolean
    val w2vModel = Word2VecModel.load(sc, word2vecModelPath)

    // load classify model
    val classifyModelPath = JSONUtil.getValue("classifyw2v", "modelParentPath_SVM")
    //    val classifyModelPath = "/home/zhangxin/work/workplace_scala/Data/classificaiton_model/32_正式可用_SVM/保险.model"
    val classifyModel = SVMModel.load(sc, classifyModelPath)

    // stopWords load
    //    val stopWordsPath = JSONUtil.getValue("kunyan", "stopwords")
    val stopWordsPath = "/Users/li/workshop/DataSet/stop_words_CN"
    val stopWords = sc.textFile(stopWordsPath).collect()

    // 构建测试集labeledpoint格式
    // val predictSetPath = "/Users/li/workshop/DataSet/trainingSets/test"
    val predictSetPath = "/Users/li/workshop/DataSet/test/"

    val predictSet = sc.wholeTextFiles(predictSetPath).collect()

    /** 对测试数据集使用训练模型进行分类预测 */
    val prediction = predictSet.map{row =>

      val startTime = System.currentTimeMillis()

      // 去停 去标点等
      val stopWordsRemoved = DataPrepare.removeStopWords(AnsjAnalyzer.cutTag(row._2, 0), stopWords)

      // textRank word2vec
      val point = TextVectors.singleTextVectorsWithWeight(stopWordsRemoved, w2vModel, modelSize, isModel)

      // predict model
      val predictionFeature = classifyModel.predict(Vectors.dense(point))

      val stopTime = System.currentTimeMillis() - startTime
      println(s"""耗时: $stopTime""")
      println(predictionFeature)
    }

    // classifyModel.clearThreshold()

    sc.stop()
  }
}
