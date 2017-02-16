import java.io.File

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.classification.{SVMWithSGD, SVMModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import utils.{DirectoryUtil, JSONUtil}

/**
  * Created by li on 2016/10/13.
  * 分类模型训练程序。
  *
  */
object ClassifyModel {

  /**
    * 准确度统计分析
    *
    * @param predictionAndLabel
    */
  def acc(predictionAndLabel: RDD[(Double, Double)],
          predictDataRdd: RDD[LabeledPoint]): Unit = {

    //统计分类准确率
    val testAccuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / predictDataRdd.count()
    println("testAccuracy：" + testAccuracy)

    val metrics = new MulticlassMetrics(predictionAndLabel)
    println("Confusion matrix:" + metrics.confusionMatrix)

    // Precision by label
    val label = metrics.labels
    label.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }

    // Recall by label
    label.foreach { l =>
      println(s"Recall($l) = " + metrics.recall(l))
    }

    // False positive rate by label
    label.foreach { l =>
      println(s"FPR($l) = " + metrics.falsePositiveRate(l))
    }

    // F-measure by label
    label.foreach { l =>
      println(s"F1-Score($l) = " + metrics.fMeasure(l))
    }

    // val roc = metrics.roc

    // // AUROC
    // val auROC = metrics.areaUnderROC
    // println("Area under ROC = " + auROC)

  }

  // 交叉验证
  def crossDef: Unit = {

  }

  /**
    * 分类模型
    * @param trainDataRdd
    * @return
    */
  def classify(trainDataRdd: RDD[LabeledPoint]): SVMModel = {

    /** NativeBayes训练模型 */
    //  val model = NaiveBayes.train(trainDataRdd, lambda = 1.0, modelType = "multinomial")

    /** SVM训练模型 */
    val numIterations = 1000
    val model = SVMWithSGD.train(trainDataRdd , numIterations)

    /** RandomForest训练模型 */
    //    val numClasses = 2
    //    val categoricalFeaturesInfo = Map[Int, Int]()
    //    val numTrees = 3
    //    val featureSubsetStrategy = "auto"
    //    val impurity = "gini"
    //    val maxDepth = 4
    //    val maxBins = 32
    //    val model = RandomForest.trainClassifier(trainDataRdd, numClasses, categoricalFeaturesInfo,numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    /** GradientBoostedTrees训练模型 */
    //    // Train a GradientBoostedTrees model.
    //    // The defaultParams for Classification use LogLoss by default.
    //    val boostingStrategy = BoostingStrategy.defaultParams("Classification")
    //    boostingStrategy.numIterations = 3 // Note: Use more iterations in practice.
    //    boostingStrategy.treeStrategy.numClasses = 2
    //    boostingStrategy.treeStrategy.maxDepth = 5
    //    // Empty categoricalFeaturesInfo indicates all features are continuous.
    //    boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
    //
    //    val model = GradientBoostedTrees.train(trainDataRdd, boostingStrategy)

    model

    }

  def main(args: Array[String]) {

    val conf = new SparkConf().setAppName("textVectors").setMaster("local")
    val sc = new SparkContext(conf)

    val word2vecModelPath = JSONUtil.getValue("w2v", "w2vmodelPath")
    //向量的长度
    val vectorSize = JSONUtil.getValue("w2v", "w2vmodelSize").toInt
    //是否是模型
    val isModel = JSONUtil.getValue("w2v", "isModel").toBoolean
    //导入word2vec模型
    val w2vModel = Word2VecModel.load(sc, word2vecModelPath)

    // 构建训练集的labeledpoint格式
    // val trainSetPath = "/Users/li/workshop/DataSet/trainingsetUnbalance/BXX.txt"
     val trainSetPath = "/Users/li/workshop/DataSet/trainingSets/计算机"
    // val trainSetPath = "/Users/li/workshop/DataSet/trainingSets/机械"

    val trainSet = DataPrepare.readData(trainSetPath)
      .map { row =>
        val temp = row.split("\t")
        (temp(0).toDouble, temp(1).split(","))
      }

    val trainData = trainSet.map{row =>{
      TextVectors.textVectorsWithWeight(row, w2vModel, vectorSize, isModel)
    }}

    val trainDataRdd = sc.parallelize(trainData)
    val classifyModel = classify(trainDataRdd)

    val classifyModelPath = JSONUtil.getValue("classifyw2v", "modelParentPath_SVM")
    DirectoryUtil.deleteDir(new File(classifyModelPath))
    classifyModel.save(sc, classifyModelPath)

    // 准确度统计分析
    //predictionAndLabel.foreach(println)
    println("==分类模型保存完毕==")

    sc.stop()
  }
}
