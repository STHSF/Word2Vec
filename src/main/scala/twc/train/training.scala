package twc.train

import java.io.PrintWriter

import com.kunyandata.nlpsuit.util.JsonConfig
import org.apache.spark.mllib.classification.{NaiveBayes, SVMWithSGD}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangxin on 16-11-9.
  *
  * twc分类模型训练程序。
  */
object training {

  /**
    * 批量训练模型，使用
    */
  def training_SVM(): Unit ={

    val conf = new SparkConf().setAppName("W2V").setMaster("local")
    val sc = new SparkContext(conf)
    val jsonPath = "/home/zhangxin/work/workplace_scala/Sentiment/src/main/scala/classificationW2V/W2VJsonConf.json"

    val jsonconf = new JsonConfig
    jsonconf.initConfig(jsonPath)

    //加载model
    val modelPath = jsonconf.getValue("w2v", "w2vmodelPath")
    val modelSize = jsonconf.getValue("w2v", "w2vmodelSize").toInt
    val isModel = jsonconf.getValue("w2v", "isModel").toBoolean
    val w2vModel = Word2VecModel.load(sc, modelPath)

    // 平衡集
    val docsParentPath = jsonconf.getValue("classifyw2v", "dataSets")
    val docsAll = sc.wholeTextFiles(docsParentPath).collect()

    // 模型保存路径
    val savePath = jsonconf.getValue("classifyw2v", "modelParentPath_SVM")
    val log = jsonconf.getValue("classifyw2v", "modelParentPath_SVM")+"log.txt"
    val writer = new PrintWriter(log, "utf-8")
    var count = 1

    // inputs
    docsAll.foreach(docs => {
      println("[正在训练模型] "+ count)
      val title = docs._1.substring(docs._1.indexOf("Sets/")+5, docs._1.length)

      val (data, posCount, negCount) = processing.process_weight(docs._2.split("\n"), sc, w2vModel, modelSize)

      println("[模型] "+ count+"[完成DOC2Vec模型]>>>>>>>>>>>>>>>>>")

      val dataRdd = sc.parallelize(data)
      val splits = dataRdd.randomSplit(Array(0.8, 0.2), seed = 11L)
      val train = splits(0)
      val test = splits(1)

      val model = SVMWithSGD.train(train, 1000)
      println("[模型] "+ count+ "[完成模型训练]>>>>>>>>>>>>>>>>>>>")

      val predictAndLabels = test.map{
        case LabeledPoint(label, features) =>
          val prediction = model.predict(features)
          (prediction, label)
      }

      writer.append(s"\n[$title]\n")
      writer.append(s"  ...[数据集总量]: ${docs._2.split("\n").length}\t[正例] $posCount\t[负例]$negCount")
      val metrics = new MulticlassMetrics(predictAndLabels)
      writer.append(s"  ...[综合_Precison]:${metrics.precision}\n")
      metrics.labels.foreach(label => {
        writer.append(s"  ...[${label}_Precison]:${metrics.precision(label)}\t[${label}_Recall]:${metrics.recall(label)}\n")
      })
      writer.flush()

      val modelPath = savePath+title+".model"
      model.save(sc, modelPath)
      println("[模型] "+ count+ "[完成模型保存]>>>>>>>>>>>>>>>>>>>")
      count += 1
    })


  }

  /**
    * 批量训练模型
    */
  def training_bayes(): Unit ={

    val conf = new SparkConf().setAppName("W2V").setMaster("local")
    val sc = new SparkContext(conf)
    // val jsonPath = "/home/zhangxin/work/workplace_scala/Sentiment/src/main/scala/classificationW2V/W2VJsonConf.json"
    val jsonPath = "/Users/li/workshop/MyRepository/Word2Vec/src/main/scala/twc/W2VJsonConf.json"
    val jsonconf = new JsonConfig
    jsonconf.initConfig(jsonPath)

    //加载model
    val modelPath = jsonconf.getValue("w2v", "w2vmodelPath")
    val modelSize = jsonconf.getValue("w2v", "w2vmodelSize").toInt
    val isModel = jsonconf.getValue("w2v", "isModel").toBoolean
    val w2vModel = Word2VecModel.load(sc, modelPath)

    // 平衡集

    val docsParentPath = jsonconf.getValue("classifyw2v", "dataSets")
    val docsAll = sc.wholeTextFiles(docsParentPath).collect()

    // 模型保存路径
    val savePath = jsonconf.getValue("classifyw2v", "modelParentPath_bayes")
    val log = jsonconf.getValue("classifyw2v", "modelParentPath_bayes")+"log.txt"
    val writer = new PrintWriter(log, "utf-8")
    var count = 1

    // inputs
    docsAll.foreach(docs => {
      println("[正在训练模型] "+ count)
      val title = docs._1.substring(docs._1.indexOf("Sets/")+5, docs._1.length)

      val (data, posCount, negCount) = processing.process_weight_beyes(docs._2.split("\n"), sc, w2vModel, modelSize)

      println("[模型] "+ count +"[完成DOC2Vec模型]>>>>>>>>>>>>>>>>>")

      val dataRdd = sc.parallelize(data)
      val splits = dataRdd.randomSplit(Array(0.8, 0.2), seed = 11L)
      val train = splits(0)
      val test = splits(1)

      val model = NaiveBayes.train(train, 1000)
      println("[模型] "+ count+ "[完成模型训练]>>>>>>>>>>>>>>>>>>>")

      val predictAndLabels = test.map{
        case LabeledPoint(label, features) =>
          val prediction = model.predict(features)
          (prediction, label)
      }

      writer.append(s"\n[$title]\n")
      writer.append(s"  ...[数据集总量]: ${docs._2.split("\n").length}\t[正例] $posCount\t[负例]$negCount\n")
      val metrics = new MulticlassMetrics(predictAndLabels)
      writer.append(s"  ...[综合_Precison]:${metrics.precision}\n")
      metrics.labels.foreach(label => {
        writer.append(s"  ...[${label}_Precison]:${metrics.precision(label)}\t[${label}_Recall]:${metrics.recall(label)}\n")
      })
      writer.flush()

      val modelPath = savePath+title+".model"
      model.save(sc, modelPath)
      println("[模型] "+ count+ "[完成模型保存]>>>>>>>>>>>>>>>>>>>")
      count += 1
    })


  }

  /**
    * 训练单模型
    */
  def trainingTest_bayes(): Unit ={

    val conf = new SparkConf().setAppName("W2V").setMaster("local")
    val sc = new SparkContext(conf)
    val jsonPath = "/home/zhangxin/work/workplace_scala/Sentiment/src/main/scala/classificationW2V/W2VJsonConf.json"

    val jsonconf = new JsonConfig
    jsonconf.initConfig(jsonPath)

    //加载model
    val modelPath = jsonconf.getValue("w2v", "w2vmodelPath")
    val modelSize = jsonconf.getValue("w2v", "w2vmodelSize").toInt
    val isModel = jsonconf.getValue("w2v", "isModel").toBoolean
    val w2vModel = Word2VecModel.load(sc, modelPath)

    // 平衡集
    //    val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/房地产"
    //    val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/有色金属"
//    val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/保险"
    //    val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/医药"
//        val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/计算机"
//        val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/教育传媒"
    //    val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/农林牧渔"
    //    val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/仪电仪表"
    //    val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/外贸"
    //    val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/工程建筑"
//        val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/机械"
        val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/建材"
//        val docsPath = "/home/zhangxin/work/workplace_scala/Data/trainingSets/纺织服装"

    val docs = sc.textFile(docsPath).collect()

    // inputs
    val (data, posCount, negCount) = processing.process_weight_beyes(docs, sc, w2vModel, 100)

    println("[完成DOC2Vec模型]>>>>>>>>>>>>>>>>>")

    val dataRdd = sc.parallelize(data)
    val splits = dataRdd.randomSplit(Array(0.8, 0.2), seed = 11L)
    val train = splits(0)
    val test = splits(1)

    val model = NaiveBayes.train(train, 1000)
    println("[完成模型训练]>>>>>>>>>>>>>>>>>>>")


    val predictAndLabels = test.map{
      case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
    }

    val metrics = new MulticlassMetrics(predictAndLabels)
    println(s"[综合_Precison] ${metrics.precision}")
    println(s"[Labels] ${metrics.labels.toList}")
    metrics.labels.foreach(label => {
      println(s"[${label}_Precison] ${metrics.precision(label)}")
    })

  }


  def main(args: Array[String]): Unit = {
    training_bayes()
  }

}
