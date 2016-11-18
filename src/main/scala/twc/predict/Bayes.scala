package twc.predict

import java.io.File
import java.math.{BigDecimal, RoundingMode}
import java.text.DecimalFormat

import com.kunyandata.nlpsuit.util.{JsonConfig, KunyanConf}
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by zhangxin on 16-11-17.
  *
  * 基于 word2vec + 贝叶斯的文本分类
  */
object Bayes {

  def init(sc: SparkContext, jsonPath: String) = {

    val jsonconf = new JsonConfig
    jsonconf.initConfig(jsonPath)

    //加载坤雁分词 + 停用词
    val kunyan = new KunyanConf
    kunyan.set(jsonconf.getValue("kunyan","ip"),jsonconf.getValue("kunyan","port").toInt)
    val stopWords = sc.textFile(jsonconf.getValue("kunyan", "stopwords")).collect()

    //加载word2vec模型
    val modelPath = jsonconf.getValue("w2v", "w2vmodelPath")
    val modelSize = jsonconf.getValue("w2v", "w2vmodelSize").toInt
    val isModel = jsonconf.getValue("w2v", "isModel").toBoolean
    val w2vModel = Word2VecModel.load(sc, modelPath)

    //加载分类模型
    val modelParentPath = jsonconf.getValue("classifyw2v", "modelParentPath_bayes")
    val modelFiles = new File(modelParentPath).listFiles()
    val classificationModels = modelFiles.map(tempFile => {
      val name = tempFile.getName
      val model = NaiveBayesModel.load(sc, tempFile.getAbsolutePath)
      (name, model)
    })

    (kunyan, stopWords, w2vModel, classificationModels)
  }

  def predict(doc: String, w2vModel: Word2VecModel, classifyModel: Array[(String, NaiveBayesModel)], kunyan: KunyanConf, stopwords: Array[String]) = {

    val docProcess = Process.process_bayes(doc, w2vModel, 100, kunyan, stopwords)
    val docVector = Vectors.dense(docProcess)

    val resultPredict = classifyModel.map(model => {
      val modelName = model._1
      val prediction = model._2.predictProbabilities(docVector).toArray


      //取数据的小数点后四位
      val temp = prediction.map(e => {
        val temp = new BigDecimal(e)
        temp.setScale(4, RoundingMode.FLOOR).toString.toDouble
      })

      //取消科学计数法
      val df = new DecimalFormat("0.0000")

      (modelName, df.format(temp(1)))

    })

    val result = resultPredict.toSeq.sortWith(_._2 > _._2).take(5)
    result.foreach(line => println("   "+line))

  }

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("W2V").setMaster("local")
    val sc = new SparkContext(conf)
    val jsonPath = "/home/zhangxin/work/workplace_scala/Sentiment/src/main/scala/classificationW2V/W2VJsonConf.json"

    val (kunyan, stopWords, w2vModel, classificationModels) = init(sc, jsonPath)

    //测试数据
    val testPath = "/home/zhangxin/work/workplace_scala/Data/classificaiton_test"

    val testData = sc.wholeTextFiles(testPath).map(doc => {
      val title = doc._1.substring(doc._1.indexOf("test")+5, doc._1.length)
      val content = doc._2

      (title, content)
    }).collect()

    testData.foreach(doc=>{
      val begin = System.currentTimeMillis()
      println(s"\n\n[${doc._1}]")

      predict(doc._2, w2vModel, classificationModels,kunyan, stopWords)

      println("   耗时: " + (System.currentTimeMillis() - begin))
    })

  }

}
