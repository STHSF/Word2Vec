package twc.train

import breeze.linalg.Vector
import com.kunyandata.nlpsuit.wordExtraction.TextRank
import org.apache.spark.SparkContext
import org.apache.spark.mllib.feature.Word2VecModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

import scala.collection.mutable

/**
  * Created by zhangxin on 16-11-9.
  *
  * 数据处理:利用word2vec模型处理输入数据,输出为labelpoint格式
  */
object processing {

  /**
    * 已分词文档,平衡集
    * @param docs
    * @param sc
    * @param w2vModel
    * @param modelSize
    * @return
    */
  def process_weight(docs:Array[String], sc: SparkContext, w2vModel: Word2VecModel, modelSize: Int) = {

    val docsTemp = docs.map(doc  => {
      val temp = doc.split("\t")
      val label = temp(0)
      val seg = temp(1).split(",")

      //textRank
      val keywords = TextRank.run("k", 10, seg.toList, 20, 50, 0.85f)
      val keywordsFilter = keywords.toArray.filter(word => word._1.length >= 2)

      (label, keywordsFilter)
    })

    //统计pos和neg的个数
    val posCount = docsTemp.count(_._1.equals("1.0"))
    val negCount = docsTemp.count(_._1.equals("0.0"))

    val result = docsTemp.map(doc => {
      val resultTemp = doc2vecWithModel_weight(doc._2, w2vModel, modelSize)
      val vector = Vectors.dense(resultTemp)
      val label = doc._1.toDouble
      LabeledPoint(label, vector)
    })

    (result, posCount, negCount)

  }

  //已分词文档,平衡集
  def process_weight_beyes(docs:Array[String], sc: SparkContext, w2vModel: Word2VecModel, modelSize: Int) = {

    val docsTemp = docs.map(doc  => {
      val temp = doc.split("\t")
      val label = temp(0)
      val seg = temp(1).split(",")

      //textRank
      val keywords = TextRank.run("k", 10, seg.toList, 20, 50, 0.85f)
      val keywordsFilter = keywords.toArray.filter(word => word._1.length >= 2)

      (label, keywordsFilter)
    })

    val posCount = docsTemp.filter(_._1.equals("1.0")).length
    val negCount = docsTemp.filter(_._1.equals("0.0")).length

    val result = docsTemp.map(doc => {
      val resultTemp = doc2vecWithModel_weight_beyes(doc._2, w2vModel, modelSize)
      val vector = Vectors.dense(resultTemp)
      val label = doc._1.toDouble
      LabeledPoint(label, vector)
    })

    (result, posCount, negCount)

  }

  /**
    * 基于word2vec model，构建单文档向量模型
    * @param doc
    * @param model 词向量模型
    * @param vectorSize 词向量的长度
    * @return
    */
  private def doc2vecWithModel_weight(doc: Array[(String, Float)], model:Word2VecModel, vectorSize: Int): Array[Double] = {

    var resultTemp = new Array[Double](vectorSize)
    var wordTemp = new Array[Double](vectorSize)

    doc.foreach(word => {
      try {
        wordTemp = model.transform(word._1).toArray
      }
      catch {
        case e: Exception => wordTemp = Vector.zeros[Double](vectorSize).toArray
      }

      for (i <- resultTemp.indices){
        resultTemp(i) += wordTemp(i) * word._2
      }
    })

    val docVec = resultTemp
    docVec
  }

  // 基于word2vec model 获取单文档词向量 ,并进行归一化, 可用于贝叶斯分类
  private def doc2vecWithModel_weight_beyes(doc: Array[(String, Float)], model:Word2VecModel, modelSize: Int): Array[Double] = {

    var resultTemp = new Array[Double](modelSize)
    var wordTemp = new Array[Double](modelSize)

    doc.foreach(word => {
      try {
        wordTemp = model.transform(word._1).toArray
      }
      catch {
        case e: Exception => wordTemp = Vector.zeros[Double](modelSize).toArray
      }

      for (i <- resultTemp.indices){
        resultTemp(i) += wordTemp(i) * word._2
      }
    })

    val docVec = resultTemp.map(_+70)

    docVec
  }

  // 基于txt型文件获取单文档的词向量
  private def doc2vecWithHash(doc: Array[String], model:mutable.HashMap[String, Array[Double]], modelSize: Int) = {
    var resultTemp = new Array[Double](modelSize)
    var wordTemp = new Array[Double](modelSize)

    doc.foreach(word => {
      try {
        wordTemp = model(word)
      }
      catch {
        case e: Exception => wordTemp = Vector.zeros[Double](modelSize).toArray
      }

      for (i <- resultTemp.indices){
        resultTemp(i) += wordTemp(i)
      }
    })

    val docVec = resultTemp.map(vec => vec/doc.length)
    docVec
  }

}
