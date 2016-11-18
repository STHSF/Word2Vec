package twc.predict

import breeze.linalg.Vector
import com.kunyandata.nlpsuit.util.{KunyanConf, TextPreprocessing}
import com.kunyandata.nlpsuit.wordExtraction.TextRank
import org.apache.spark.mllib.feature.Word2VecModel

/**
  * Created by zhangxin on 16-11-17.
  */
object Process {

  /*****************************************************************/
  /*SVM 数据处理
  *****************************************************************/
  //未分词, (标签, 全文)
  def process(doc: String, w2vModel: Word2VecModel, modelSize: Int, kunyan: KunyanConf, stopwords: Array[String]) = {

    val docSeg = TextPreprocessing.process(doc, stopwords, kunyan)

    //textRank
    val keywords = TextRank.run("k", 10, docSeg.toList, 20, 50, 0.85f)
    val keywordsFilter = keywords.toArray.filter(word => word._1.length >= 2)
    val result = doc2vecWithModel_weight(keywordsFilter, w2vModel, modelSize)

    result

  }

  /**
    * 获得词向量,并由词向量获取文档向量
    * @param doc
    * @param model
    * @param modelSize
    * @return
    */
  private def doc2vecWithModel_weight(doc: Array[(String, Float)], model:Word2VecModel, modelSize: Int): Array[Double] = {

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

    val docVec = resultTemp
    docVec
  }


  /*****************************************************************/
  /* Bayes 数据处理
  *****************************************************************/
  //未分词, (标签, 全文)
  def process_bayes(doc: String, w2vModel: Word2VecModel, modelSize: Int, kunyan: KunyanConf, stopwords: Array[String]) = {

    val docSeg = TextPreprocessing.process(doc, stopwords, kunyan)

    //textRank
    val keywords = TextRank.run("k", 10, docSeg.toList, 20, 50, 0.85f)
    val keywordsFilter = keywords.toArray.filter(word => word._1.length >= 2)
    val result = doc2vecWithModel_weight_beyes(keywordsFilter, w2vModel, modelSize)

    result

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
}
