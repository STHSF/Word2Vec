package utils

import org.json.JSONObject

import scala.io.Source
import scala.util.parsing.json.JSON


/**
  * Created by li on 16/8/29.
  * 读取json格式的额配置文件信息.
  */
object JSONUtil {

  private var config = new JSONObject()

  /**
    * 初始化类
    *
    * @param confDir 配置文件路径
    */
  private def initConfig(confDir: String): Unit = {

    val jsObj = Source.fromFile(confDir).getLines().mkString("")
    config = new JSONObject(jsObj)
  }


  /**
    * 获取配置文件中的相应的值
    *
    * @param key1 定位key
    * @param key2 定位key
    * @return 返回字符串
    */
  def getValue(key1: String, key2: String): String = {

    val jsonPath = "/Users/li/workshop/MyRepository/Word2Vec/src/main/scala/twc/W2VJsonConf.json"
    initConfig(jsonPath)
    config.getJSONObject(key1).getString(key2)
  }


  private def readConfigFile(confDir: String): Map[String, Any] = {

    val jsonFile = Source.fromFile(confDir).mkString

    val json = JSON.parseFull(jsonFile)

    json match {

      case Some(map: Map[String, Any]) => map
      //      case None => println("Parsing failed")
      //      case other => println("Unknown data structure: " + other)
    }

  }

}
