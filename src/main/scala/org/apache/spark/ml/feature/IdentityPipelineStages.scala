package org.apache.spark.ml.feature

import org.apache.spark.annotation.{ Experimental, Since }
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.Vectors
//import org.apache.spark.ml.param.{ Params, IntParam, DoubleParam, ParamMap, ParamValidators, StringArrayParam, Param, BooleanParam }
import org.apache.spark.ml.param.shared.{ HasInputCols, HasOutputCol, HasWeightCol }
import org.apache.spark.ml.util.{ DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils }
import org.apache.spark.sql.{ DataFrame, Dataset, Row }
import org.apache.spark.sql.functions._
//import org.apache.spark.sql.types.{StructType,LongType,ArrayType,StringType,StructField,ByteType,DoubleType}
import org.apache.spark.util.Utils
import org.apache.spark.util.collection.OpenHashMap
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.graphframes._

object UDF extends Serializable {

  val zip4 = (aaa: Map[String, Int]) => udf((x: Seq[String], y: Seq[String], z: Seq[String]) => (x, y, z).zipped.toList.filter(a => aaa.contains(a._1)).map {
    case (a: String, b: String, c: String) => aaa.get(a).get.toString() ++ c ++ b
    case (a: String, b: String, null)      => aaa.get(a).get.toString() ++ b
    case (_)                               => "BAD_ID"
  })

  def addressShingles(stopWords: Set[String], toLowercase: Boolean = true, gaps: Boolean = true, minLength: Integer = 3, pattern: String = "\\W+", tokenTypeDelimeter: String = "$", postalCodeDelimeter: String = "-") = udf((use: String, line: Seq[String], city: String, state: String, postalCode: String, country: String) => {
    val originStr = if (line == null) "" else line.mkString(" ")
    val re = pattern.r
    val str = if (toLowercase) originStr.toLowerCase() else originStr
    val streetTokens = if (gaps) re.split(str).toSeq else re.findAllIn(str).toSeq
    val filteredStreetTokens = streetTokens.filter { token => (token.length >= minLength || token.forall(Character.isDigit(_))) && !stopWords.contains(token) }
    val useToken = if (use != null) s"u${tokenTypeDelimeter}${use}" else use
    val cityToken = if (city != null) s"c${tokenTypeDelimeter}${city}" else city
    val stateToken = if (state != null) s"s${tokenTypeDelimeter}${state}" else state
    val zipToken = if (postalCode != null) {
      val nodel = postalCode.replaceAll(postalCodeDelimeter, ""); val main = if (nodel.size >= 5) nodel.substring(0, 5) else nodel; Set(main, nodel).map(z => s"z${tokenTypeDelimeter}${z}").toSeq
    } else Seq()
    val countryToken = if (country != null && !country.trim.toLowerCase.equals("US")) s"t${tokenTypeDelimeter}${country}" else null
    val shingles = zipToken ++ Seq(countryToken, cityToken, useToken).filter(_ != null) ++ filteredStreetTokens
    shingles
  })

  val isEmpty = udf((x: Seq[String]) => x == null || x.isEmpty)

  val flat = udf((x: Seq[Seq[String]]) => if (x == null || x.isEmpty) Seq[String]() else x.filter(_ != null).flatten)

  val dateShingle2 = udf((s: String) => { val y = s.substring(0, 4); val m = s.substring(4, 6); val d = s.substring(6, 8); List(m, d, y, m ++ d, d ++ m, m ++ d ++ y) })

  val score = udf((weights: Seq[Double]) => { weights.foldLeft(0.0) { (a, b) => a + b } })

  val dobGenderShingles = udf((value: Long, offset: Int, gender: String) => {
    val TICKS_AT_EPOCH = 621355968000000000l; val TICKS_PER_MILLISECOND = 10000
    val dateTimeTick = ((value - TICKS_AT_EPOCH) / TICKS_PER_MILLISECOND) + (offset * 60000); Seq(gender, dateTimeTick.toString)
  })

  val telecomShingles = udf((telecomType: Seq[String], telecomValue: Seq[String]) => (telecomType, telecomValue).zipped.toList.map {
    case (tt: String, tv:String) => tv.filter(Character.isDigit)
    case (_)                     => "BAD_TELECOM"
  })

  
  def aliasAll(t: DataFrame, p: String = "", s: String = ""): DataFrame = { t.select(t.columns.map { c => t.col(c).as(p + c + s) }: _*) }
}

trait HasMatchTypeAndIdColumn extends  org.apache.spark.ml.param.Params {
  protected val matchType = new  org.apache.spark.ml.param.Param(this, "matchType",
    "match type must be one of { Identifier,GivenName,FamilyName,Address,BirthDateGender}",
     org.apache.spark.ml.param.ParamValidators.inArray(Array("Identifier", "GivenName", "FamilyName", "Address", "BirthDateGender")))

  def getMatchType: String = $(matchType)
  def setMatchType(value: String): this.type = set(matchType, value)

  protected val matchTypeIdx = new  org.apache.spark.ml.param.IntParam(this, "matchTypeId", "Integer number starting from 0",  org.apache.spark.ml.param.ParamValidators.gt(-1))
  setDefault(matchTypeIdx, 0)
  def getMatchTypeIdx: Int = $(matchTypeIdx)
  def setMatchTypeIdx(value: Int): this.type = set(matchTypeIdx, value)

  // This is the "id" column not the "identifier"
  private val identifierColumnName = new  org.apache.spark.ml.param.Param[String](this, "identifierColumnName", "default is \"id\"")
  setDefault(identifierColumnName, "id")
  def getIdentifierColumnName: String = $(identifierColumnName)
  def setIdentifierColumnName(value: String): this.type = set(identifierColumnName, value)

  protected def requireColumn(schema: org.apache.spark.sql.types.StructType, columnName: String) = { require(schema.fieldNames.contains(columnName), s"$columnName column is required but not found among {${schema.fieldNames.mkString(",")}}") }

  def validateMatchTypeAndIdColumn(schema: org.apache.spark.sql.types.StructType) = {
    getMatchType match {
      case "Identifier"               => requireColumn(schema, "identifier");
      case "GivenName" | "FamilyName" => requireColumn(schema, "name")
      case "Address"                  => requireColumn(schema, "address")
      case "BirthDateGender"          => { requireColumn(schema, "birthDate"); requireColumn(schema, "gender") }
      case "Telecom"                  => requireColumn(schema, "telecom")
      case _                          => { throw new Exception(s"Unsupported matchType:$matchType") }
    }
    assert(schema.fields.find(_.name == getIdentifierColumnName).get.dataType == org.apache.spark.sql.types.LongType, s"$getIdentifierColumnName must be of type Long")
  }

}

trait HasLabelColumn extends  org.apache.spark.ml.param.Params {
  protected val keepLabelColumn = new  org.apache.spark.ml.param.BooleanParam(this, "keepLabelColumn", "match type must be one of {true,false}. default is false. It is not recommended to run this with true for production for performance reasons.")
  setDefault(keepLabelColumn -> false)
  def getKeepLabelColumn: Boolean = $(keepLabelColumn)
  def setKeepLabelColumn(value: Boolean): this.type = set(keepLabelColumn, value)

  val labelColumnName = new  org.apache.spark.ml.param.Param[String](this, "labelColumnName", "default is \"syntheticId\"")
  setDefault(labelColumnName, "syntheticId")
  def getLabelColumnName: String = $(labelColumnName)
  def setLabelColumnName(value: String): this.type = set(labelColumnName, value)
}

class PatientColumnSelector(override val uid: String) extends Transformer with HasLabelColumn with HasMatchTypeAndIdColumn with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("PatientColumnSelector"))

  def columnsToKeep(schema: org.apache.spark.sql.types.StructType, matchType: String) = {
    val labeled = (schema.fieldNames.contains("repoId") && getKeepLabelColumn && (schema.fieldNames.contains(getLabelColumnName)))

    val selectColumns = getMatchType match {
      case "Identifier"               => List("resourceId", "identifier", getIdentifierColumnName)
      case "GivenName" | "FamilyName" => List("resourceId", "name", getIdentifierColumnName)
      case "Address"                  => List("resourceId", "address", getIdentifierColumnName)
      case "BirthDateGender"          => List("resourceId", "birthDate", "gender", getIdentifierColumnName)
      case "Telecom"                  => List("resourceId", "telecom", getIdentifierColumnName)
      case _                          => List()
    }

    if (labeled) selectColumns ++ List("repoId", "resourceId", getLabelColumnName) else selectColumns
  }

  override def transformSchema(schema: org.apache.spark.sql.types.StructType): org.apache.spark.sql.types.StructType = {

    validateMatchTypeAndIdColumn(schema);
    val columns = columnsToKeep(schema, this.getMatchType)
    val s = org.apache.spark.sql.types.StructType(schema.filter { f => columns.contains(f.name) })
    s
  }

  override def transform(dataset: org.apache.spark.sql.Dataset[_]): org.apache.spark.sql.DataFrame = {
    val columns = columnsToKeep(dataset.schema, this.getMatchType)
    dataset.select(columns.head, columns.tail: _*);
  }
  override def copy(extra: org.apache.spark.ml.param.ParamMap): org.apache.spark.ml.Transformer = defaultCopy(extra)

}

object PatientColumnSelector extends DefaultParamsReadable[PatientColumnSelector] {

  override def load(path: String): PatientColumnSelector = super.load(path)

  def apply(matchType: String) = { val pcs = new PatientColumnSelector(); pcs.setMatchType(matchType); pcs }
}

class PatientShingler(override val uid: String) extends Transformer with HasLabelColumn with HasMatchTypeAndIdColumn with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("PatientShingler"))

  private val defaultStopWords = Set("ln", "st", "street", "pl", "place", "ct", "court", "loop", "crt", "apt", "way", "rd", "w", "e", "n", "s", "cir", "circle", "av.", "ave", "ave.", "dr", "blvd", "road", "rd", "po", "box")

  protected val stopWords = new  org.apache.spark.ml.param.StringArrayParam(this, "stopWords", "must be a sequence of words commonly used in Address'es. Example: {ln,street,place, etc.}")
  setDefault(stopWords -> defaultStopWords.toArray[String])
  def getStopWords: Array[String] = $(stopWords)
  def setStopWords(value: Array[String]): this.type = set(stopWords.asInstanceOf[org.apache.spark.ml.param.Param[Any]], value.toSet.toArray)

  private val sentaraAAA = Set("1.2.840.114350.1.13.163.2.7.3.688884.100", "2.16.840.1.113883.3.564.610", "2.16.840.1.113883.3.564.1754")

  protected val aaa = new  org.apache.spark.ml.param.StringArrayParam(this, "aaa", "must be a sequence of AAA identifiers. Example: {\"1.2.840.114350.1.13.163.2.7.3.688884.100\", \"2.16.840.1.113883.3.564.610\"}")
  setDefault(aaa -> sentaraAAA.toArray[String])
  def getAAA: Map[String, Int] = $(aaa).zipWithIndex.toMap
  def setAAA(value: Array[String]): this.type = set(aaa.asInstanceOf[org.apache.spark.ml.param.Param[Any]], value.toSet.toArray)

  override def transformSchema(schema: org.apache.spark.sql.types.StructType): org.apache.spark.sql.types.StructType = {
    validateMatchTypeAndIdColumn(schema)
    requireColumn(schema, getIdentifierColumnName);
    if (getKeepLabelColumn) for (c <- List(getLabelColumnName, "repoId", "resourceId")) requireColumn(schema, c)

    getMatchType match {
      case "Identifier"      => schema.add(org.apache.spark.sql.types.StructField("identifierShingles", org.apache.spark.sql.types.ArrayType(org.apache.spark.sql.types.StringType, true), true))
      case "GivenName"       => schema.add(org.apache.spark.sql.types.StructField("givenNameShingles", org.apache.spark.sql.types.ArrayType(org.apache.spark.sql.types.StringType, true), true))
      case "FamilyName"      => schema.add(org.apache.spark.sql.types.StructField("familyNameShingles", org.apache.spark.sql.types.ArrayType(org.apache.spark.sql.types.StringType, true), true))
      case "Address"         => schema.add(org.apache.spark.sql.types.StructField("addressShingles", org.apache.spark.sql.types.ArrayType(org.apache.spark.sql.types.StringType, true), true))
      case "BirthDateGender" => schema.add(org.apache.spark.sql.types.StructField("birthDateGenderShingles", org.apache.spark.sql.types.ArrayType(org.apache.spark.sql.types.StringType, true), true))
      case "Telecom"         => schema.add(org.apache.spark.sql.types.StructField("telecomShingles", org.apache.spark.sql.types.ArrayType(org.apache.spark.sql.types.StringType, true), true))
      case _                 => schema
    }
  }

  // Members declared in org.apache.spark.ml.Transformer
  override def copy(extra: org.apache.spark.ml.param.ParamMap): org.apache.spark.ml.Transformer = defaultCopy(extra)

  def transform(dataset: org.apache.spark.sql.Dataset[_]): org.apache.spark.sql.DataFrame = {

    val p = dataset

    def selectNameColumns(p: Dataset[_], labeled: Boolean) = {
      if (labeled) p.filter("name is not null").select("syntheticId", "resourceId", "repoId", "id", "name.given", "name.family")
      else p.filter("name is not null").select(col(getIdentifierColumnName), col("name.given"), col("name.family"))
    }

    (getMatchType, getKeepLabelColumn) match {
      case ("Identifier", labeled) => {
        val pId0 = if (labeled) p.filter("identifier is not null").select("syntheticId", "resourceId", "repoId", "id", "identifier.system", "identifier.value", "identifier.type.text")
        else p.filter("identifier is not null").select("id", "identifier.system", "identifier.value", "identifier.type.text")

        val pId1 = pId0.withColumn("identifierShingles", UDF.zip4(getAAA)(col("system"), col("value"), col("text"))).drop("system", "value", "text")

        val pIdSchema = pId1.schema
        val filteredPId = pId0.rdd.filter(!_.getList[String](pIdSchema.fieldIndex("identifierShingles")).isEmpty)
        dataset.sqlContext.createDataFrame(filteredPId, pIdSchema)
      }
      case ("Address", labeled) =>
        {
          val pAddress0 = p.filter("address is not null").withColumnRenamed("address", "addresses").withColumn("address", explode(col("addresses"))).drop("addresses")
          val pAddress1 = if (labeled) pAddress0.select("syntheticId", "resourceId", "repoId", "id", "address.use.value", "address.line", "address.city", "address.state", "address.postalCode", "address.country")
          else pAddress0.select("resourceId", "id", "address.use.value", "address.line", "address.city", "address.state", "address.postalCode", "address.country")

          val pAddress2 = pAddress1.withColumn("addressShingles", UDF.addressShingles(getStopWords.toSet)(col("value"), col("line"), col("city"), col("state"), col("postalCode"), col("country")))
          pAddress2.filter(!UDF.isEmpty(col("addressShingles"))).drop("value", "line", "city", "state", "country", "postalCode")
        }
      case ("GivenName", labeled)  => { selectNameColumns(p, labeled).withColumn("givenNameShingles", UDF.flat(col("given"))).drop("given", "family") }

      case ("FamilyName", labeled) => { selectNameColumns(p, labeled).withColumn("familyNameShingles", UDF.flat(col("family"))).drop("family", "given") }

      case ("BirthDateGender", labeled) => {
        if (labeled) p.select("syntheticId", "resourceId", "repoId", "id", "birthDate", "gender").withColumn("birthDateGenderShingles", UDF.dobGenderShingles(col("birthDate.value"), col("birthDate.offset"), col("gender.value"))).drop("birthDate", "gender")
        else p.select("id", "birthDate", "gender").withColumn("birthDateGenderShingles", UDF.dobGenderShingles(col("birthDate.value"), col("birthDate.offset"), col("gender.value"))).drop("birthDate", "gender")
      }

      case ("Telecom", labeled) => {
        if (labeled) p.select("syntheticId", "resourceId", "repoId", "id", "birthDate", "gender").withColumn("telecomShingles", UDF.telecomShingles(col("telecom.system.value"), col("telecom.value")))
        else p.select("id", "birthDate", "gender").withColumn("telecomShingles", UDF.telecomShingles(col("telecom.system.value"), col("telecom.value"))).drop("telecom")
      }

      case (matchType, _) => { throw new Exception(s"Unsupported matchType:$matchType") } // raise error
    }
  }
}

object PatientShingler extends DefaultParamsReadable[PatientShingler] {

  override def load(path: String): PatientShingler = super.load(path)
  def apply(matchType: String) = { val ps = new PatientShingler(); ps.setMatchType(matchType); ps }
}

class SimilarPairFinder(override val uid: String) extends Transformer with HasLabelColumn with HasMatchTypeAndIdColumn with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("SimilarPairFinder"))

  val matchTreshold = new  org.apache.spark.ml.param.DoubleParam(this, "threshold", "default is \"0.1\"")
  setDefault(matchTreshold, 0.1)
  def getMatchThreshold: Double = $(matchTreshold)
  def setMatchThreshold(value: Double): this.type = set(matchTreshold, value)

  val lshModel = new  org.apache.spark.ml.param.Param[MinHashLSHModel](this, "lshModel", "LSH model")
  def getLshModel: MinHashLSHModel = $(lshModel)
  def setLshModel(value: MinHashLSHModel): this.type = set(lshModel, value)

  val weightColumnName = new  org.apache.spark.ml.param.Param[String](this, "weightColumnName", "default is \"weight\"")
  setDefault(weightColumnName, "weight")
  def getWeightColumnName: String = $(weightColumnName)
  def setWeightColumnName(value: String): this.type = set(weightColumnName, value)

  val distanceColumnName = new  org.apache.spark.ml.param.Param[String](this, "distanceColumnName", "default is \"distance\"")
  setDefault(distanceColumnName, "distance")
  def getDistanceColumnName: String = $(distanceColumnName)
  def setDistanceColumnName(value: String): this.type = set(distanceColumnName, value)

  val matchTypeColumnName = new  org.apache.spark.ml.param.Param[String](this, "matchTypeColumnName", "default is \"matchType\"")
  setDefault(matchTypeColumnName, "matchType")
  def getMatchTypeColumnName: String = $(matchTypeColumnName)
  def setMatchTypeColumnName(value: String): this.type = set(matchTypeColumnName, value)

  protected val filterIdentityPairs = new  org.apache.spark.ml.param.BooleanParam(this, "filterIdentityPairs", "match type must be one of {true,false}. default is false. Whether or not to filter out pairs of rows with same id")
  setDefault(filterIdentityPairs -> false)
  def getFilterIdentityPairs: Boolean = $(filterIdentityPairs)
  def setFilterIdentityPairs(value: Boolean): this.type = set(filterIdentityPairs, value)

  val aggreementWeight = new  org.apache.spark.ml.param.DoubleParam(this, "aggreementWeight", "typically positive number. default is \"4\"",  org.apache.spark.ml.param.ParamValidators.gt(-20))
  setDefault(aggreementWeight, 4.toDouble)
  def getAggreementWeight: Double = $(aggreementWeight)
  def setAggreementWeight(value: Double): this.type = set(aggreementWeight, value)

  val disaggreementWeight = new  org.apache.spark.ml.param.DoubleParam(this, "disaggreementWeight", "typically negative number. default is \"-1\"",  org.apache.spark.ml.param.ParamValidators.lt(0))
  setDefault(disaggreementWeight, (-1).toDouble)
  def getDisaggreementWeight: Double = $(disaggreementWeight)
  def setDisaggreementWeight(value: Double): this.type = set(disaggreementWeight, value)

  override def transformSchema(schema: org.apache.spark.sql.types.StructType): org.apache.spark.sql.types.StructType = {
    schema.add(org.apache.spark.sql.types.StructField("idA", org.apache.spark.sql.types.ArrayType(org.apache.spark.sql.types.LongType, true), true))
      .add(org.apache.spark.sql.types.StructField("idB", org.apache.spark.sql.types.ArrayType(org.apache.spark.sql.types.LongType, true), true))
      .add(org.apache.spark.sql.types.StructField(getMatchTypeColumnName, org.apache.spark.sql.types.ByteType, true))
      .add(org.apache.spark.sql.types.StructField(getDistanceColumnName, org.apache.spark.sql.types.DoubleType, true))
      .add(org.apache.spark.sql.types.StructField(getWeightColumnName, org.apache.spark.sql.types.DoubleType, true))
  }

  // Members declared in org.apache.spark.ml.Transformer
  override def copy(extra: org.apache.spark.ml.param.ParamMap): org.apache.spark.ml.Transformer = defaultCopy(extra)

  def transform(dataset: org.apache.spark.sql.Dataset[_]): org.apache.spark.sql.DataFrame = {

    val pairs = getLshModel.approxSimilarityJoin(dataset, dataset, getMatchThreshold, getDistanceColumnName)
      .select(col(s"datasetA.$getIdentifierColumnName").alias("idA"), col(s"datasetB.$getIdentifierColumnName").alias("idB"), col(getDistanceColumnName))
pairs.printSchema()
dataset.show()
pairs.show()
println("Pairs before filter" + pairs.count)
    val filteredPairs = if (getFilterIdentityPairs) pairs.filter(col("idA") =!= col("idB")) else pairs
println("Pairs after filter" + filteredPairs.count)
   
    filteredPairs.withColumn(getMatchTypeColumnName, lit(getMatchTypeIdx))
      .withColumn(getWeightColumnName, lit(getAggreementWeight))
  }

}

object SimilarPairFinder extends DefaultParamsReadable[SimilarPairFinder] {

  override def load(path: String): SimilarPairFinder = super.load(path)
  def apply(matchType: String, matchThreshold: Double, aggreementWeight: Double, disaggreementWeight: Double) = {
    new SimilarPairFinder().setMatchType(matchType).setMatchThreshold(matchThreshold).setAggreementWeight(aggreementWeight).setDisaggreementWeight(disaggreementWeight)
  }
}

trait PipelineCombiner extends  org.apache.spark.ml.param.Params {

  def dependsOn(name: String, pipelineModel: PipelineModel) = {
    val internalName = s"${name}_pipelineModel"
    val pipelineModelParam = new  org.apache.spark.ml.param.Param[PipelineModel](this, internalName, internalName)
    set(internalName, pipelineModelParam)
  }

  def getPipelineModel(name: String): Option[PipelineModel] = {
    val internalName = s"${name}_pipelineModel"
    val parValue = params.find(p => internalName == p).map(get(_).get);
    if (parValue.isDefined) {
      val v = parValue.get
      if (v.isInstanceOf[PipelineModel]) parValue.asInstanceOf[Option[PipelineModel]]
    }
    None
  }
}


class MatchPairCombiner(override val uid: String) extends Transformer with PipelineCombiner with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("IdentityProfile"))

  val numberOfPartitions = new  org.apache.spark.ml.param.IntParam(this, "numberOfPartitions", "Number of partitions after all matches are united. Default is \"500\"")
  setDefault(numberOfPartitions, 500)
  def getNumberOfPartitions: Int = $(numberOfPartitions)
  def setNumberOfPartitions(value: Int): this.type = set(numberOfPartitions, value)

  val scoreColumnName = new  org.apache.spark.ml.param.Param[String](this, "scoreColumnName", "default is \"score\"")
  setDefault(scoreColumnName, "score")
  def getScoreColumnName: String = $(scoreColumnName)
  def setScoreColumnName(value: String): this.type = set(scoreColumnName, value)

  val weightsColumnName = new  org.apache.spark.ml.param.Param[String](this, "weightsColumnName", "default is \"weights\"")
  setDefault(weightsColumnName, "weights")
  def getWeightsColumnName: String = $(weightsColumnName)
  def setWeightsColumnName(value: String): this.type = set(weightsColumnName, value)
  
  val scoreThreshold = new  org.apache.spark.ml.param.DoubleParam(this, "scoreThreshold", "default is \"20\"")
  setDefault(scoreThreshold, 20.toDouble)
  def getScoreThreshold: Double = $(scoreThreshold)
  def setScoreThreshold(value: Double): this.type = set(scoreThreshold, value)


  protected val getDisagreementWeights: Map[String, Double] = {
    getMatchTypes.map { matchType =>
      val pm = getPipelineModel(matchType).get
      val spc = pm.stages.find(_.isInstanceOf[SimilarPairFinder]).get.asInstanceOf[SimilarPairFinder]
      (matchType -> spc.getDisaggreementWeight)
    }.toMap
  }

  val defaultMatchTypes: Seq[String] = Seq("Identifier", "Address", "GivenName", "FamilyName", "BirthDateGender", "Telecom")

  protected val matchTypes = new  org.apache.spark.ml.param.StringArrayParam(this, "matchTypes", "Example: {Identifier,Address,GivenName,FamilyName,Gender etc.}")
  setDefault(matchTypes -> defaultMatchTypes.toArray[String])
  def getMatchTypes: Array[String] = $(matchTypes)
  def setMatchTypes(value: Array[String]): this.type = set(matchTypes, value)

  // Members declared in org.apache.spark.ml.Transformer
  override def copy(extra: org.apache.spark.ml.param.ParamMap): org.apache.spark.ml.Transformer = defaultCopy(extra)

  protected def getMatchingPairs(pipelineModel: PipelineModel, dataset: Dataset[_]) = {
    val lshModel = pipelineModel.stages.find(_.isInstanceOf[MinHashLSHModel]).get.asInstanceOf[MinHashLSHModel]
    val sp = pipelineModel.stages.find(_.isInstanceOf[SimilarPairFinder]).get.asInstanceOf[SimilarPairFinder]
    val paramMap =  org.apache.spark.ml.param.ParamMap(sp.lshModel -> lshModel)
    pipelineModel.transform(dataset, paramMap)
  }

  override def transformSchema(schema: org.apache.spark.sql.types.StructType): org.apache.spark.sql.types.StructType = {
    schema
  }

  case class Match(idA: Long, idB: Long, matchType: Byte, distance: Double, weight: Double)

  def transform(dataset: org.apache.spark.sql.Dataset[_]): org.apache.spark.sql.DataFrame = {
    import dataset.sparkSession.implicits._

    val weightsColumn = getWeightsColumnName
    val matchingPairs = getMatchTypes.map(getPipelineModel(_)).map { pm => getMatchingPairs(pm.get, dataset) }

    val unitedMatchPairs = matchingPairs.reduce(_ union _).coalesce(getNumberOfPartitions)

    val matchPairsWithWeight = matchPairsWithWeights(unitedMatchPairs.as[Match], getDisagreementWeights).toDF("__key", weightsColumn)
    matchPairsWithWeight.select($"__key._1".alias("idA"), $"__key._2".alias("idB"), col(weightsColumn)).withColumn(getScoreColumnName, UDF.score(col(weightsColumn)))
    
    matchPairsWithWeight.filter(col(getScoreColumnName) > getScoreThreshold)
  }

  /**
   * sort within partitions
   * reduce to pick the highest agreement weight if multiple matches of the same match type
   * TODO add custom partitioner
   * TODO repartition
   * TODO persist
   */
  def matchPairsWithWeights(
    matchPairs:                                      Dataset[Match],
    matchTypeWeights:                                Map[String, Double],
    multipleMatchWithSameMatchTypeReductionConstant: Double              = 0.5): RDD[((Long, Long), Array[Double])] = {

    val matchPairsRDD = mapToKeyValuePairs(matchPairs)

    // we ignore the distance, handle duplicate matches of same type
    val reducedToAggrementByMatchTypeScore = matchPairsRDD.reduceByKey((a, b) => // TODO
      if (a._2 > b._2) (a._1, a._2 + b._2 * multipleMatchWithSameMatchTypeReductionConstant)
      else (a._1, b._2 + a._2 * multipleMatchWithSameMatchTypeReductionConstant))

    def sortedDisagreementWeights = (for (mt <- matchTypeWeights.keys.toList.sorted) yield (matchTypeWeights.get(mt).get)).toArray

    val weightedMatchPairs = reducedToAggrementByMatchTypeScore.mapPartitions(rows => {

      val matchPairWeights = new scala.collection.mutable.HashMap[(Long, Long), Array[Double]]()

      def putToMap(key: (Long, Long), mtId: Byte, agreeWeight: Double) = { matchPairWeights.getOrElseUpdate(key, sortedDisagreementWeights)(mtId) = agreeWeight }
      rows.foreach { r =>
        {
          val (idA, idB, matchTypeId) = r._1;
          val (distance, agreeWeight) = r._2
          putToMap((idA, idB), matchTypeId, agreeWeight)
        }
      }
      matchPairWeights.toIterator
    }, preservesPartitioning = true)
    weightedMatchPairs
  }

  protected def mapToKeyValuePairs(matchPairs: Dataset[Match]): RDD[((Long, Long, Byte), (Double, Double))] = {
    matchPairs.rdd.map {
      case matchPair =>
        val key = (matchPair.idA, matchPair.idB, matchPair.matchType)
        val value = (matchPair.distance, matchPair.weight)
        (key, value)
    }
  }
}

object MatchPairCombiner extends DefaultParamsReadable[MatchPairCombiner] {

  override def load(path: String): MatchPairCombiner = super.load(path)
  def apply(scoreThreshold: Double, partitionSize: Int) = {
    new MatchPairCombiner().setNumberOfPartitions(partitionSize)
  }
}

class TransitiveClusterer(override val uid: String) extends Transformer  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("TransitiveClusterer"))

  val clusterColumnName = new  org.apache.spark.ml.param.Param[String](this, "clusterColumnName", "default is \"cluster\"")
  setDefault(clusterColumnName, "cluster")
  def getClusterColumnName: String = $(clusterColumnName)
  def setClusterColumnName(value: String): this.type = set(clusterColumnName, value)

  val patientVertice = new  org.apache.spark.ml.param.Param[DataFrame](this, "patientVertice", "patient data frame with identifier")
  def getPatientVertice: DataFrame = $(patientVertice)
  def setPatientVertice(value: DataFrame): this.type = set(patientVertice, value)

  val tempFolderPath = new  org.apache.spark.ml.param.Param[String](this, "tempFolderPath", "default is \"/temp/Identity\"")
  setDefault(tempFolderPath, "/temp/identity")
  def getTempFolderPath: String = $(tempFolderPath)
  def setTempFolderPath(value: String): this.type = set(tempFolderPath, value)
  
  private val identifierColumnName = new  org.apache.spark.ml.param.Param[String](this, "identifierColumnName", "default is \"id\"")
  setDefault(identifierColumnName, "id")
  def getIdentifierColumnName: String = $(identifierColumnName)
  def setIdentifierColumnName(value: String): this.type = set(identifierColumnName, value)


    // Members declared in org.apache.spark.ml.Transformer
  override def copy(extra: org.apache.spark.ml.param.ParamMap): org.apache.spark.ml.Transformer = defaultCopy(extra)

  
  override def transformSchema(schema: org.apache.spark.sql.types.StructType): org.apache.spark.sql.types.StructType = {
    require(getPatientVertice.schema.fieldNames.contains(getIdentifierColumnName), s"Patient dataframe must have $getIdentifierColumnName")
    var newSchema = schema.add(org.apache.spark.sql.types.StructField(getClusterColumnName, org.apache.spark.sql.types.StringType, true))
    for (sf<-getPatientVertice.schema.fields) {if (!schema.fieldNames.contains(sf.name)) newSchema = newSchema.add(sf)}
    newSchema
  }

  case class Match(idA: Long, idB: Long, matchType: Byte, distance: Double, weight: Double)

  def transform(dataset: org.apache.spark.sql.Dataset[_]): org.apache.spark.sql.DataFrame = {
      dataset.sparkSession.sparkContext.setCheckpointDir(getTempFolderPath)
      val edgesRenamed = dataset.withColumnRenamed("idA", "src").withColumnRenamed("idB", "dst")
      val g = GraphFrame(getPatientVertice, edgesRenamed)
      val result = g.connectedComponents.run()
      result.withColumnRenamed("component", getClusterColumnName)
  }
}

object TransitiveClusterer extends DefaultParamsReadable[TransitiveClusterer] {

  override def load(path: String): TransitiveClusterer = super.load(path)

  def apply(patient:Dataset[_]) = {
    new TransitiveClusterer().setPatientVertice(patient.toDF())
  }
}

