package org.apache.spark.ml.feature

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.{ Pipeline, PipelineModel }
import org.apache.log4j._
import org.apache.spark.ml.param.{ Params, ParamMap, ParamValidators }
import org.apache.spark.sql.{ Dataset, DataFrame }

abstract class MatchConfig(
  val name:                        String,
  val transitiveMatch:             Boolean,
  val matchThreshold:              Double,
  val weightCalculationThresholds: Seq[Double],
  val agreementWeight:             Double,
  val disagreementWeight:          Double,
  val sampleSize:                  Long,
  val weightColumnName:            String      = "weight",
  val numberOfFeatures:            Int,
  val numberOfHashTables:          Int         = 1,
  var samplePath:                  String      = "samplePath",
  var modelPath:                   String      = "modelPath") {}

//case class IdentifierConfig(override val transitiveMatch:Boolean, override val matchThreshold: Double, override val weightCalculationThresholds: Seq[Double], override val agreementWeight: Double,override val disagreementWeight: Double, sizeOfSample: Long, override val numberOfFeatures: Int, override val numberOfHashTables: Int, aaa: Set[String]) 
//extends MatchConfig("Identifier", transitiveMatch, matchThreshold, weightCalculationThresholds, agreementWeight, disagreementWeight, sizeOfSample, "weight", numberOfFeatures, numberOfHashTables)

case class IdentifierMatchConfig(override val transitiveMatch: Boolean, override val matchThreshold: Double, override val weightCalculationThresholds: Seq[Double], override val agreementWeight: Double, override val disagreementWeight: Double, override val sampleSize: Long, override val numberOfFeatures: Int, override val numberOfHashTables: Int, val aaa: Set[String])
  extends MatchConfig("Identifier", transitiveMatch, matchThreshold, weightCalculationThresholds, agreementWeight, disagreementWeight, sampleSize, "weight", numberOfFeatures, numberOfHashTables) {}

case class AddressMatchConfig(override val transitiveMatch: Boolean, override val matchThreshold: Double, override val weightCalculationThresholds: Seq[Double], override val agreementWeight: Double, override val disagreementWeight: Double, override val sampleSize: Long, override val numberOfFeatures: Int, override val numberOfHashTables: Int, val stopWords: Set[String])
  extends MatchConfig("Address", transitiveMatch, matchThreshold, weightCalculationThresholds, agreementWeight, disagreementWeight, sampleSize, "weight", numberOfFeatures, numberOfHashTables) {}

case class GivenNameMatchConfig(override val transitiveMatch: Boolean, override val matchThreshold: Double, override val weightCalculationThresholds: Seq[Double], override val agreementWeight: Double, override val disagreementWeight: Double, override val sampleSize: Long, override val numberOfFeatures: Int, override val numberOfHashTables: Int)
  extends MatchConfig("GivenName", transitiveMatch, matchThreshold, weightCalculationThresholds, agreementWeight, disagreementWeight, sampleSize, "weight", numberOfFeatures, numberOfHashTables) {}

case class FamilyNameMatchConfig(override val transitiveMatch: Boolean, override val matchThreshold: Double, override val weightCalculationThresholds: Seq[Double], override val agreementWeight: Double, override val disagreementWeight: Double, override val sampleSize: Long, override val numberOfFeatures: Int, override val numberOfHashTables: Int)
  extends MatchConfig("FamilyName", transitiveMatch, matchThreshold, weightCalculationThresholds,  agreementWeight, disagreementWeight,  sampleSize,   "weight",  numberOfFeatures,  numberOfHashTables) {}

case class BirthDateGenderMatchConfig(override val transitiveMatch: Boolean, override val matchThreshold: Double, override val weightCalculationThresholds: Seq[Double], override val agreementWeight: Double, override val disagreementWeight: Double, override val sampleSize: Long, override val numberOfFeatures: Int, override val numberOfHashTables: Int)
  extends MatchConfig("BirthDateGender", transitiveMatch,  matchThreshold, weightCalculationThresholds, agreementWeight, disagreementWeight, sampleSize, "weight", numberOfFeatures, numberOfHashTables) {}

case class TelecomMatchConfig(override val transitiveMatch: Boolean, override val matchThreshold: Double, override val weightCalculationThresholds: Seq[Double], override val agreementWeight: Double, override val disagreementWeight: Double, override val sampleSize: Long, override val numberOfFeatures: Int, override val numberOfHashTables: Int)
  extends MatchConfig("Telecom", transitiveMatch, matchThreshold, weightCalculationThresholds, agreementWeight, disagreementWeight, sampleSize, "weight", numberOfFeatures, numberOfHashTables) {}


class Configuration(
  val name:               String,
  var rawPatientsURL:     String,
  var labeledPatientsURL: String,
  val scoreThreshold:     Double,
  val matchConfigs:       Set[MatchConfig],
  val baseFolderPath:     String,
  val baseTempFolderPath: String,
  val numberOfPartitions: Int              = 500,
  val labelColumnName:    String           = "syntheticId",
  val clusterColumnName:  String           = "clusterId",
  val weightsColumnName:  String           = "weights",
  val scoreColumnName:    String           = "score") {

  @transient lazy val modelFolderPath = s"${baseFolderPath}/models/$name"
  @transient lazy val pipelineFolderPath = s"${baseFolderPath}/pipelines/$name"
  @transient lazy val sampleFolderPath = s"${baseFolderPath}/samples/$name"
  @transient lazy val weightsPath = s"${baseFolderPath}/config/$name/matchConfigurations"
  @transient lazy val tempFolderPath = s"${baseTempFolderPath}/$name"

  def getMatchConfig(matchType: String): Option[MatchConfig] = this.matchConfigs.find(_.name == matchType)
  def modelPath(matchType: String) = s"$modelFolderPath/${matchType}_model"
  def pipelinePath(matchType: String) = s"$pipelineFolderPath/${matchType}_pipeline"
  def samplePatientsPath(name: String) = { val today = java.util.Calendar.getInstance.getTime; val f = new java.text.SimpleDateFormat("YY-MM-dd-hhmmss"); s"$sampleFolderPath/${name}${f.format(java.util.Calendar.getInstance().getTime())}" }

  for (mc <- matchConfigs) { mc.modelPath = modelPath(mc.name); mc.samplePath = samplePatientsPath(mc.name) }
}

object Configuration {

  var current: Configuration = _

  var baseFolderPath = "adl://finadls01cuuklmu6bqqfu.azuredatalakestore.net/data/identityPOC"
  var baseTempFolderPath = "/data/IdentityPOC/temp"

  var defaultRawPatientsURL = "adl://finadls01cuuklmu6bqqfu.azuredatalakestore.net/data/identityPOC/Sentara/Sentara_RawAndProfilingOrderedByBirthDate20180116P1500Repartition/Patient.parquet"
  var defaultLabeledPatientsURL = "adl://finadls01cuuklmu6bqqfu.azuredatalakestore.net/data/identityPOC/Sentara/Sentara_RawAndProfilingOrderedByBirthDate20180116P1500Repartition/Patient.parquet"

  val configurations = scala.collection.mutable.Map[String, Configuration]()

  def apply(
    name:               String,
    scoreThreshold:     Double,
    matchConfigs:       Set[MatchConfig],
    baseFolderPath:     String,
    baseTempFolderPath: String,
    numberOfPartitions: Int  ): Configuration = {
    apply(name, defaultRawPatientsURL, defaultLabeledPatientsURL, scoreThreshold, matchConfigs, baseFolderPath, baseTempFolderPath, numberOfPartitions, "syntheticId", "clusterId", "weights", "score")
  }

  def apply(
    name:               String,
    rawPatientsURL:     String,
    labeledPatientsURL: String,
    scoreThreshold:     Double,
    matchConfigs:       Set[MatchConfig],
    baseFolderPath:     String,
    baseTempFolderPath: String,
    numberOfPartitions: Int,
    labelColumnName:    String           = "syntheticId",
    clusterColumnName:  String           = "clusterId",
    weightsColumnName:  String           = "weights",
    scoreColumnName:    String           = "score"): Configuration = {

    val config = new Configuration(name, rawPatientsURL, labeledPatientsURL, scoreThreshold, matchConfigs, baseFolderPath, baseTempFolderPath, numberOfPartitions, labelColumnName, clusterColumnName, weightsColumnName, scoreColumnName)
    if (current == null) current = config
    configurations += (name -> config)
    config
  }

}

object Identity {

  /**
   * run identity profile
   */
  def identityProfile(spark: SparkSession, config: Configuration, patientFileURL: Option[String] = None): DataFrame = {
    val patients = spark.read.parquet(patientFileURL.getOrElse(config.rawPatientsURL))
    val matchPairCombiner = MatchPairCombiner(config.scoreThreshold, config.numberOfPartitions).setMatchTypes(config.matchConfigs.map(_.name).toArray)
    for (mc <- config.matchConfigs) matchPairCombiner.dependsOn(mc.name, getMatchingPipelineModel(mc, patients))

    val patientVertice = new PatientShingler("Identifier").setKeepLabelColumn(true).transform(patients)
    val transitiveClusterer = TransitiveClusterer(patientVertice)

    val pipeline = new Pipeline().setStages(Array(matchPairCombiner, transitiveClusterer))
    val fittedPipeline = pipeline.fit(patients)
    fittedPipeline.transform(patients)
  }

  /**
   * train and save models
   */
  def train(spark: SparkSession, configuration: Configuration, patientFileURL: Option[String] = None): PipelineModel = {
    val patients = spark.read.parquet(patientFileURL.getOrElse(configuration.rawPatientsURL))
    val matchPairCombiner = new MatchPairCombiner()
    for (mc <- configuration.matchConfigs) matchPairCombiner.dependsOn(mc.name, getMatchingPipelineModel(mc, patients, true))

    val patientVertice = new PatientShingler("Identifier").setKeepLabelColumn(true).transform(patients)
    val transitiveClusterer = TransitiveClusterer(patientVertice)

    val pipeline = new Pipeline().setStages(Array(matchPairCombiner, transitiveClusterer))
    pipeline.fit(patients)
  }

  /**
   * get matchingPipelineModel. First try to load trained models, if not found, create new pipeline and train it.
   */
  def getMatchingPipelineModel(matchConfig: MatchConfig, patients: Dataset[_], save: Boolean = false): PipelineModel = loadPipelineModel(matchConfig.modelPath).getOrElse(createMatchingPipeline(matchConfig, patients, save).get)

  def loadPipelineModel(path: String): Option[PipelineModel] = {
    try {
      Some(PipelineModel.load(path))
    } catch {
      case ioe: java.io.IOException => None
      case e: Exception             => e.printStackTrace(); None
    }
  }

  /**
   * Create new pipeline model for matchConfig
   */
  def createMatchingPipeline(matchConfig: MatchConfig, patients: Dataset[_], save: Boolean = true): Option[PipelineModel] = {

    val pipelineModel = matchConfig match {
      case IdentifierMatchConfig(transitiveMatch, matchThreshold, weightCalculationThresholds, aggreementWeight, disagreementWeight, sampleSize, numberOfFeatures, numberOfHashTables, aaa) => {

        val matchType = "Identifier"
        val cs = PatientColumnSelector(matchType)
        val ps = PatientShingler(matchType).setAAA(aaa.toArray)
        val ht = new HashingTF().setInputCol("identifierShingles").setOutputCol("rawIdentifierFeatures").setNumFeatures(numberOfFeatures)
        val mh = new MinHashLSH().setNumHashTables(numberOfHashTables).setInputCol("rawIdentifierFeatures").setOutputCol("idHashes")
        val sp = SimilarPairFinder(matchType, matchThreshold, aggreementWeight, disagreementWeight)
        val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh, sp))
        pipeline.fit(patients)
      }

      case AddressMatchConfig(transitiveMatch, matchThreshold, weightCalculationThresholds, aggreementWeight, disagreementWeight, sampleSize, numberOfFeatures, numberOfHashTables, stopWords) => {

        val matchType = "Address"
        val cs = PatientColumnSelector(matchType)
        val ps = PatientShingler(matchType).setStopWords(stopWords.toArray)
        val ht = new HashingTF().setInputCol("addressShingles").setOutputCol("rawAddressFeatures").setNumFeatures(numberOfFeatures)
        val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawAddressFeatures").setOutputCol("addressHashes")
        val sp = SimilarPairFinder(matchType, matchThreshold, aggreementWeight, disagreementWeight)
        val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh, sp))
        pipeline.fit(patients)
      }

      case GivenNameMatchConfig(transitiveMatch, matchThreshold, weightCalculationThresholds, aggreementWeight, disagreementWeight, sampleSize, numberOfFeatures, numberOfHashTables) => {

        val matchType = "GivenName"
        val cs = PatientColumnSelector(matchType)
        val ps = PatientShingler(matchType)

        val ht = new HashingTF().setInputCol("givenNameShingles").setOutputCol("rawGivenNameFeatures").setNumFeatures(numberOfFeatures)
        val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawGivenNameFeatures").setOutputCol("givenNameHashes")
        val sp = SimilarPairFinder(matchType, matchThreshold, aggreementWeight, disagreementWeight)
        val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh, sp))
        pipeline.fit(patients)
      }

      case FamilyNameMatchConfig(transitiveMatch, matchThreshold, weightCalculationThresholds, aggreementWeight, disagreementWeight, sampleSize, numberOfFeatures, numberOfHashTables) => {

        val matchType = "FamilyName"
        val cs = PatientColumnSelector(matchType)
        val ps = PatientShingler(matchType)

        val ht = new HashingTF().setInputCol("familyNameShingles").setOutputCol("rawFamilyNameFeatures").setNumFeatures(numberOfFeatures)
        val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawFamilyNameFeatures").setOutputCol("familyNameHashes")
        val sp = SimilarPairFinder(matchType, matchThreshold, aggreementWeight, disagreementWeight)
        val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh, sp))
        pipeline.fit(patients)
      }
      case BirthDateGenderMatchConfig(transitiveMatch, matchThreshold, weightCalculationThresholds, aggreementWeight, disagreementWeight, sampleSize, numberOfFeatures, numberOfHashTables) => {

        val matchType = "BirthDateGender"
        val cs = PatientColumnSelector(matchType)

        val ps = PatientShingler(matchType)
        val ht = new HashingTF().setInputCol("birthDateGenderShingles").setOutputCol("rawBirthDateGenderFeatures").setNumFeatures(numberOfFeatures)
        val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawBirthDateGenderFeatures").setOutputCol("birthDateGenderHashes")
        val sp = SimilarPairFinder(matchType, matchThreshold, aggreementWeight, disagreementWeight)
        val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh, sp))
        pipeline.fit(patients)
      }
      case TelecomMatchConfig(transitiveMatch, matchThreshold, weightCalculationThresholds, aggreementWeight, disagreementWeight, sampleSize, numberOfFeatures, numberOfHashTables) => {

        val matchType = "Telecom"
        val cs = PatientColumnSelector(matchType)
        val ps = PatientShingler(matchType)

        val ht = new HashingTF().setInputCol("telecomShingles").setOutputCol("rawTelecomFeatures").setNumFeatures(numberOfFeatures)
        val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawTelecomFeatures").setOutputCol("telecomHashes")
        val sp = SimilarPairFinder(matchType, matchThreshold, aggreementWeight, disagreementWeight)
        val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh, sp))
        pipeline.fit(patients)
      }
      
      case _ => throw new IllegalArgumentException(s"Unsupported MatchConfig: $matchConfig")
    }
    if (save) pipelineModel.write.overwrite().save(matchConfig.modelPath)
    Some(pipelineModel)
  }

  /**
   * Helper method to find matching pairs for a match type
   */
  def findMatchingPairs(spark: SparkSession, matchConfig: MatchConfig, patientFileURL: String) = {
    val patients = spark.read.parquet(patientFileURL)
    val pipelineModel = getMatchingPipelineModel(matchConfig, patients)
    val lshModel = pipelineModel.stages.find(_.isInstanceOf[MinHashLSHModel]).get.asInstanceOf[MinHashLSHModel]
    val sp = pipelineModel.stages.find(_.isInstanceOf[SimilarPairFinder]).get.asInstanceOf[SimilarPairFinder]
    val paramMap = ParamMap(sp.lshModel -> lshModel)
    pipelineModel.transform(patients, paramMap)
  }
}

object Workbench {

  
  val stopWords = Set("ln", "st", "street", "pl", "place", "ct", "court", "loop", "crt", "apt", "way", "rd", "w", "e", "n", "s", "cir", "circle", "av.", "ave", "ave.", "dr", "blvd", "road", "rd", "po", "box")
  val defaultAAA = Set("1.2.840.114350.1.13.163.2.7.3.688884.100", "2.16.840.1.113883.3.564.610", "2.16.840.1.113883.3.564.1754")

  val smc: Set[MatchConfig] = Set(
    IdentifierMatchConfig(true, 0.1, Seq(0.1, 0.2, 0.3), 15.3, (-0.22), 1048576, 1073741824, 1, defaultAAA),
    GivenNameMatchConfig(false, 0.3, Seq(0.1, 0.2, 0.3), 11.8, (-0.363), 1048576, 1073741824, 2),
    FamilyNameMatchConfig(false, 0.3, Seq(0.1, 0.2, 0.3), 10.37, (-0.12), 1048576, 1073741824, 2),
    BirthDateGenderMatchConfig(false, 0.01, Seq(0.01, 0.1, 0.5), 13.59, (-2.09), 1048576, 1073741824, 2),
    AddressMatchConfig(false, 0.3, Seq(0.1, 0.2, 0.3), 16.3, (-4.11), 1048576, 1073741824, 3, stopWords))

  /**
   * Sample configuration
   */
  val SentaraConfig = Configuration("Sentara", 12, smc, Configuration.baseFolderPath, Configuration.baseTempFolderPath, 500)

  /**
   * main entry to identity profile
   */
  def identity(spark: SparkSession, config: Configuration) = Identity.identityProfile(spark, config)

  /**
   * Train models for the current configuration
   */
  def train(spark: SparkSession, config: Configuration) = Identity.train(spark, config)

}