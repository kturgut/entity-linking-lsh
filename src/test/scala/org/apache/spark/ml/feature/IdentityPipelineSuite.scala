package org.apache.spark.ml.feature

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{ Dataset, Row }
import org.apache.spark.ml.{ Pipeline, PipelineModel }
//import org.apache.spark.ml.feature.MinHashLSH
//import org.apache.log4j._
import org.apache.spark.ml.param.{ Params, ParamMap, ParamValidators }

class IdentityPipelineIntegrationSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  override def beforeAll() = {
    super.beforeAll()
    sc.setLogLevel("ERROR")
  }
  
  test("original raw")  {
    import org.apache.spark.sql.functions._
    import org.apache.spark.sql.{ DataFrame, Dataset, Row }

      val aaaSantara = Map("1.2.840.114350.1.13.163.2.7.3.688884.100"->"1", "2.16.840.1.113883.3.564.610"->"2","2.16.840.1.113883.3.564.1754"->"3")
    
    def zip4(aaa:Map[String,String]) = udf((x: Seq[String], y: Seq[String], z: Seq[String]) => (x, y, z).zipped.toList.filter(a => aaa.contains(a._1)).map {
        case (a: String, b: String, c: String) => aaa.get(a).get ++ c ++ b
        case (a: String, b: String, null)      => aaa.get(a).get ++ b
        case (_)                               => "BAD_ID"
    })  
   

    val rawSentaraPatientsURL = "adl://finadls01cuuklmu6bqqfu.azuredatalakestore.net/data/identityPOC/Sentara_Repartitioned20171218"
    val labeledSentaraPatientsURL = "adl://finadls01cuuklmu6bqqfu.azuredatalakestore.net/data/identityPOC/Sentara/Sentara_Labeled_Repartitioned20171229"
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")

//    val patients = spark.read.parquet(rawSentaraPatientsURL)
//    val patients = spark.read.parquet("adl://finadls01cuuklmu6bqqfu.azuredatalakestore.net/data/identityPOC/Sentara_Repartitioned20171218")

    val p = patients.select(
    //  $"syntheticId",    
      col("id"),
      col("repoId"),
      col("resourceId"),
      col("identifier"))

p.show      


    val pId0 = p.filter("identifier is not null").select("id","identifier.system","identifier.value","identifier.type.text").withColumn("identifiers", zip4(aaaSantara )(col("system"), col("value"), col("text"))).drop("system", "value", "text")

    // if too many nulls, better to recreate the df after filter -- TODO verify the performance    
    val pIdSchema = pId0.schema
    val filteredPId = pId0.rdd.filter(!_.getList[String](1).isEmpty)
    val pId = spark.sqlContext.createDataFrame(filteredPId,pIdSchema).cache
    
pId.show

    def idHasher: HashingTF           = new HashingTF().setInputCol("identifiers").setOutputCol("rawIdentifierFeatures").setNumFeatures(1048576)
    
    def fitLSHModel(featurizedPatients: DataFrame, numberOfHashTables: Integer, inputColumn: String, outputColumn: String): MinHashLSHModel = {
        val mh = new org.apache.spark.ml.feature.MinHashLSH().setNumHashTables(numberOfHashTables).setInputCol(inputColumn).setOutputCol(outputColumn)
        val fittedModel = mh.fit(featurizedPatients)
        fittedModel
    }
  
    def findSimilarRecords(model:MinHashLSHModel,datasetA: Dataset[_],datasetB: Dataset[_],threshold:Double, distanceColumn:String="JacardDistance", idColumn:String="id") = {
        model.approxSimilarityJoin(datasetA, datasetB, threshold, distanceColumn)
            .select(col(s"datasetA.$idColumn").alias("idA"), col(s"datasetB.$idColumn").alias("idB"), col(distanceColumn))
    }
    
    
  
    val idFeaturizedP         = idHasher.transform(pId)
println("IdFeaturizedP")
idFeaturizedP.show
    val identiferMatchingModel=  fitLSHModel(idFeaturizedP        ,3,"rawIdentifierFeatures" , "idHashes")
    
  
     
     
    val idHashedP             = identiferMatchingModel.transform(idFeaturizedP)
idHashedP.show    
    val idMatches             = findSimilarRecords(identiferMatchingModel,idHashedP,idHashedP, 0.01)

    idMatches.show()

  }

//    test("shingle identifier raw") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "Identifier"
//  
//      val cs = PatientColumnSelector(matchType).setIdentifierColumnName("id")
//        .setKeepLabelColumn(false)
//  
//      val sentaraAAA = Set("1.2.840.114350.1.13.163.2.7.3.688884.100", "2.16.840.1.113883.3.564.610", "2.16.840.1.113883.3.564.1754")
//      val ps = PatientShingler(matchType).setMatchType(matchType).setAAA(sentaraAAA.toArray)
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val shingledPatients = fittedPipelineModel.transform(patients)
//  
//      shingledPatients.printSchema
//      shingledPatients.show()
//      assert(!shingledPatients.select("id","identifierShingles").collect().isEmpty, "identifiers should not have been empty")
//  
//    }
//  
//    test("shingle address raw") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//      val defaultStopWords = Set("ln", "st", "street", "pl", "place", "ct", "court", "loop", "crt", "apt", "way", "rd", "w", "e", "n", "s", "cir", "circle", "av.", "ave", "ave.", "dr", "blvd", "road", "rd", "po", "box")
//  
//      val matchType = "Address"
//  
//      val cs = PatientColumnSelector(matchType).setKeepLabelColumn(false)
//  
//      val ps = PatientShingler(matchType).setStopWords(defaultStopWords.toArray)
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val shingledPatients = fittedPipelineModel.transform(patients)
//  
//      shingledPatients.printSchema
//      shingledPatients.show()
//      assert(!shingledPatients.select("id","addressShingles").collect().isEmpty, "addresses should not have been empty")
//    }
//  
//    test("shingle givenName raw") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "GivenName"
//  
//      val cs = PatientColumnSelector(matchType).setKeepLabelColumn(false)
//  
//      val ps = PatientShingler(matchType)
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val shingledPatients = fittedPipelineModel.transform(patients)
//  
//      shingledPatients.printSchema
//      shingledPatients.show()
//      assert(!shingledPatients.select("givenNameShingles").collect().isEmpty, "givenNames should not have been empty")
//    }
//  
//    test("shingle familyName raw") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "FamilyName"
//  
//      val cs = PatientColumnSelector(matchType).setKeepLabelColumn(false)
//  
//      val ps = PatientShingler(matchType)
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val shingledPatients = fittedPipelineModel.transform(patients)
//  
//      shingledPatients.printSchema
//      shingledPatients.show()
//      assert(!shingledPatients.select("familyNameShingles").collect().isEmpty, "familyNames should not have been empty")
//    }
//  
//    test("shingle birthDateGender raw") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "BirthDateGender"
//  
//      val cs = PatientColumnSelector(matchType).setKeepLabelColumn(false)
//  
//      val ps = PatientShingler(matchType)
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val shingledPatients = fittedPipelineModel.transform(patients)
//  
//      shingledPatients.printSchema
//      shingledPatients.show()
//      assert(!shingledPatients.select("birthDateGenderShingles").collect().isEmpty, "birthdateGender should not have been empty")
//    }
//  
//    test("shingle Address raw") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "Address"
//  
//      val cs = PatientColumnSelector(matchType).setKeepLabelColumn(false)
//  
//      val ps = PatientShingler(matchType)
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val shingledPatients = fittedPipelineModel.transform(patients)
//  
//      shingledPatients.printSchema
//      shingledPatients.show()
//      assert(!shingledPatients.select("addressShingles").collect().isEmpty, "addressShingles should not have been empty")
//    }
//  
//    test("identity minhash") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "Identifier"
//  
//       val cs = PatientColumnSelector(matchType)
//  
//      val sentaraAAA = Set("1.2.840.114350.1.13.163.2.7.3.688884.100", "2.16.840.1.113883.3.564.610", "2.16.840.1.113883.3.564.1754")
//      val ps = PatientShingler(matchType).setAAA(sentaraAAA.toArray)
//  
//      val ht = new HashingTF().setInputCol("identifierShingles").setOutputCol("rawIdentifierFeatures").setNumFeatures(1000000)
//      val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawIdentifierFeatures").setOutputCol("idHashes")
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val minHashedPatients = fittedPipelineModel.transform(patients)
//  
//      minHashedPatients.printSchema
//      minHashedPatients.show()
//      assert(!minHashedPatients.select("id", "identifierShingles", "idHashes").collect().isEmpty, "idHashes should not have been empty")
//    }
//  
//  
//  
//    test("givenName minhash") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "GivenName"
//  
//       val cs = PatientColumnSelector(matchType)
//  
//      val ps = PatientShingler(matchType)
//  
//      val ht = new HashingTF().setInputCol("givenNameShingles").setOutputCol("rawGivenNameFeatures").setNumFeatures(1000000)
//      val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawGivenNameFeatures").setOutputCol("givenNameHashes")
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val minHashedPatients = fittedPipelineModel.transform(patients)
//  
//      minHashedPatients.printSchema
//      minHashedPatients.show()
//      assert(!minHashedPatients.select("id", "givenNameShingles", "givenNameHashes").collect().isEmpty, "givenNameHashes should not have been empty")
//    }
//  
//    test("familyName minhash") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "FamilyName"
//  
//      val cs = PatientColumnSelector(matchType)
//  
//      val ps = PatientShingler(matchType)
//  
//      val ht = new HashingTF().setInputCol("familyNameShingles").setOutputCol("rawFamilyNameFeatures").setNumFeatures(1000000)
//      val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawFamilyNameFeatures").setOutputCol("familyNameHashes")
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val minHashedPatients = fittedPipelineModel.transform(patients)
//  
//      minHashedPatients.printSchema
//      minHashedPatients.show()
//      assert(!minHashedPatients.select("id", "familyNameShingles", "familyNameHashes").collect().isEmpty, "familyNameHashes should not have been empty")
//    }
//  
//    ////        case MatchType.Address         => createHasherAndLSH("addressShingles", "rawAddressFeatures", "addressHashes", config.matchConfigs.get(matchType).get) // 268435456
//  
//    test("address minhash") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "Address"
//  
//      val cs = PatientColumnSelector(matchType)
//  
//      val ps = PatientShingler(matchType)
//  
//      val ht = new HashingTF().setInputCol("addressShingles").setOutputCol("rawAddressFeatures").setNumFeatures(1000000)
//      val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawAddressFeatures").setOutputCol("addressHashes")
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val minHashedPatients = fittedPipelineModel.transform(patients)
//  
//      minHashedPatients.printSchema
//      minHashedPatients.show()
//      assert(!minHashedPatients.select("id", "addressShingles", "addressHashes").collect().isEmpty, "addressHashes should not have been empty")
//    }
//  
//    test("birthdate gender minhash") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "BirthDateGender"
//  
//      val cs = PatientColumnSelector(matchType)
//  
//      val ps = PatientShingler(matchType)
//  
//      //    "identifiers", "rawIdentifierFeatures", "idHashes"
//      val ht = new HashingTF().setInputCol("birthDateGenderShingles").setOutputCol("rawBirthDateGenderFeatures").setNumFeatures(1000000)
//      val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawBirthDateGenderFeatures").setOutputCol("birthDateGenderHashes")
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//      val minHashedPatients = fittedPipelineModel.transform(patients)
//  
//      minHashedPatients.printSchema
//      minHashedPatients.show()
//      assert(!minHashedPatients.select("id", "birthdateGenderShingles", "birthDateGenderHashes").collect().isEmpty, "birthDateGenderHashes should not have been empty")
//    }

//  test("identifier minhash lsh") {
//    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//
//    val matchType = "Identifier"
//
//    val cs = PatientColumnSelector(matchType)
//
//    val sentaraAAA = Set("1.2.840.114350.1.13.163.2.7.3.688884.100", "2.16.840.1.113883.3.564.610", "2.16.840.1.113883.3.564.1754")
//    val ps = PatientShingler(matchType).setAAA(sentaraAAA.toArray)
//
//    val ht = new HashingTF().setInputCol("identifierShingles").setOutputCol("rawIdentifierFeatures").setNumFeatures(1048576)
//
//    val pipeline = new Pipeline().setStages(Array(cs, ps, ht))
//
//    val fittedPipelineModel = pipeline.fit(patients)
//
////    val lshModel = fittedPipelineModel.stages.find(_.isInstanceOf[MinHashLSHModel]).get.asInstanceOf[MinHashLSHModel]
////    println("LSH Params")
////    lshModel.explainParams()
//
////    println("Pipeline Params")
////    fittedPipelineModel.explainParams();
////    val paramMap = ParamMap(sp.lshModel -> lshModel)
//
//    val featurizedPatients = fittedPipelineModel.transform(patients)
//
//    println("FeaturizedPatients")
//    featurizedPatients.show
//    
//    val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawIdentifierFeatures").setOutputCol("idHashes")
//
//    val fittedLSH = mh.fit(featurizedPatients)
//    val minHashedPatients = fittedLSH.transform(featurizedPatients)
//    println("MinHashedPatients")
//    minHashedPatients.show
//    
//    val pairs = fittedLSH.approxSimilarityJoin(minHashedPatients, minHashedPatients, 0, "distance")
//    println("Pairs Manual")
//    pairs.show
//
//    
//    val sp = SimilarPairFinder(matchType, 0, 4, -1)
//    sp.setLshModel(fittedLSH)
//    val similarPatients = sp.transform(minHashedPatients)
//
//    similarPatients.printSchema
//    similarPatients.show()
//    assert(!similarPatients.select("idA", "idB", "distance", "matchType", "weight").collect().isEmpty, "similirPairs beased on ID should not have been empty")
//  }
  
  

//    test("givenName minhash lsh") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "GivenName"
//  
//       val cs = PatientColumnSelector(matchType)
//  
//      val ps = PatientShingler(matchType)
//  
//      val ht = new HashingTF().setInputCol("givenNameShingles").setOutputCol("rawGivenNameFeatures").setNumFeatures(1000000)
//      val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawGivenNameFeatures").setOutputCol("givenNameHashes")
//  
//      val sp = SimilarPairFinder(matchType,0,4, -1)
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh, sp))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//  
//      val lshModel = fittedPipelineModel.stages.find(_.isInstanceOf[MinHashLSHModel]).get.asInstanceOf[MinHashLSHModel]
//      val paramMap = ParamMap(sp.lshModel -> lshModel)
//  
//      val similarPatients = fittedPipelineModel.transform(patients,paramMap)
//  
//      similarPatients.printSchema
//      similarPatients.show()
//      assert(!similarPatients.select("idA", "idB", "distance", "matchType", "weight").collect().isEmpty, "similar pairs based on Given name should not have been empty")
//    }
//  
//    test("familyName minhash lsh") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "FamilyName"
//  
//       val cs = PatientColumnSelector(matchType)
//  
//      val ps = PatientShingler(matchType)
//  
//      val ht = new HashingTF().setInputCol("familyNameShingles").setOutputCol("rawFamilyNameFeatures").setNumFeatures(1000000)
//      val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawFamilyNameFeatures").setOutputCol("familyNameHashes")
//  
//      val sp = SimilarPairFinder(matchType,0,4, -1)
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh, sp))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//  
//      val lshModel = fittedPipelineModel.stages.find(_.isInstanceOf[MinHashLSHModel]).get.asInstanceOf[MinHashLSHModel]
//      val paramMap = ParamMap(sp.lshModel -> lshModel)
//  
//      val similarPatients = fittedPipelineModel.transform(patients,paramMap)
//  
//      similarPatients.printSchema
//      similarPatients.show()
//      assert(!similarPatients.select("idA", "idB", "distance", "matchType", "weight").collect().isEmpty, "similar pairs based on Family name should not have been empty")
//    }
//  
//    test("address minhash lsh") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "Address"
//  
//      val defaultStopWords = Set("ln", "st", "street", "pl", "place", "ct", "court", "loop", "crt", "apt", "way", "rd", "w", "e", "n", "s", "cir", "circle", "av.", "ave", "ave.", "dr", "blvd", "road", "rd", "po", "box")
//  
//       val cs = PatientColumnSelector(matchType)
//  
//      val ps = PatientShingler(matchType).setStopWords(defaultStopWords.toArray)
//  
//      val ht = new HashingTF().setInputCol("addressShingles").setOutputCol("rawAddressFeatures").setNumFeatures(1000000)
//      val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawAddressFeatures").setOutputCol("addressHashes")
//  
//      val sp = SimilarPairFinder(matchType,0,4, -1)
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh, sp))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//  
//      val lshModel = fittedPipelineModel.stages.find(_.isInstanceOf[MinHashLSHModel]).get.asInstanceOf[MinHashLSHModel]
//      val paramMap = ParamMap(sp.lshModel -> lshModel)
//  
//      val similarPatients = fittedPipelineModel.transform(patients,paramMap)
//  
//      similarPatients.printSchema
//      similarPatients.show()
//      assert(!similarPatients.select("idA", "idB", "distance", "matchType", "weight").collect().isEmpty, "similar pairs based on Family name should not have been empty")
//    }
//  
//    test("birthDateGender minhash lsh") {
//      val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
//  
//      val matchType = "BirthDateGender"
//  
//  
//       val cs = PatientColumnSelector(matchType)
//  
//      val ps = PatientShingler(matchType)
//      val ht = new HashingTF().setInputCol("birthDateGenderShingles").setOutputCol("rawBirthDateGenderFeatures").setNumFeatures(1000000)
//      val mh = new MinHashLSH().setNumHashTables(1).setInputCol("rawBirthDateGenderFeatures").setOutputCol("birthDateGenderHashes")
//  
//      val sp = SimilarPairFinder(matchType,0,4, -1)
//  
//      val pipeline = new Pipeline().setStages(Array(cs, ps, ht, mh, sp))
//  
//      val fittedPipelineModel = pipeline.fit(patients)
//  
//      val lshModel = fittedPipelineModel.stages.find(_.isInstanceOf[MinHashLSHModel]).get.asInstanceOf[MinHashLSHModel]
//      val paramMap = ParamMap(sp.lshModel -> lshModel)
//  
//      val similarPatients = fittedPipelineModel.transform(patients,paramMap)
//  
//      similarPatients.printSchema
//      similarPatients.show()
//      assert(!similarPatients.select("idA", "idB", "distance", "matchType", "weight").collect().isEmpty, "similar pairs based on Family name should not have been empty")
//    }
//  
//  
//    test("identity Profile") {
//          val patientsURL = "/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId"
//          val profile = Identity.identityProfile(spark, Workbench.SentaraConfig, Some(patientsURL))
//          assert(profile != null, "should have been able to produce identity profile")
//    }

}