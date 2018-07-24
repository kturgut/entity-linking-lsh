package org.apache.spark.ml.feature

import scala.beans.BeanInfo
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{ Dataset, Row }
import org.apache.spark.sql.types.StructType

@BeanInfo
case class PatientTestData(rawText: String, wantedTokens: Array[String])

class PatientColumnSelectorSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {
  import PatientColumnSelectorSuite._

  test("params") {
    ParamsSuite.checkParams(new PatientColumnSelector)
  }

//  test("read/write") {
//    val t = new PatientColumnSelector()
//      .setIdentifierColumnName("id")
//      .setMatchType("Identifier")
//      .setLabelColumnName("syntheticId")
//      .setKeepLabelColumn(true)
//    testDefaultReadWrite(t)
//  }

  test("select for identity labeled") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")

    patients.select("identifier", "repoId", "resourceId", "id", "name", "birthDate", "gender").printSchema()

    val cs = new PatientColumnSelector()
      .setMatchType("Identifier")
      .setIdentifierColumnName("id")
      .setLabelColumnName("syntheticId")
      .setKeepLabelColumn(true)
    testPatientColumnnSelector(cs, patients, Seq("resourceId", "id", "syntheticId", "repoId", "identifier"))
  }

  test("select for identity raw") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val cs = new PatientColumnSelector()
      .setMatchType("Identifier")
      .setIdentifierColumnName("id")
      .setKeepLabelColumn(false)
    testPatientColumnnSelector(cs, patients, Seq("resourceId", "id", "identifier"))
  }

  test("select for givenName labeled") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val cs = new PatientColumnSelector()
      .setMatchType("GivenName")
      .setLabelColumnName("syntheticId")
      .setKeepLabelColumn(true)
    testPatientColumnnSelector(cs, patients, Seq("resourceId", "id", "syntheticId", "repoId", "name"))
  }

  test("select for givenName raw") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val cs = new PatientColumnSelector()
      .setMatchType("GivenName")
      .setKeepLabelColumn(false)
    testPatientColumnnSelector(cs, patients, Seq("resourceId", "id", "name"))
  }

  test("select for familyName labeled") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val cs = new PatientColumnSelector()
      .setMatchType("FamilyName")
      .setLabelColumnName("syntheticId")
      .setKeepLabelColumn(true)
    testPatientColumnnSelector(cs, patients, Seq("resourceId", "id", "syntheticId", "repoId", "name"))
  }

  test("select for familyName raw") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val cs = new PatientColumnSelector()
      .setMatchType("FamilyName")
      .setKeepLabelColumn(false)
    testPatientColumnnSelector(cs, patients, Seq("resourceId", "id", "name"))
  }

  test("select for Address labeled") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val cs = new PatientColumnSelector()
      .setMatchType("Address")
      .setLabelColumnName("syntheticId")
      .setKeepLabelColumn(true)
    testPatientColumnnSelector(cs, patients, Seq("resourceId", "id", "syntheticId", "repoId", "address"))
  }

  test("select for Address raw") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val cs = new PatientColumnSelector()
      .setMatchType("Address")
      .setKeepLabelColumn(false)
    testPatientColumnnSelector(cs, patients, Seq("resourceId", "id", "address"))
  }

  test("select for BirthDateGender labeled") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val cs = new PatientColumnSelector()
      .setMatchType("BirthDateGender")
      .setLabelColumnName("syntheticId")
      .setKeepLabelColumn(true)
    testPatientColumnnSelector(cs, patients, Seq("resourceId", "id", "syntheticId", "repoId", "address"))
  }

  test("select for BirthDateGender raw") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val cs = new PatientColumnSelector()
      .setMatchType("BirthDateGender")
      .setKeepLabelColumn(false)
    testPatientColumnnSelector(cs, patients, Seq("resourceId", "id", "gender", "birthDate"))
  }

}

object PatientColumnSelectorSuite extends SparkFunSuite {

  def testPatientColumnnSelector(pcs: PatientColumnSelector, dataset: Dataset[_], expectedColumns: Seq[String]): Unit = {
    val schema = pcs.transform(dataset).schema
    for (col <- expectedColumns) assert(schema.fieldNames.contains(col), s"cound not findfind column $col among ${schema.fieldNames}")
    val remainder = schema.filterNot { a => expectedColumns.contains(a.name) }
    assert(remainder.size == 0, s"all columns have not been filtered: ${for (col <- remainder) yield (col.name)}")
  }
}


