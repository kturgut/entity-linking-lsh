package org.apache.spark.ml.feature

import scala.beans.BeanInfo
import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.sql.{ Dataset, Row }
import org.apache.spark.sql.types.StructType




class PatientShinglerSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  import PatientShinglerSuite._
  
  test("params") {
  //  ParamsSuite.checkParams(new PatientShingler)
  }

//  test("read/write") {
//    val t = PatientShingler("Identifier")
//      .setAAA(Array("id1", "id2"))
//      .setStopWords(Array("street", "lane"))
//      .setIdentifierColumnName("id")
//      .setMatchType("Identifier")
//      .setLabelColumnName("syntheticId")
//      .setKeepLabelColumn(true)
//    testDefaultReadWrite(t)
//  }

  test("shingle identity labeled") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val ps = new PatientShingler()
      .setMatchType("Identifier")
      .setIdentifierColumnName("id")
      .setLabelColumnName("syntheticId")
      .setKeepLabelColumn(true)
      .setAAA(Array("aa1","aa2","aa3"))
    val tp = ps.transform(patients)
    expectColumns(tp, Seq("syntheticId", "resourceId", "repoId", "id", "identifierShingles"))
    assert({val id=tp.schema.find(_.name == "identifiers").get; id.dataType.isInstanceOf[StructType]}, "identifier column is expected to be a structure")
  }

  test("shingle identity raw") {
    val patients = spark.read.parquet("/Users/kaganturgut/git/IdentityPOC/com.changehealthcare.identity/src/main/resources/samplePatientsWithId")
    val ps = new PatientShingler()
      .setMatchType("Identifier")
      .setIdentifierColumnName("id")
      .setLabelColumnName("syntheticId")
      .setKeepLabelColumn(false)
      .setAAA(Array("aa1","aa2","aa3"))
    val tp = ps.transform(patients)
    expectColumns(tp, Seq("id", "identifiers"))
    assert({val id=tp.schema.find(_.name == "identifierShingles").get; id.dataType.isInstanceOf[StructType]}, "identifier column is expected to be a structure")
    assert(tp.schema.fieldNames.size==2, s"expecting only 2 columns {id,identifierShingles}, found others ${tp.schema.fieldNames}")
  }
  
  // TODO expand on this, and check data types

}

object PatientShinglerSuite extends SparkFunSuite {

  def expectColumns(dataset: Dataset[_], expectedColumns:Seq[String]): Unit = {
    for (col <- expectedColumns) assert(dataset.schema.fieldNames.contains(col), s"cound not findfind column $col among ${dataset.schema.fieldNames}")
  }
}
