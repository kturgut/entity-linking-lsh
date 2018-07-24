import Dependencies._

lazy val sparkVersion = "2.2.0"


lazy val root = (project in file(".")).
  settings(
    inThisBuild(List(
      organization := "com.changehealthcare",
      scalaVersion := "2.11.8",
      version      := "0.1.0-SNAPSHOT"
    )),
    name := "ChangeHealthcare-LSH",
    
    
    
    libraryDependencies ++= Seq(	
    
		"graphframes"   % "graphframes"  % "0.5.0-spark2.1-s_2.11",
		"org.apache.spark" %% "spark-core" % sparkVersion % "provided" withSources() withJavadoc()  classifier "tests",
	    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided" withSources() withJavadoc() classifier "tests",
	    "org.scalatest" %% "scalatest-funsuite" % "3.0.0-SNAP13"
	    
	) 	
   
   

)