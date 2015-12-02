/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.github.cloudml.zen.examples.ml

import java.nio.charset.Charset

import breeze.linalg.{SparseVector => BSV}
import com.github.cloudml.zen.ml.recommendation.{FMRegression, FMModel, FMClassification, FM}
import com.github.cloudml.zen.ml.util.SparkHacker
import com.google.common.io.Files
import org.apache.spark.graphx.GraphXUtils
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkConf, SparkContext}
import scopt.OptionParser

object MovieLensFM extends Logging {

  case class Params(
    input: String = null,
    out: String = null,
    confPath: String = null,
    numPartitions: Int = -1,
    useAdaGrad: Boolean = false,
    kryo: Boolean = true) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("FM") {
      head("MovieLensFM: an example app for FM.")
      opt[Int]("numPartitions")
        .text(s"number of partitions, default: ${defaultParams.numPartitions}")
        .action((x, c) => c.copy(numPartitions = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[String]("confPath")
        .text(s"conf file Path, default: ${defaultParams.confPath}")
        .action((x, c) => c.copy(confPath = x))
      opt[Unit]("adagrad")
        .text("use AdaGrad")
        .action((_, c) => c.copy(useAdaGrad = true))
      arg[String]("<input>")
        .required()
        .text("input paths")
        .action((x, c) => c.copy(input = x))
      arg[String]("<out>")
        .required()
        .text("out paths (model)")
        .action((x, c) => c.copy(out = x))
      note(
        """
          | For example, the following command runs this app on a synthetic dataset:
          |
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.MovieLensFM \
          | examples/target/scala-*/zen-examples-*.jar \
          | --confPath conf/MovieLensFM.txt --kryo \
          | data/mllib/sample_movielens_data.txt
          | data/mllib/fm_model
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val Params(input, out, confPath, numPartitions, useAdaGrad, kryo) = params
    val storageLevel = StorageLevel.MEMORY_AND_DISK
    val conf = new SparkConf().setAppName(s"FM with $params")
    if (kryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    val checkpointDir = s"$out/checkpoint"
    sc.setCheckpointDir(checkpointDir)
    SparkHacker.gcCleaner(60 * 10, 60 * 10, "MovieLensFM")
    val (trainSet, testSet, validationSet, _) = MovieLensUtils.crossValidation(sc,
      input, numPartitions, storageLevel)
    import scala.collection.JavaConversions._
    val lines = Files.readLines(new java.io.File(confPath), Charset.defaultCharset())
    lines.filter(l => !l.startsWith("#")).foreach { line =>
      val arr = line.trim.split("\\s+").filter(_.nonEmpty)
      val rank = arr(0).toInt
      val stepSize = arr(1).toDouble
      val regs = arr(2).split(",").map(_.toDouble)
      val l2 = (regs(0), regs(1), regs(2))
      val numIterations = arr(3).toInt
      val isValidation = if (arr.length > 4) arr(4).toBoolean else true
      val lfm = new FMRegression(trainSet, stepSize, l2, rank, useAdaGrad, 1.0, storageLevel)
      var iter = 0
      var model: FMModel = null
      while (iter < numIterations) {
        val thisItr = math.min(50, numIterations - iter)
        iter += thisItr
        if (model != null) model.factors.unpersist(false)
        lfm.run(thisItr)
        model = lfm.saveModel()
        model.factors.persist(storageLevel)
        model.factors.count()
        val pout = s"rank=$rank stepSize=$stepSize l2=$l2"
        if (isValidation) {
          val rmse = model.loss(validationSet)
          logInfo(f"(Iteration $iter/$numIterations $pout) Validation RMSE:                     $rmse%1.4f")
          println(f"(Iteration $iter/$numIterations $pout) Validation RMSE:                     $rmse%1.4f")
        } else {
          val rmse = model.loss(testSet)
          logInfo(f"(Iteration $iter/$numIterations $pout) Test RMSE:                     $rmse%1.4f")
          println(f"(Iteration $iter/$numIterations $pout) Test RMSE:                     $rmse%1.4f")
        }
      }
      // model.save(sc, out)
    }
    sc.stop()
  }
}
