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
import com.github.cloudml.zen.ml.recommendation.{ThreeWayFMModel, ThreeWayFMRegression, ThreeWayFM}
import com.github.cloudml.zen.ml.util.SparkHacker
import com.google.common.io.Files
import org.apache.spark.graphx.GraphXUtils
import org.apache.spark.mllib.linalg.{SparseVector => SSV}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{Logging, SparkConf, SparkContext}
import scopt.OptionParser

object MovieLensThreeWayFM extends Logging {

  case class Params(
    input: String = null,
    out: String = null,
    confPath: String = null,
    numPartitions: Int = -1,
    useAdaGrad: Boolean = false,
    useWeightedLambda: Boolean = false,
    kryo: Boolean = true) extends AbstractParams[Params]

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("MovieLensThreeWayFM") {
      head("MovieLensThreeWayFM: an example app for ThreeWayFM.")
      opt[Int]("numPartitions")
        .text(s"number of partitions, default: ${defaultParams.numPartitions}")
        .action((x, c) => c.copy(numPartitions = x))
      opt[String]("confPath")
        .text(s"conf file Path, default: ${defaultParams.confPath}")
        .action((x, c) => c.copy(confPath = x))
      opt[Unit]("kryo")
        .text("use Kryo serialization")
        .action((_, c) => c.copy(kryo = true))
      opt[Unit]("adagrad")
        .text("use AdaGrad")
        .action((_, c) => c.copy(useAdaGrad = true))
      opt[Unit]("weightedLambda")
        .text("use weighted lambda regularization")
        .action((_, c) => c.copy(useWeightedLambda = true))
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
          | bin/spark-submit --class com.github.cloudml.zen.examples.ml.MovieLensThreeWayFM \
          | examples/target/scala-*/zen-examples-*.jar \
          | --rank2 10 --rank3 10  --numIterations 50 --regular 0.01,0.01,0.01,0.01 --kryo \
          | data/mllib/sample_movielens_data.txt
          | data/mllib/ThreeWayFM_model
        """.stripMargin)
    }

    parser.parse(args, defaultParams).map { params =>
      run(params)
    }.getOrElse {
      System.exit(1)
    }
  }

  def run(params: Params): Unit = {
    val Params(input, out, confPath, numPartitions, useAdaGrad, useWeightedLambda, kryo) = params
    val storageLevel = StorageLevel.MEMORY_AND_DISK

    val conf = new SparkConf().setAppName(s"ThreeWayFM with $params")
    if (kryo) {
      GraphXUtils.registerKryoClasses(conf)
      // conf.set("spark.kryoserializer.buffer.mb", "8")
    }
    val sc = new SparkContext(conf)
    val checkpointDir = s"$out/checkpoint"
    sc.setCheckpointDir(checkpointDir)
    SparkHacker.gcCleaner(60 * 10, 60 * 10, "MovieLensThreeWayFM")
    val (trainSet, testSet, validationSet, views) = MovieLensUtils.crossValidation(sc,
      input, numPartitions, storageLevel)
    import scala.collection.JavaConversions._
    val lines = Files.readLines(new java.io.File(confPath), Charset.defaultCharset())
    lines.filter(l => !l.startsWith("#")).foreach { line =>
      val arr = line.trim.split("\\s+").filter(_.nonEmpty)
      val rank2 = arr(0).toInt
      val rank3 = arr(1).toInt
      val stepSize = arr(2).toDouble
      val regs = arr(3).split(",").map(_.toDouble)
      val l2 = (regs(0), regs(1), regs(2), regs(3))
      val numIterations = arr(4).toInt
      val isValidation = if (arr.length > 5) arr(5).toBoolean else true
      val lfm = new ThreeWayFMRegression(trainSet, stepSize, views, l2, rank2, rank3,
        useAdaGrad, useWeightedLambda, 1.0, storageLevel)
      var iter = 0
      var model: ThreeWayFMModel = null
      while (iter < numIterations) {
        val thisItr = math.min(50, numIterations - iter)
        iter += thisItr
        if (model != null) model.factors.unpersist(false)
        lfm.run(thisItr)
        model = lfm.saveModel()
        model.factors.persist(storageLevel)
        model.factors.count()
        val pout = s"rank2=$rank2 rank3=$rank3 stepSize=$stepSize l2=$l2"
        if (isValidation) {
          val rmse = model.loss(validationSet)
          logInfo(f"(Iteration $iter/$numIterations $pout) validation RMSE:                     $rmse%1.4f")
          println(f"(Iteration $iter/$numIterations $pout) validation RMSE:                     $rmse%1.4f")
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
