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

package com.github.cloudml.zen.ml.parameterserver.recommendation

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, sum => brzSum}
import com.github.cloudml.zen.ml.recommendation.{FMModel, MVMModel}
import com.github.cloudml.zen.ml.util._
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel
import org.scalatest.{FunSuite, Matchers}

class FMSuite extends FunSuite with SharedSparkContext with Matchers {
  test("movieLens 100k (uid,mid) ") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/ml-1m/ratings.dat"
    val checkpointDir = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpointDir)

    val movieLens = sc.textFile(dataSetFile, 2).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split("::")
        val gen = (1125899906842597L * timestamp.toLong).abs
        (userId.toInt, movieId.toInt, rating.toDouble, timestamp.toInt / (60 * 60 * 24), gen)
      }
    }.persist(StorageLevel.MEMORY_AND_DISK)
    val maxUserId = movieLens.map(_._1).max + 1
    val maxMovieId = movieLens.map(_._2).max + 1
    val numFeatures = maxUserId + maxMovieId
    val trainSet = movieLens.filter(t => t._5 % 5 != 3).map { case (userId, movieId, rating, _, _) =>
      val sv = BSV.zeros[Double](numFeatures)
      sv(userId) = 1.0
      sv(movieId + maxUserId) = 1.0
      sv.compact()
      new LabeledPoint(rating, new SSV(sv.length, sv.index, sv.data))
    }.persist(StorageLevel.MEMORY_AND_DISK)
    val testSet = movieLens.filter(t => t._5 % 5 == 3).map { case (userId, movieId, rating, _, _) =>
      val sv = BSV.zeros[Double](numFeatures)
      sv(userId) = 1.0
      sv(movieId + maxUserId) = 1.0
      sv.compact()
      new LabeledPoint(rating, new SSV(sv.length, sv.index, sv.data))
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
    trainSet.count()
    testSet.count()
    movieLens.unpersist()

    // MLUtils.saveAsLibSVMFile(trainSet.repartition(1), s"$sparkHome/data/ml-1m/trainSet/")
    // MLUtils.saveAsLibSVMFile(testSet.map(_._2).repartition(1), s"$sparkHome/data/ml-1m/testSet/")
    // sys.exit(-1)


    val stepSize = 0.05
    val numIterations = 10000
    val regParam = (0.1, 0.12, 0.12)
    val eta = 1E-6
    val samplingFraction = 1D
    val rank = 32
    val useAdaGrad = true
    val miniBatch = 100

    val lfm = new FMRegression(trainSet, rank, stepSize, regParam, miniBatch,
      useAdaGrad, samplingFraction, eta)
    var iter = 0
    var model: FMModel = null
    while (iter < numIterations) {
      val thisItr = if (iter < 10) {
        math.min(5, numIterations - iter)
      } else {
        math.min(5, numIterations - iter)
      }
      iter += thisItr
      lfm.run(thisItr)
      model = lfm.saveModel()
      model.factors.count()
      val rmse = model.loss(testSet)
      println(f"(Iteration $iter/$numIterations) Test RMSE:                     $rmse%1.6f")
    }
  }

  ignore("binary classification") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/binary_classification_data.txt"
    val checkpoint = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpoint)
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).map {
      case LabeledPoint(label, features) =>
        val newLabel = if (label > 0.0) 1.0 else 0.0
        val bsv = SparkUtils.toBreeze(features).asInstanceOf[BSV[Double]]
        bsv(40) = 0
        bsv.compact()
        LabeledPoint(newLabel, SparkUtils.fromBreeze(bsv))
    }.persist()
    val numFeatures = dataSet.first().features.size
    val views = Array(40, numFeatures).map(_.toLong)
    val stepSize = 0.1
    val numIterations = 1000
    val regParam = 0
    val rank = 4
    val useAdaGrad = true
    val miniBatch = 100
    val mvm = new MVMClassification(dataSet, views, rank, stepSize, regParam, miniBatch, useAdaGrad)

    mvm.run(numIterations)
  }
}
