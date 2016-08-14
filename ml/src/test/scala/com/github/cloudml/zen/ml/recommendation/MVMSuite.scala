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

package com.github.cloudml.zen.ml.recommendation

import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, sum => brzSum}
import com.github.cloudml.zen.ml.util._
import com.google.common.io.Files
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel
import org.scalatest.{FunSuite, Matchers}

class MVMSuite extends FunSuite with SharedSparkContext with Matchers {

  test("movieLens 1m (uid,mid) ") {
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
    }.zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)
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

    val views = Array(maxUserId, numFeatures).map(_.toLong)
    val stepSize = 0.05
    val numIterations = 1000
    val regParam = 0.12
    val rank = 32

    val lfm = new VecMVMRegression(rank, stepSize, regParam, views, trainSet, 0.01,
      numFeatures, 0, StorageLevel.MEMORY_AND_DISK)
    var iter = 0
    var model: MVMModel = null
    while (iter < numIterations) {
      val thisItr = if (iter < 100) {
        math.min(100, numIterations - iter)
      } else {
        math.min(100, numIterations - iter)
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
    val dataSetFile = classOf[MVMSuite].getClassLoader().getResource("binary_classification_data.txt").toString()
    val checkpoint = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpoint)
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).zipWithIndex().map {
      case (LabeledPoint(label, features), id) =>
        val newLabel = if (label > 0.0) 1.0 else 0.0
        (id, LabeledPoint(newLabel, features))
    }
    val stepSize = 0.1
    val regParam = 1e-2
    val l2 = (regParam, regParam, regParam)
    val rank = 20
    val useAdaGrad = true
    val trainSet = dataSet.cache()
    val fm = new FMClassification(trainSet, stepSize, l2, rank, useAdaGrad)

    val maxIter = 10
    val pps = new Array[Double](maxIter)
    var i = 0
    val startedAt = System.currentTimeMillis()
    while (i < maxIter) {
      fm.run(1)
      pps(i) = fm.saveModel().loss(trainSet)
      i += 1
    }
    println((System.currentTimeMillis() - startedAt) / 1e3)
    pps.foreach(println)

    val ppsDiff = pps.init.zip(pps.tail).map { case (lhs, rhs) => lhs - rhs }
    assert(ppsDiff.count(_ < 0).toDouble / ppsDiff.size > 0.05)

    val fmModel = fm.saveModel()
    val tempDir = Files.createTempDir()
    tempDir.deleteOnExit()
    val path = tempDir.toURI.toString
    fmModel.save(sc, path)
    val sameModel = FMModel.load(sc, path)
    assert(sameModel.k === fmModel.k)
    assert(sameModel.classification === fmModel.classification)
    assert(sameModel.factors.sortByKey().map(_._2).collect() ===
      fmModel.factors.sortByKey().map(_._2).collect())
  }

  ignore("url_combined classification") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = classOf[MVMSuite].getClassLoader().getResource("binary_classification_data.txt").toString()
    val checkpointDir = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpointDir)
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile).zipWithIndex().map {
      case (LabeledPoint(label, features), id) =>
        val newLabel = if (label > 0.0) 1.0 else 0.0
        (id, LabeledPoint(newLabel, features))
    }.cache()
    val numFeatures = dataSet.first()._2.features.size
    val stepSize = 0.1
    val numIterations = 500
    val regParam = 1e-3
    val rank = 20
    val views = Array(20, numFeatures / 2, numFeatures).map(_.toLong)
    val useAdaGrad = true
    val useWeightedLambda = true
    val miniBatchFraction = 1
    val Array(trainSet, testSet) = dataSet.randomSplit(Array(0.8, 0.2))
    trainSet.cache()
    testSet.cache()

    val fm = new MVMClassification(trainSet, stepSize, views, regParam, 0.0, rank,
      useAdaGrad, useWeightedLambda, miniBatchFraction)
    fm.run(numIterations)
    val model = fm.saveModel()
    println(f"Test loss: ${model.loss(testSet.cache())}%1.4f")

  }

}
