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
import com.github.cloudml.zen.ml.recommendation.MVMModel
import com.github.cloudml.zen.ml.util._
import com.google.common.io.Files
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.storage.StorageLevel
import org.scalatest.{FunSuite, Matchers}

class MVMSuite extends FunSuite with SharedSparkContext with Matchers {
  test("movieLens 100k (uid,mid) ") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/ml-100k/u.data"
    val checkpointDir = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpointDir)

    val movieLens = sc.textFile(dataSetFile, 2).mapPartitions { iter =>
      iter.filter(t => !t.startsWith("userId") && !t.isEmpty).map { line =>
        val Array(userId, movieId, rating, timestamp) = line.split("\t")
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

    val views = Array(maxUserId, numFeatures).map(_.toLong)
    val stepSize = 0.1
    val numIterations = 10000
    val regParam = 0.065
    val rank = 4
    val useAdaGrad = false
    val miniBatch = 100

    val lfm = new MVMRegression(trainSet, views, rank, stepSize, regParam, miniBatch, useAdaGrad)
    var iter = 0
    var model: MVMModel = null
    while (iter < numIterations) {
      val thisItr = if (iter < 50) {
        math.min(25, numIterations - iter)
      } else {
        math.min(10, numIterations - iter)
      }
      iter += thisItr
      lfm.run(thisItr)
      model = lfm.saveModel()
      model.factors.count()
      val rmse = model.loss(testSet)
      println(f"(Iteration $iter/$numIterations) Test RMSE:                     $rmse%1.6f")
    }

  }
}
