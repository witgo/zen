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

import com.github.cloudml.zen.ml.neuralNetwork.{MLP, DBN}
import com.github.cloudml.zen.ml.util._
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics}
import org.apache.spark.mllib.regression.LabeledPoint
import com.google.common.io.Files
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, sum => brzSum, Vector => BV}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, Vector => SV, SparseVector => SSV, Vectors}
import org.apache.spark.storage.StorageLevel

import org.scalatest.{Matchers, FunSuite}

class MVMSuite extends FunSuite with SharedSparkContext with Matchers {

  def getDataWithTime(sqlContext: SQLContext): RDD[(Int, LabeledPoint)] = {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val trainFile = s"$sparkHome/data/1017_140331_141101.tsv"
    import com.github.cloudml.zen.ml.util.SparkUtils
    val data = sc.textFile(trainFile).filter(_.nonEmpty).
      map(_.split("\t")).filter(_.length == 114).map { line =>
      val label = line.head.toDouble
      val time = line(1).substring(1, 7).toInt
      val features = BDV.apply(line.drop(2).map(_.toDouble))
      (time, LabeledPoint(label, SparkUtils.fromBreeze(features)))
    }.filter(_._2.label >= 0).filter(_._1 != 201410)
    val featureSummary = data.map(_._2.features).aggregate(new MultivariateOnlineSummarizer())(
      (summary, feat) => summary.add(feat),
      (sum1, sum2) => sum1.merge(sum2))

    val max = featureSummary.max
    val min = featureSummary.min

    data.map { case (time, LabeledPoint(label, features)) =>
      val v = SparkUtils.toBreeze(features.copy)
      for (i <- 0 until v.length) {
        v(i) = if (max(i) == min(i)) 0 else (v(i) - min(i)) / (max(i) - min(i))
      }
      (time, LabeledPoint(label, SparkUtils.fromBreeze(v)))
    }.persist(StorageLevel.MEMORY_ONLY_SER)
  }

  ignore("binary classification") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val dataSetFile = s"$sparkHome/data/binary_classification_data.txt"
    val checkpoint = s"$sparkHome/tmp"
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
    val dataSetFile = s"$sparkHome/data/binary_classification_data.txt"
    val checkpointDir = s"$sparkHome/tmp"
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

  ignore("小微贷款 风控 LR") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val checkpoint = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpoint)
    val data = getDataWithTime(new SQLContext(sc))
    assert(data.map(_._2.label).distinct().count() == 2)
    println(s"data size: ${data.count()}")

    val trainSet = data.filter(_._1 != 201410).map(_._2).persist(StorageLevel.MEMORY_AND_DISK)
    val testSet = data.filter(_._1 == 201410).map(_._2).persist(StorageLevel.MEMORY_AND_DISK)

    val pc = trainSet.filter(_.label == 0).count
    val fc = trainSet.filter(_.label == 1).count
    val ac = trainSet.count
    println(s"$pc + $fc => $ac")
    testSet.count()
    trainSet.count
    data.unpersist()

    val lr = new LogisticRegressionWithLBFGS()
    lr.setNumClasses(2).optimizer.setConvergenceTol(1e-5)
    // .optimizer.setRegParam(1e-3)
    val model = lr.run(trainSet).clearThreshold()
    val scoreAndLabels = testSet.map { s =>
      val label = s.label
      val score = model.predict(s.features)
      (score, label)
    }

    scoreAndLabels.persist(StorageLevel.MEMORY_AND_DISK)
    scoreAndLabels.repartition(1).map(
      t => s"${t._1}\t${t._2}").saveAsTextFile(s"$checkpoint/lr/${System.currentTimeMillis()}")
    val testAccuracy = new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
    println(f"Test AUC = $testAccuracy%1.6f")
  }

  ignore("小微贷款 风控 MIS") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val checkpoint = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpoint)
    val data = getDataWithTime(new SQLContext(sc))
    assert(data.map(_._2.label).distinct().count() == 2)
    println(s"data size: ${data.count()}")

    val trainSet = data.filter(_._1 != 201410).map(_._2).zipWithIndex().map(_.swap)
      .persist(StorageLevel.MEMORY_AND_DISK)
    val testSet = data.filter(_._1 == 201410).map(_._2).zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)

    val pc = trainSet.filter(_._2.label == 0).count
    val fc = trainSet.filter(_._2.label == 1).count
    val ac = trainSet.count
    println(s"$pc + $fc => $ac")
    testSet.count()
    trainSet.count
    data.unpersist()

    import com.github.cloudml.zen.ml.regression.LogisticRegression
    val model = LogisticRegression.trainMIS(trainSet, 200, 0.1, 0.0, 1e-6, true)
    val scoreAndLabels = testSet.join(testSet.map { t =>
      (t._1, model.predict(t._2.features))
    }).map { case (_, (lp, score)) =>
      (score, lp.label)
    }

    scoreAndLabels.persist(StorageLevel.MEMORY_AND_DISK)
    scoreAndLabels.repartition(1).map(
      t => s"${t._1}\t${t._2}").saveAsTextFile(s"$checkpoint/mis/${System.currentTimeMillis()}")
    val testAccuracy = new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
    println(f"Test AUC = $testAccuracy%1.6f")
  }

  ignore("小微贷款 风控 FM") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val checkpoint = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpoint)
    val data = getDataWithTime(new SQLContext(sc))
    assert(data.map(_._2.label).distinct().count() == 2)
    println(s"data size: ${data.count()}")

    val trainSet = data.filter(_._1 != 201410).map(_._2).zipWithIndex().map(_.swap)
      .persist(StorageLevel.MEMORY_AND_DISK)
    val testSet = data.filter(_._1 == 201410).map(_._2).zipWithIndex().map(_.swap).persist(StorageLevel.MEMORY_AND_DISK)

    val pc = trainSet.filter(_._2.label == 0).count
    val fc = trainSet.filter(_._2.label == 1).count
    val ac = trainSet.count
    println(s"$pc + $fc => $ac")
    testSet.count()
    trainSet.count
    data.unpersist()

    val model = FM.trainClassification(trainSet, 600, 0.005, (0.0, 0.0, 0.0), 20, true)
    val scoreAndLabels = testSet.join(model.predict(testSet.map(
      t => (t._1, t._2.features)))).map { case (_, (lp, score)) =>
      (score, lp.label)
    }

    scoreAndLabels.persist(StorageLevel.MEMORY_AND_DISK)
    scoreAndLabels.repartition(1).map(t => s"${t._1}\t${t._2}").
      saveAsTextFile(s"$checkpoint/fm/${System.currentTimeMillis()}")
    val testAccuracy = new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
    println(f"Test AUC = $testAccuracy%1.6f")

  }

  test("小微贷款 风控 DBN") {
    import com.github.cloudml.zen.ml.util.SparkUtils._

    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val checkpoint = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpoint)
    val data = getDataWithTime(new SQLContext(sc))
    assert(data.map(_._2.label).distinct().count() == 2)
    println(s"data size: ${data.count()}")

    def toV(rdd: RDD[LabeledPoint]): RDD[(SV, SV)] = {
      rdd.map { case LabeledPoint(label, features) =>
        val v = new Array[Double](2)
        if (label > 0) {
          v(0) = 0.98
          v(1) = 0.02
        } else {
          v(0) = 0.02
          v(1) = 0.98
        }
        (features, Vectors.dense(v))
      }.persist(StorageLevel.MEMORY_ONLY_SER)
    }
    val trainSet = toV(data.filter(_._1 != 201409).map(_._2))
    val testSet = toV(data.filter(_._1 == 201409).map(_._2))

    val layer1Size = testSet.first()._1.size
    testSet.count()
    trainSet.count()
    data.unpersist()

    val topology = Array(layer1Size, 800, 1000, 800, 2)
    val dropout = Array(0.0, 0.5, 0.5, 0.0)
    var mlp = new MLP(MLP.initLayers(topology), dropout)
    mlp = MLP.train(trainSet, 100, 4000, mlp, 0.02, 0.005, 1e-3)
    val scoreAndLabels = testSet.map { case (features, label) =>
      val f = toBreeze(features)
      val out = mlp.predict(f.toDenseVector.asDenseMatrix.t)
      (out(0, 0), if (label(0) > 0.5) 1.0 else 0.0)
    }
    scoreAndLabels.persist(StorageLevel.MEMORY_ONLY_SER)
    scoreAndLabels.repartition(1).map(t => s"${t._1}\t${t._2}").
      saveAsTextFile(s"$checkpoint/mlp/${System.currentTimeMillis()}")
    val testAccuracy = new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
    println(f"Test AUC = $testAccuracy%1.6f")
  }

  ignore("小微贷款 风控 GBDT") {
    val sparkHome = sys.props.getOrElse("spark.test.home", fail("spark.test.home is not set!"))
    val checkpoint = s"$sparkHome/target/tmp"
    sc.setCheckpointDir(checkpoint)
    val data = getDataWithTime(new SQLContext(sc))
    assert(data.map(_._2.label).distinct().count() == 2)
    println(s"data size: ${data.count()}")

    val trainSet = data.filter(_._1 != 201410).map(_._2).persist(StorageLevel.MEMORY_AND_DISK)
    val testSet = data.filter(_._1 == 201410).map(_._2).persist(StorageLevel.MEMORY_AND_DISK)

    val pc = trainSet.filter(_.label == 0).count()
    val fc = trainSet.filter(_.label == 1).count
    val ac = trainSet.count()
    println(s"$pc + $fc => $ac")
    data.unpersist()

    import org.apache.spark.mllib.tree._
    val boostingStrategy = configuration.BoostingStrategy.defaultParams("Classification")
    boostingStrategy.setNumIterations(30)
    // boostingStrategy.treeStrategy.setSubsamplingRate(0.3)
    boostingStrategy.treeStrategy.setNumClasses(2)
    boostingStrategy.treeStrategy.setMaxDepth(9)

    val model = GradientBoostedTrees.train(trainSet, boostingStrategy)
    val scoreAndLabels = testSet.map { s =>
      val label = s.label
      val score = model.predict(s.features)
      (score, label)
    }

    scoreAndLabels.persist(StorageLevel.MEMORY_AND_DISK)
    val testAccuracy = new MulticlassMetrics(scoreAndLabels).precision
    println(f"Test accuracy = $testAccuracy%1.6f")
    scoreAndLabels.repartition(1).map(
      t => s"${t._1}\t${t._2}").saveAsTextFile(s"$checkpoint/gbdt/${System.currentTimeMillis()}")

    //    val testAccuracy = new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
    //    println(f"Test AUC = $testAccuracy%1.6f")
  }
}
