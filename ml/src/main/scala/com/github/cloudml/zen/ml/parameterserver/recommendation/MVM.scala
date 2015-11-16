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

import java.util.UUID

import breeze.linalg.{DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import com.github.cloudml.zen.graphx.util.CompletionIterator
import com.github.cloudml.zen.ml.recommendation.MVMModel
import com.github.cloudml.zen.ml.util.{SparkUtils, Utils}
import org.apache.commons.math3.distribution.GammaDistribution
import org.apache.commons.math3.random.{RandomGenerator, Well19937c}
import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.parameterserver.client.{MatrixReader, PSClient}
import org.parameterserver.protocol.matrix.{Row, RowData}
import org.parameterserver.protocol.vector.{DenseVector => PDV, SparseVector => PSV, Vector => PV}
import org.parameterserver.protocol.{Array => PSArray, DataType, DoubleArray}
import org.parameterserver.{Configuration => PSConf}

private[ml] abstract class MVM(
  val data: RDD[LabeledPoint],
  val views: Array[Long],
  val rank: Int) extends Serializable with Logging {

  import MVM._

  def useAdaGrad: Boolean

  def stepSize: Double

  def regParam: Double

  def batchSize: Int

  private var innerEpoch: Int = 1
  var storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  val numSamples: Long = data.count()
  val numFeature = data.first().features.size
  protected val viewFeaturesIds = views.indices.map(i => numFeature + i)
  protected val viewSize = views.length
  @transient lazy val featureIds: RDD[Int] = {
    (data.sparkContext.parallelize(views.indices.map(i => numFeature + i)) ++
      data.map(_.features.toSparse).flatMap(_.indices)).distinct.persist(storageLevel)
  }

  protected val weightName = {
    val psClient = new PSClient(new PSConf(true))
    val maxId = featureIds.max()
    val wName = UUID.randomUUID().toString
    psClient.createMatrix(wName, false, true, maxId + 1, rank, DataType.Double)
    featureIds.mapPartitionsWithIndex { case (pid, iter) =>
      val rand = new Well19937c(pid + 13)
      iter.map { featureId =>
        // parameter point
        val parms = Array.fill(rank) {
          rand.nextGaussian() * 1E-2
        }
        (featureId, parms)
      }
    }.foreachPartition { iter =>
      iter.grouped(batchSize).foreach { seq =>
        val (ids, values) = seq.unzip
        psClient.updateMatrix(wName, array2RowData(values.toArray, ids.toArray))
      }
      psClient.close()
    }

    psClient.close()
    wName
  }
  protected val gardSumName: String = {
    val psClient = new PSClient(new PSConf(true))
    val maxId = featureIds.max()
    val gName = UUID.randomUUID().toString
    psClient.createMatrix(gName, false, true, maxId + 1, rank, DataType.Double)

    featureIds.map { featureId =>
      (featureId, new Array[Double](rank))
    }.foreachPartition { iter =>
      iter.grouped(batchSize).foreach { seq =>
        val (ids, values) = seq.unzip
        psClient.updateMatrix(gName, array2RowData(values.toArray, ids.toArray))
      }
      psClient.close()
    }
    psClient.close()
    gName
  }

  def features: RDD[(Int, Array[Double])] = {
    featureIds.mapPartitions { iter =>
      val psClient = new PSClient(new PSConf(true))
      val newIter = iter.grouped(batchSize).flatMap { seq =>
        val ids = seq.toArray
        val rows = psClient.getMatrix(weightName, ids.map(r => new Row(r)))
        ids.zip(rowData2Array(rows))
      }
      CompletionIterator[(Int, Array[Double]), Iterator[(Int, Array[Double])]](newIter, psClient.close())
    }
  }

  def cleanGardSum(): Unit = {
    val featureIds = data.map(_.features.toSparse).flatMap(_.indices).distinct
    featureIds.map { featureId =>
      (featureId, new Array[Double](rank))
    }.foreachPartition { iter =>
      val psClient = new PSClient(new PSConf(true))
      iter.grouped(batchSize).foreach { seq =>
        val (ids, values) = seq.unzip
        psClient.updateMatrix(gardSumName, array2RowData(values.toArray, ids.toArray))
      }
      psClient.close()
    }
  }

  def run(iterations: Int): Unit = {
    for (epoch <- 1 to iterations) {
      // if (innerEpoch % 10 == 9) cleanGardSum()
      logInfo(s"Start train (Iteration $epoch/$iterations)")
      val pSize = data.partitions.length
      val startedAt = System.nanoTime()
      val gammaDist = samplingGammaDist()
      val thisStepSize = stepSize
      // val lossSum = data.sortBy(t => Utils.random.nextLong()).mapPartitionsWithIndex { case (pid, iter) =>
      val costSum = data.mapPartitionsWithIndex { case (pid, iter) =>
        val psClient = new PSClient(new PSConf(true))
        val reader = new MatrixReader(psClient, weightName)
        val rand = new Well19937c(innerEpoch * pSize + pid)
        val rankIndices = 0 until rank
        var innerIter = 0

        val newIter = iter.grouped(batchSize).map { samples =>
          var costSum = 0D
          val sampleSize = samples.length
          val featureIds = (samples.map(_.features.toSparse).flatMap(_.indices).distinct ++
            viewFeaturesIds).sortBy(t => t).toArray
          val features = rowData2Array(reader.read(featureIds))
          val f2i = featureIds.zipWithIndex.toMap
          val grad = new Array[Array[Double]](featureIds.length)
          samples.foreach { sample =>
            val arr = new Array[Double](rank * viewSize)
            val label = sample.label
            forwardSample(sample.features, f2i, features, arr)
            val multi = multiplier(arr, label)
            costSum += loss(multi, label)
            backwardSample(sample.features, multi, f2i, features, grad)
          }

          reader.clear()

          grad.foreach { g =>
            rankIndices.foreach { i =>
              g(i) = g(i) / sampleSize
            }
          }

          innerIter += 1
          updateWeight(grad, featureIds, gammaDist, psClient, rand, thisStepSize, innerIter)
          costSum
        }
        CompletionIterator[Double, Iterator[Double]](newIter, psClient.close())
      }.sum
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      println(s"(Iteration $epoch/$iterations) loss:                     ${lossSum(costSum, numSamples)}")
      logInfo(s"End  train (Iteration $epoch/$iterations) takes:         $elapsedSeconds")
      innerEpoch += 1
    }
  }

  private def backwardSample(
    sample: SV,
    multi: Array[Double],
    fId2Offset: Map[Int, Int],
    features: Array[Array[Double]],
    grad: Array[Array[Double]]): Unit = {
    val ssv = sample.toSparse
    var i = 0
    while (i < ssv.indices.length) {
      val featureId = ssv.indices(i)
      val value = ssv.values(i)
      val viewId = featureId2viewId(featureId, views)
      val b = backward(rank, viewId, value, multi, multi.last)
      for (rankId <- 0 until rank) {
        b(i) += regParam * features(fId2Offset(featureId))(rankId)
      }
      if (grad(fId2Offset(featureId)) == null) {
        grad(fId2Offset(featureId)) = b
      } else {
        reduceInterval(grad(fId2Offset(featureId)), b)
      }
      i += 1
    }

    viewFeaturesIds.foreach { featureId =>
      val viewId = featureId2viewId(featureId, views)
      val value = 1D
      val b = backward(rank, viewId, value, multi, multi.last)
      for (rankId <- 0 until rank) {
        b(i) += regParam * features(fId2Offset(featureId))(rankId)
      }
      if (grad(fId2Offset(featureId)) == null) {
        grad(fId2Offset(featureId)) = b
      } else {
        reduceInterval(grad(fId2Offset(featureId)), b)
      }
    }
  }


  private def forwardSample(
    sample: SV, fId2Offset: Map[Int, Int],
    features: Array[Array[Double]], arr: Array[Double]): Unit = {
    val ssv = sample.toSparse
    var i = 0
    while (i < ssv.indices.length) {
      val featureId = ssv.indices(i)
      val value = ssv.values(i)
      val viewId = featureId2viewId(featureId, views)
      forward(rank, viewId, arr, value, features(fId2Offset(featureId)))
      i += 1
    }
    viewFeaturesIds.foreach { featureId =>
      val viewId = featureId2viewId(featureId, views)
      val value = 1D
      forward(rank, viewId, arr, value, features(fId2Offset(featureId)))
    }
  }

  def updateWeight(
    grad: Array[Array[Double]],
    featuresIds: Array[Int],
    gammaDist: Array[Double],
    psClient: PSClient,
    rand: RandomGenerator,
    stepSize: Double,
    iter: Int): Unit = {
    if (useAdaGrad) adaGrad(grad, featuresIds, psClient)
    val rankIndices = 0 until rank
    val gamma = 0.501
    val tss = if (useAdaGrad) stepSize else stepSize / math.pow(iter + 17D, gamma)
    val epsilon = 1.0 / math.pow(iter + 17D, gamma)
    grad.zip(featuresIds).foreach { case (g, vid) =>
      val viewId = featureId2viewId(vid, views)
      rankIndices.foreach { i =>
        g(i) = -(tss * g(i) + rand.nextGaussian() * math.sqrt(gammaDist(i + viewId * rank)) * epsilon)
      }
    }
    psClient.add2Matrix(weightName, array2RowData(grad, featuresIds))
  }

  def updateWeight(
    grad: Array[Array[Double]],
    featuresIds: Array[Int],
    psClient: PSClient,
    rand: RandomGenerator,
    stepSize: Double,
    iter: Int): Unit = {
    if (useAdaGrad) adaGrad(grad, featuresIds, psClient)
    val rankIndices = 0 until rank
    val gamma = 0.501
    val tss = if (useAdaGrad) stepSize else stepSize / math.pow(iter + 17D, gamma) / 2D
    val epsilon = stepSize / math.pow(iter + 17D, gamma) / numSamples
    grad.foreach { g =>
      rankIndices.foreach { i =>
        g(i) = -(tss * g(i) + epsilon * rand.nextGaussian() * math.sqrt(epsilon))
      }
    }
    psClient.add2Matrix(weightName, array2RowData(grad, featuresIds))
  }


  def adaGrad(
    grad: Array[Array[Double]],
    featuresIds: Array[Int],
    psClient: PSClient): Unit = {
    val rankIndices = 0 until rank
    val t2Sum = grad.map { g =>
      val p = new Array[Double](rank)
      rankIndices.foreach { i =>
        p(i) = g(i) * g(i)
      }
      p
    }

    val rowData = psClient.getMatrix(gardSumName, featuresIds.map(f => new Row(f)))
    featuresIds.indices.foreach { i =>
      assert(featuresIds(i) == rowData(i).getRow)
    }
    val g2Sum = rowData2Array(rowData)
    featuresIds.indices.foreach { i =>
      val g = grad(i)
      val sum = g2Sum(i)
      val t2 = t2Sum(i)
      rankIndices.foreach { offset =>
        g(offset) /= 1e-6 + math.sqrt(sum(offset) + t2(offset))
      }
    }
    psClient.add2Matrix(gardSumName, array2RowData(t2Sum, featuresIds))
  }

  private def samplingGammaDist(): Array[Double] = {
    val rankIndices = 0 until rank
    val viewSize = views.length
    val features = this.features.persist(storageLevel)
    val mean = features.aggregate(new Array[Double](rank * viewSize))((arr, a) => {
      val (vid, weight) = a
      val viewId = featureId2viewId(vid, views)
      for (i <- rankIndices) {
        arr(i + viewId * rank) += weight(i)
      }
      arr
    }, reduceInterval)
    val dist = features.aggregate(new Array[Double](2 * rank * viewSize))((arr, a) => {
      val (vid, weight) = a
      val viewId = featureId2viewId(vid, views)
      for (i <- rankIndices) {
        val offset = i + viewId * rank
        arr(offset) += 1
        arr(offset + viewSize * rank) += math.pow(weight(i) - mean(offset), 2)
      }
      arr
    }, reduceInterval)
    val gamma = new Array[Double](rank * viewSize)
    val alpha = 1.0
    val beta = 1.0
    val rand = new Well19937c(Utils.random.nextLong())
    for (viewId <- 0 until viewSize) {
      for (rankId <- rankIndices) {
        val offset = rankId + viewId * rank
        val shape = (alpha + dist(offset)) / 2.0
        val scale = (beta + dist(offset + viewSize * rank)) / 2.0
        val rng = new GammaDistribution(rand, shape, scale)
        gamma(offset) = 1.0 / rng.sample()
      }
    }
    features.unpersist(blocking = false)
    gamma
  }

  def saveModel(): MVMModel

  def predict(arr: Array[Double]): Double

  def forward(rank: Int, viewId: Int, arr: Array[Double], z: ED, w: Array[Double]): Array[Double] = {
    forwardInterval(rank, viewId, arr, z, w)
  }

  def multiplier(arr: Array[Double], label: Double): Array[Double]

  def loss(arr: Array[Double], label: Double): Double

  def lossSum(loss: Double, numSamples: Long): Double

  def backward(rank: Int, viewId: Int, x: ED, arr: Array[Double], multi: Double): Array[Double] = {
    backwardInterval(rank, viewId, x, arr, multi)
  }
}

class MVMRegression(
  @transient override val data: RDD[LabeledPoint],
  override val views: Array[Long],
  override val rank: Int,
  val stepSize: Double,
  val regParam: Double,
  val batchSize: Int,
  val useAdaGrad: Boolean) extends MVM(data, views, rank) {

  override def saveModel(): MVMModel = {
    new MVMModel(rank, views, false, features.map(t => (t._1.toLong, t._2)))
  }

  override def predict(arr: Array[Double]): Double = {
    val result = MVM.predictInterval(rank, arr)
    result
  }

  override def lossSum(loss: Double, numSamples: Long): Double = {
    math.sqrt(loss / numSamples)
  }

  override def loss(arr: Array[Double], label: Double): Double = {
    val diff = arr.last / 2
    diff * diff
  }

  override def multiplier(arr: Array[Double], label: Double): Array[Double] = {
    val ret = MVM.sumInterval(rank, arr)
    if (com.github.cloudml.zen.ml.util.Utils.random.nextDouble() < 1e-9) println(s"${label} - ${ret.last}")
    val diff = ret.last - label
    ret(ret.length - 1) = diff * 2.0
    ret
  }

}

object MVM {
  private[ml] type ED = Double
  private[ml] type VD = BDV[Double]

  def trainRegression(
    input: RDD[LabeledPoint],
    views: Array[Long],
    rank: Int,
    numIterations: Int,
    stepSize: Double,
    regParam: Double,
    miniBatch: Int = 100,
    useAdaGrad: Boolean = false): Unit = {
    val mvm = new MVMRegression(input, views, rank, stepSize, regParam, miniBatch, useAdaGrad)
    mvm.run(numIterations)
  }

  private[ml] def rowData2Array(rows: Array[RowData]): Array[Array[Double]] = {
    rows.map(_.getData.getValues.asInstanceOf[DoubleArray].getValues)
  }

  private[ml] def array2RowData(values: Array[Array[Double]], features: Array[Int]): Array[RowData] = {
    val pdv = values.map(v => new PDV(new DoubleArray(v)))
    features.indices.map { i =>
      val rd = new RowData(features(i))
      rd.setData(pdv(i))
      rd
    }.toArray
  }

  private[ml] def reduceInterval(a: Array[Double], b: Array[Double]): Array[Double] = {
    var i = 0
    while (i < a.length) {
      a(i) += b(i)
      i += 1
    }
    a
  }

  private[ml] def featureId2viewId(featureId: Long, views: Array[Long]): Int = {
    val numFeatures = views.last
    val viewId = if (featureId >= numFeatures) {
      featureId - numFeatures
    } else {
      val viewSize = views.length
      var adj = 0
      var found = false
      while (adj < viewSize - 1 && !found) {
        if (featureId < views(adj)) {
          found = true
        } else {
          adj += 1
        }
      }
      adj
    }
    viewId.toInt
  }

  /**
    * arr(k + v * rank)= (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)})
    * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
    */
  private[ml] def predictInterval(rank: Int, arr: Array[Double]): ED = {
    val viewSize = arr.length / rank
    var sum = 0.0
    var i = 0
    while (i < rank) {
      var multi = 1.0
      var viewId = 0
      while (viewId < viewSize) {
        multi *= arr(i + viewId * rank)
        viewId += 1
      }
      sum += multi
      i += 1
    }
    sum
  }

  /**
    * arr = rank * viewSize
    * when f belongs to [viewId * rank ,(viewId +1) * rank)
    * arr[f] = z_{i_v}^{(1)}a_{i_{i},j}^{(1)}
    */
  private[ml] def forwardInterval(rank: Int, viewId: Int, arr: Array[Double],
    z: ED, w: Array[Double]): Array[Double] = {
    var i = 0
    while (i < rank) {
      arr(i + viewId * rank) += z * w(i)
      i += 1
    }
    arr
  }

  /**
    * arr = rank * viewSize
    * when v=viewId , k belongs to [0,rank]
    * arr(k + v * rank) = \frac{\partial \hat{y}(x|\Theta )}{\partial\theta }
    * return multi * \frac{\partial \hat{y}(x|\Theta )}{\partial\theta }
    * clustering: multi = 1/(1+ \exp(-\hat{y}(x|\Theta)) ) - y
    * regression: multi = 2(\hat{y}(x|\Theta) -y)
    */
  private[ml] def backwardInterval(
    rank: Int,
    viewId: Int,
    x: ED,
    arr: Array[Double],
    multi: ED): Array[Double] = {
    val m = new Array[Double](rank)
    var i = 0
    while (i < rank) {
      m(i) = multi * x * arr(i + viewId * rank)
      i += 1
    }
    m
  }

  /**
    * arr(k + v * rank)= (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
    * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
    */
  private[ml] def sumInterval(rank: Int, arr: Array[Double]): Array[Double] = {
    val viewSize = arr.length / rank
    val m = new Array[Double](rank)
    var sum = 0.0
    var i = 0
    while (i < rank) {
      var multi = 1.0
      var viewId = 0
      while (viewId < viewSize) {
        multi *= arr(i + viewId * rank)
        viewId += 1
      }
      m(i) += multi
      sum += multi
      i += 1
    }

    val ret = new Array[Double](rank * viewSize + 1)
    i = 0
    while (i < rank) {
      var viewId = 0
      while (viewId < viewSize) {
        val vm = arr(i + viewId * rank)
        ret(i + viewId * rank) = if (vm == 0.0) {
          0.0
        } else {
          m(i) / vm
        }
        viewId += 1
      }
      i += 1
    }
    ret(rank * viewSize) = sum
    ret
  }
}
