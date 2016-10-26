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

import java.util.{UUID, Random => JavaRandom}

import breeze.linalg.{DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV}
import org.apache.commons.math3.random.Well19937c
import org.apache.commons.math3.distribution.GammaDistribution

import org.apache.spark.Logging
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import org.parameterserver.client.{MatrixClient, PSClient}
import org.parameterserver.protocol.matrix.{Row, RowData}
import org.parameterserver.protocol.vector.{DenseVector => PDV, SparseVector => PSV, Vector => PV}
import org.parameterserver.protocol.{DataType, DoubleArray, Array => PSArray}
import org.parameterserver.{Configuration => PSConf}

import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.graphx.util.CompletionIterator
import com.github.cloudml.zen.ml.recommendation.MVMModel
import com.github.cloudml.zen.ml.util.{Utils, XORShiftRandom}

private[ml] abstract class MVM(
  @transient val data: RDD[LabeledPoint],
  val views: Array[Long],
  val rank: Int) extends Serializable with Logging {

  import MVM._

  def stepSize: Double

  def regParam: (Double, Double)

  def batchSize: Int

  def samplingFraction: Double = 1D

  private var innerEpoch: Int = 1
  var storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  val numSamples: Long = data.count()
  val numFeature = data.first().features.size

  protected val viewSize = views.length

  @transient val extDataSet: RDD[LabeledPoint] = data.map(x => {
    val start = views.last.toInt
    val sv = BSV.zeros[Double](views.length + numFeature)
    for (i <- views.indices) {
      sv(start + i) = 1.0
    }
    x.features.activeIterator.foreach(y => sv(y._1) = y._2)
    sv.compact()
    new LabeledPoint(x.label, sv)
  }).persist(storageLevel)
  extDataSet.count()

  @transient val featureIds: RDD[Int] = extDataSet.map(_.features.toSparse).flatMap(_.indices).
    distinct.persist(storageLevel)
  featureIds.count()

  protected val weightName = {
    val psClient = new PSClient(new PSConf(true))
    val maxId = featureIds.max()
    val wName = UUID.randomUUID().toString
    psClient.createMatrix(wName, false, true, maxId + 1, rank, DataType.Double)
    featureIds.mapPartitionsWithIndex { case (pid, iter) =>
      val rand = new Well19937c(pid + 1173)
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
    val gName = UUID.randomUUID().toString
    psClient.createMatrix(gName, weightName)
    psClient.close()
    gName
  }

  def eta: Double = 2D / numSamples

  def features: RDD[(Int, VD)] = {
    featureIds.mapPartitions { iter =>
      val psClient = new PSClient(new PSConf(true))
      val newIter = iter.grouped(batchSize).flatMap { seq =>
        val ids = seq.toArray
        val rows = psClient.getMatrix(weightName, ids.map(r => new Row(r)))
        ids.zip(rowData2Array(rows))
      }
      CompletionIterator[(Int, VD), Iterator[(Int, VD)]](newIter, psClient.close())
    }
  }

  protected def cleanGardSum(rho: Double): Unit = {
    val psClient = new PSClient(new PSConf(true))
    psClient.matrixAxpby(gardSumName, rho, weightName, 0D)
    psClient.close()
  }

  def run(iterations: Int): Unit = {
    for (epoch <- 1 to iterations) {
      // cleanGardSum(math.exp(-math.log(2D) / 40))
      logInfo(s"Start train (Iteration $epoch/$iterations)")
      val startedAt = System.nanoTime()
      val thisStepSize = stepSize
      val pSize = extDataSet.partitions.length
      val sampledData = if (samplingFraction == 1D) {
        extDataSet
      } else {
        extDataSet.sample(withReplacement = false, samplingFraction, innerEpoch + 17)
      }.mapPartitionsWithIndex { case (pid, iter) =>
        val rand = new XORShiftRandom(innerEpoch * pSize + pid + 119)
        iter.map(t => (rand.nextInt(), t))
      }.sortByKey().map(_._2)
      val costSum = sampledData.mapPartitionsWithIndex { case (pid, iter) =>
        val psClient = new PSClient(new PSConf(true))
        val reader = new MatrixClient(psClient, weightName)
        val rand = new XORShiftRandom(innerEpoch * pSize + pid)
        var innerIter = 0
        val newIter = iter.grouped(batchSize).map { samples =>
          var costSum = 0D
          val sampledSize = samples.length
          val featureIds = samples.flatMap(_.features.toSparse.indices).distinct.sorted.toArray
          val weights = rowData2Array(reader.read(featureIds))
          val f2i = featureIds.zipWithIndex.toMap
          val grad = new Array[VD](featureIds.length)
          samples.foreach { case LabeledPoint(label, features) =>
            val ssv = features.toSparse
            val indices = ssv.indices
            val values = ssv.values
            val arr = forwardSample(indices, values, f2i, weights)
            val (multi, loss) = multiplier(arr, label)
            costSum += loss
            backwardSample(indices, values, multi, f2i, weights, grad)
          }

          reader.clear()
          grad.foreach(g => g.indices.foreach(i => g(i) /= sampledSize))
          l2(featureIds, weights, grad)
          innerIter += 1
          updateWeight(grad, weights, featureIds, psClient, rand, thisStepSize, innerIter)
          Array(costSum, sampledSize.toDouble)
        }
        CompletionIterator[Array[Double], Iterator[Array[Double]]](newIter, psClient.close())
      }.reduce(reduceInterval)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      val loss = lossSum(costSum.head, costSum.last.toLong)
      println(s"(Iteration $epoch/$iterations) loss:                     $loss")
      logInfo(s"End  train (Iteration $epoch/$iterations) takes:         $elapsedSeconds")
      innerEpoch += 1
    }
  }

  private def backwardSample(
    indices: Array[Int],
    values: Array[Double],
    multi: Array[Double],
    fId2Offset: Map[Int, Int],
    features: Array[VD],
    grad: Array[VD]): Unit = {
    var i = 0
    while (i < indices.length) {
      val featureId = indices(i)
      val value = values(i)
      val fo = fId2Offset(featureId)
      val viewId = featureId2viewId(featureId, views)
      if (grad(fo) == null) grad(fo) = new Array[Double](rank + 1)
      backward(rank, viewId, value, multi, multi.last, grad(fo))
      i += 1
    }
  }

  private def l2(
    featureIds: Array[Int],
    features: Array[VD],
    grad: Array[VD]): Unit = {
    val rankIndices = 0 until rank
    featureIds.indices.foreach { i =>
      val fid = featureIds(i)
      val w = features(i)
      val g = grad(i)
      val deg = g.last
      rankIndices.foreach { rankId =>
        val reg = if (fid < numFeature) regParam._2 else regParam._1
        g(rankId) += deg * reg * w(rankId)
      }
    }
  }

  private def forwardSample(
    indices: Array[Int],
    values: Array[Double],
    fId2Offset: Map[Int, Int],
    features: Array[VD]): Array[Double] = {
    val arr = new Array[Double](rank * viewSize)
    var i = 0
    while (i < indices.length) {
      val featureId = indices(i)
      val value = values(i)
      val fo = fId2Offset(featureId)
      val w = features(fo)
      val viewId = featureId2viewId(featureId, views)
      forward(rank, viewId, arr, value, w)
      i += 1
    }
    arr
  }

  def updateWeight(
    grad: Array[VD],
    features: Array[VD],
    featuresIds: Array[Int],
    psClient: PSClient,
    rand: JavaRandom,
    stepSize: Double,
    iter: Int): Unit = {
    val rankIndices = 0 until rank
    val newGrad = Array.fill(featuresIds.length)(new VD(rank))
    val g2Sum = adaGrad(grad, featuresIds, psClient)
    val nuEpsilon = stepSize * eta
    featuresIds.indices.foreach { i =>
      val g2 = g2Sum(i)
      // val w = features(i)
      val g = grad(i)
      val ng = newGrad(i)
      val deg = g.last
      assert(deg <= 1D)
      rankIndices.foreach { rankId =>
        val nu = deg * rand.nextGaussian() * math.sqrt(nuEpsilon / g2(rankId))
        // ng(rankId) = -stepSize * (g(rankId) + regParam * w(rankId)) + nu
        ng(rankId) = -stepSize * g(rankId) + nu
      }
    }
    psClient.add2Matrix(weightName, array2RowData(newGrad, featuresIds))
  }

  def adaGrad(
    grad: Array[VD],
    featuresIds: Array[Int],
    psClient: PSClient): Array[VD] = {
    val rankIndices = 0 until rank
    val t2Sum = grad.map { g =>
      val t2 = new Array[Double](rank)
      rankIndices.foreach { i =>
        t2(i) = g(i) * g(i)
      }
      t2
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
        sum(offset) = 1E-6 + math.sqrt(sum(offset) + t2(offset))
        g(offset) /= sum(offset)
      }
    }
    psClient.add2Matrix(gardSumName, array2RowData(t2Sum, featuresIds))
    g2Sum
  }

  def saveModel(): MVMModel

  def predict(arr: Array[Double]): Double

  def forward(rank: Int, viewId: Int, arr: Array[Double], z: ED, w: VD): Array[Double] = {
    forwardInterval(rank, viewId, arr, z, w)
  }

  def multiplier(arr: Array[Double], label: Double): (Array[Double], Double)

  def lossSum(loss: Double, numSamples: Long): Double

  def backward(rank: Int, viewId: Int, x: ED, arr: Array[Double],
    multi: Double, grad: VD): Array[Double] = {
    backwardInterval(rank, viewId, x, arr, multi, grad)
  }
}

class MVMRegression(
  @transient override val data: RDD[LabeledPoint],
  override val views: Array[Long],
  override val rank: Int,
  override val stepSize: Double,
  override val regParam: (Double, Double),
  override val batchSize: Int,
  override val samplingFraction: Double = 1D,
  override val eta: Double = 1E-6) extends MVM(data, views, rank) {
  require(samplingFraction > 0 && samplingFraction <= 1,
    s"Sampling fraction ($samplingFraction) must be > 0 and <= 1")
  val max = data.map(_.label).max
  val min = data.map(_.label).min

  override def saveModel(): MVMModel = {
    new MVMModel(rank, views, false, features.map(t => (t._1.toLong, t._2)), max, min)
  }

  override def predict(arr: Array[Double]): Double = {
    var result = MVM.predictInterval(rank, arr)
    result = Math.max(result, min)
    result = Math.min(result, max)
    result
  }

  override def lossSum(loss: Double, numSamples: Long): Double = {
    math.sqrt(loss / numSamples)
  }

  override def multiplier(arr: Array[Double], label: Double): (Array[Double], Double) = {
    val multi = MVM.sumInterval(rank, arr)
    var sum = multi.last
    sum = Math.max(sum, min)
    sum = Math.min(sum, max)
    val diff = sum - label
    multi(multi.length - 1) = diff * 2.0
    (multi, diff * diff)
  }
}

class MVMClassification(
  @transient override val data: RDD[LabeledPoint],
  override val views: Array[Long],
  override val rank: Int,
  override val stepSize: Double,
  override val regParam: (Double, Double),
  override val batchSize: Int,
  override val samplingFraction: Double = 1D,
  override val eta: Double = 1E-6) extends MVM(data, views, rank) {
  require(samplingFraction > 0 && samplingFraction <= 1,
    s"Sampling fraction ($samplingFraction) must be > 0 and <= 1")

  override def saveModel(): MVMModel = {
    new MVMModel(rank, views, true, features.map(t => (t._1.toLong, t._2)), 1D, 0D)
  }

  override def predict(arr: Array[Double]): Double = {
    val result = MVM.predictInterval(rank, arr)
    sigmoid(result)
  }

  @inline private def sigmoid(x: Double): Double = {
    1D / (1D + math.exp(-x))
  }

  override def lossSum(loss: Double, numSamples: Long): Double = {
    loss / numSamples
  }

  override def multiplier(arr: Array[Double], label: Double): (Array[Double], Double) = {
    val multi = MVM.sumInterval(rank, arr)
    val z = multi.last
    val diff = sigmoid(z) - label
    multi(multi.length - 1) = diff
    (multi, Utils.log1pExp(if (label > 0D) -z else z))
  }
}

object MVM {
  private[ml] type ED = Double
  private[ml] type VD = Array[Double]

  def trainRegression(
    input: RDD[LabeledPoint],
    views: Array[Long],
    rank: Int,
    numIterations: Int,
    stepSize: Double,
    regParam: (Double, Double),
    miniBatch: Int = 100,
    samplingFraction: Double = 1D,
    eta: Double = 1E-6): MVMModel = {
    val mvm = new MVMRegression(input, views, rank, stepSize, regParam, miniBatch,
      samplingFraction, eta)
    mvm.run(numIterations)
    mvm.saveModel()
  }

  def trainClassification(
    input: RDD[LabeledPoint],
    views: Array[Long],
    rank: Int,
    numIterations: Int,
    stepSize: Double,
    regParam: (Double, Double),
    miniBatch: Int = 100,
    samplingFraction: Double = 1D,
    eta: Double = 1E-6): MVMModel = {
    val data = input.map { case LabeledPoint(label, features) =>
      LabeledPoint(if (label > 0D) 1D else 0D, features)
    }
    val mvm = new MVMClassification(data, views, rank, stepSize, regParam, miniBatch,
      samplingFraction, eta)
    mvm.run(numIterations)
    mvm.saveModel()
  }

  private[ml] def rowData2Array(rows: Array[RowData]): Array[VD] = {
    rows.map { row =>
      val pv = row.getData
      if (pv.getValues == null) {
        new VD(pv.getSize)
      } else {
        pv.getValues.asInstanceOf[DoubleArray].getValues
      }
    }
  }

  private[ml] def array2RowData(values: Array[VD], features: Array[Int]): Array[RowData] = {
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
  private[ml] def forwardInterval(rank: Int, viewId: Int, arr: Array[Double], z: ED, w: VD): Array[Double] = {
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
    multi: ED,
    m: Array[Double]): Array[Double] = {
    var i = 0
    while (i < rank) {
      m(i) += multi * x * arr(i + viewId * rank)
      i += 1
    }
    m(rank) += 1
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
