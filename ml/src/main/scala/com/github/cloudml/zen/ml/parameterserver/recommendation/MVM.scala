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

import java.util.{Random => JavaRandom, UUID}

import breeze.linalg.{DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import com.github.cloudml.zen.graphx.util.CompletionIterator
import com.github.cloudml.zen.ml.recommendation.MVMModel
import com.github.cloudml.zen.ml.util.{XORShiftRandom, SparkUtils, Utils}
import org.apache.commons.math3.distribution.GammaDistribution
import org.apache.commons.math3.random.{RandomGenerator, Well19937c}
import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.parameterserver.client.{MatrixClient, PSClient}
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

  def shrinkageVal: Double = 1e-9

  def batchSize: Int

  def samplingFraction: Double = 1D

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
    if (useAdaGrad) {
      val psClient = new PSClient(new PSConf(true))
      val gName = UUID.randomUUID().toString
      psClient.createMatrix(gName, weightName)
      // psClient.matrixAxpby(gName, 1D, weightName, 0D)
      psClient.close()
      gName
    } else {
      null
    }
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
    if (useAdaGrad) {
      val psClient = new PSClient(new PSConf(true))
      psClient.matrixAxpby(gardSumName, rho, weightName, 0D)
      psClient.close()
    }
  }

  def run(iterations: Int): Unit = {
    for (epoch <- 1 to iterations) {
      // cleanGardSum(math.exp(-math.log(2D) / 40))
      logInfo(s"Start train (Iteration $epoch/$iterations)")
      val startedAt = System.nanoTime()
      val beta = 1D / 300D
      val alpha = regParam * 300D
      val regRand = new GammaDistribution(new Well19937c(Utils.random.nextLong()), alpha, beta)
      val regDist = regRand.sample(viewSize * rank)
      val gammaDist: Array[Double] = null
      //  samplingGammaDist(innerEpoch)
      val thisStepSize = stepSize
      val pSize = data.partitions.length
      val sampledData = if (samplingFraction == 1D) {
        data
      } else {
        data.sample(withReplacement = false, samplingFraction, innerEpoch + 17)
      }.mapPartitionsWithIndex { case (pid, iter) =>
        val rand = new XORShiftRandom(innerEpoch * pSize + pid + 119)
        iter.map(t => (rand.nextInt(), t))
      }.sortByKey().map(_._2)
      val costSum = sampledData.mapPartitionsWithIndex { case (pid, iter) =>
        val psClient = new PSClient(new PSConf(true))
        val reader = new MatrixClient(psClient, weightName)
        val rand = new XORShiftRandom(innerEpoch * pSize + pid)
        // val regRand = new GammaDistribution(new Well19937c(rand.nextLong()), alpha, beta)
        // val regDist = regRand.sample(viewSize * rank)
        var innerIter = 0
        val newIter = iter.grouped(batchSize).map { samples =>
          var costSum = 0D
          val sampledSize = samples.length
          val featureIds = (samples.map(_.features.toSparse).flatMap(_.indices).distinct ++
            viewFeaturesIds).sorted.toArray
          val features = rowData2Array(reader.read(featureIds))
          val f2i = featureIds.zipWithIndex.toMap
          val grad = new Array[VD](featureIds.length)
          samples.foreach { sample =>
            val ssv = sample.features.toSparse
            val indices = ssv.indices
            val values = ssv.values
            // val values = ssv.values.map(v => v + rand.nextGaussian() * 1E-1)
            val arr = forwardSample(indices, values, f2i, features)
            val (multi, loss) = multiplier(arr, sample.label)
            costSum += loss
            backwardSample(indices, values, multi, f2i, features, grad)
          }

          reader.clear()
          grad.foreach(g => g.indices.foreach(i => g(i) /= sampledSize))
          l2(gammaDist, regDist, rand, featureIds, features, grad)

          innerIter += 1
          updateWeight(grad, features, featureIds, psClient, gammaDist, rand, thisStepSize, innerIter)
          Array(costSum, sampledSize.toDouble)
        }
        CompletionIterator[Array[Double], Iterator[Array[Double]]](newIter, psClient.close())
      }.reduce(reduceInterval)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      val loss = lossSum(costSum.head, costSum.last.toLong)
      logInfo(s"(Iteration $epoch/$iterations) loss:                     $loss")
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

    viewFeaturesIds.foreach { featureId =>
      val viewId = featureId2viewId(featureId, views)
      val fo = fId2Offset(featureId)
      val value = 1D
      if (grad(fo) == null) grad(fo) = new Array[Double](rank + 1)
      backward(rank, viewId, value, multi, multi.last, grad(fo))
    }
  }

  private def l2(
    gammaDist: Array[Double],
    regDist: Array[Double],
    rand: JavaRandom,
    featureIds: Array[Int],
    features: Array[VD],
    grad: Array[VD]): Unit = {
    val rankIndices = 0 until rank
    features.indices.foreach { i =>
      val featureId = featureIds(i)
      val viewId = featureId2viewId(featureId, views)
      val w = features(i)
      val g = grad(i)
      val deg = g.last
      rankIndices.foreach { rankId =>
        assert(!(g(rankId).isNaN || g(rankId).isInfinity))
        val reg = regDist(rankId + viewId * rank)
        // val gamma = gammaDist(rankId + viewId * rank)
        // g(rankId) += deg * (reg + rand.nextGaussian() * gamma) * w(rankId)
        // g(rankId) += deg * (reg * w(rankId) + rand.nextGaussian() * gamma)
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
    viewFeaturesIds.foreach { featureId =>
      val viewId = featureId2viewId(featureId, views)
      val fo = fId2Offset(featureId)
      val w = features(fo)
      val value = 1D
      forward(rank, viewId, arr, value, w)
    }
    arr
  }

  def updateWeight(
    grad: Array[VD],
    features: Array[VD],
    featuresIds: Array[Int],
    psClient: PSClient,
    gammaDist: Array[Double],
    rand: JavaRandom,
    stepSize: Double,
    iter: Int): Unit = {
    val rankIndices = 0 until rank
    val newGrad = Array.fill(featuresIds.length)(new VD(rank))
    if (useAdaGrad) {
      val g2Sum = adaGrad(grad, featuresIds, psClient)
      // val nuEpsilon = stepSize * math.pow(iter + 48D, -0.51)
      // val nuEpsilon = stepSize / (math.sqrt(iter + 15D) * math.log(iter + 14D))
      // val nuEpsilon = 2D * stepSize / numSamples
      val nuEpsilon = stepSize * eta
      featuresIds.indices.foreach { i =>
        val g2 = g2Sum(i)
        val g = grad(i)
        val ng = newGrad(i)
        val deg = g.last
        assert(deg <= 1D)
        rankIndices.foreach { rankId =>
          val nu = deg * rand.nextGaussian() * math.sqrt(nuEpsilon / g2(rankId))
          // if (Utils.random.nextDouble() < 1E-7) println(g2(rankId))
          ng(rankId) = -stepSize * g(rankId) + nu
        }
      }
      l1(grad, g2Sum, features, featuresIds, newGrad, rand, stepSize, iter)
    } else {
      val epsilon = stepSize * math.pow(iter + 48D, -0.51)
      // val epsilon = stepSize / (math.sqrt(iter + 15D) * math.log(iter + 14D))
      val nuEpsilon = epsilon * eta
      featuresIds.indices.foreach { i =>
        val g = grad(i)
        // val w = features(i)
        // val vid = featuresIds(i)
        val ng = newGrad(i)
        val deg = g.last
        assert(deg <= 1D)
        // val viewId = featureId2viewId(vid, views)
        rankIndices.foreach { rankId =>
          // val gamma = deg * rand.nextGaussian() * gammaDist(rankId + viewId * rank) / g2(rankId)
          val nu = deg * rand.nextGaussian() * math.sqrt(nuEpsilon)
          ng(rankId) = -epsilon * g(rankId) + nu
        }
      }
    }
    psClient.add2Matrix(weightName, array2RowData(newGrad, featuresIds))
  }

  def l1(
    grad: Array[VD],
    g2Sum: Array[VD],
    features: Array[VD],
    featuresIds: Array[Int],
    newGrad: Array[VD],
    rand: JavaRandom,
    stepSize: Double,
    iter: Int): Unit = {
    val rankIndices = 0 until rank
    featuresIds.indices.filter(i => featuresIds(i) < numFeature).foreach { i =>
      val w = features(i)
      val g = grad(i)
      val g2 = g2Sum(i)
      val ng = newGrad(i)
      val deg = g.last
      assert(deg <= 1D)
      rankIndices.foreach { rankId =>
        if (shrinkageVal > 0) {
          val si = stepSize * deg * shrinkageVal / (1E-6 + math.sqrt(g2(rankId)))
          if (w(rankId).abs < si) {
            ng(rankId) = -1 * w(rankId)
          } else {
            ng(rankId) += -1 * w(rankId).signum * si
          }
        }
      }
    }
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

  private def samplingGammaDist(innerEpoch: Int): Array[Double] = {
    val rankIndices = 0 until rank
    val viewSize = views.length
    val dist = features.aggregate(new Array[Double]((rank + 1) * viewSize))((arr, a) => {
      val (vid, weight) = a
      val viewId = featureId2viewId(vid, views)
      arr(rank * viewSize + viewId) += 1
      for (rankId <- rankIndices) {
        val offset = rankId + viewId * rank
        arr(offset) += math.pow(weight(rankId), 2)
      }
      arr
    }, reduceInterval)
    val gamma = new Array[Double](rank * viewSize)
    val alpha = 1D
    val beta = 1D
    val rand = new Well19937c(innerEpoch)
    for (viewId <- 0 until viewSize) {
      val shape = alpha + dist(rank * viewSize + viewId) / 2D
      for (rankId <- rankIndices) {
        val offset = rankId + viewId * rank
        val scale = beta + dist(offset) / 2D
        val rng = new GammaDistribution(rand, shape, scale)
        gamma(offset) = math.sqrt(1D / rng.sample())
      }
    }
    gamma
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
  override val regParam: Double,
  override val batchSize: Int,
  override val useAdaGrad: Boolean,
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
  override val regParam: Double,
  override val batchSize: Int,
  override val useAdaGrad: Boolean,
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
    regParam: Double,
    miniBatch: Int = 100,
    useAdaGrad: Boolean = false,
    samplingFraction: Double = 1D,
    eta: Double = 1E-4): MVMModel = {
    val mvm = new MVMRegression(input, views, rank, stepSize, regParam, miniBatch,
      useAdaGrad, samplingFraction, eta)
    mvm.run(numIterations)
    mvm.saveModel()
  }

  def trainClassification(
    input: RDD[LabeledPoint],
    views: Array[Long],
    rank: Int,
    numIterations: Int,
    stepSize: Double,
    regParam: Double,
    miniBatch: Int = 100,
    useAdaGrad: Boolean = false,
    samplingFraction: Double = 1D,
    eta: Double = 1E-4): MVMModel = {
    val data = input.map { case LabeledPoint(label, features) =>
      LabeledPoint(if (label > 0D) 1D else 0D, features)
    }
    val mvm = new MVMClassification(data, views, rank, stepSize, regParam, miniBatch,
      useAdaGrad, samplingFraction, eta)
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
