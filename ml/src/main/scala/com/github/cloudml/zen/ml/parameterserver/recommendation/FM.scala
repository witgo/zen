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
import com.github.cloudml.zen.ml.recommendation.FMModel
import com.github.cloudml.zen.ml.util.{XORShiftRandom, SparkUtils, Utils}
import org.apache.commons.math3.distribution.GammaDistribution
import org.apache.commons.math3.random.{RandomGenerator, Well19937c}
import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.parameterserver.client.{VectorReader, MatrixReader, PSClient}
import org.parameterserver.protocol.matrix.{Row, RowData}
import org.parameterserver.protocol.vector.{DenseVector => PDV, SparseVector => PSV, Vector => PV}
import org.parameterserver.protocol.{Array => PSArray, DataType, DoubleArray}
import org.parameterserver.{Configuration => PSConf}

private[ml] abstract class FM(
  val data: RDD[LabeledPoint],
  val rank: Int) extends Serializable with Logging {

  import FM._

  def useAdaGrad: Boolean

  def stepSize: Double

  def regParam: (Double, Double, Double)

  def batchSize: Int

  def samplingFraction: Double = 1D

  private var innerEpoch: Int = 1
  var storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  val numSamples: Long = data.count()
  val numFeature = data.first().features.size
  @transient lazy val featureIds: RDD[Int] = {
    data.map(_.features.toSparse).flatMap(_.indices).distinct.persist(storageLevel)
  }

  protected val weightName = {
    val psClient = new PSClient(new PSConf(true))
    val maxId = featureIds.max()
    val wName = UUID.randomUUID().toString
    psClient.createMatrix(wName, false, true, maxId + 1, rank + 1, DataType.Double)
    featureIds.mapPartitionsWithIndex { case (pid, iter) =>
      val rand = new Well19937c(pid + 1173)
      iter.map { featureId =>
        // parameter point
        val parms = Array.fill(rank + 1) {
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
  protected val biasName = {
    val psClient = new PSClient(new PSConf(true))
    val wName = UUID.randomUUID().toString
    psClient.createVector(wName, true, 2, DataType.Double)
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
      val b2 = psClient.getVector(biasName, Array(1)).asInstanceOf[DoubleArray].getValues.head
      psClient.updateVector(biasName, Array(1), new DoubleArray(Array(rho * b2)))
      psClient.close()
    }
  }

  def run(iterations: Int): Unit = {
    for (epoch <- 1 to iterations) {
      // cleanGardSum(math.exp(-math.log(2D) / 40))
      logInfo(s"Start train (Iteration $epoch/$iterations)")
      val startedAt = System.nanoTime()
      val regDist = genRegDist()
      val gammaDist: Array[Double] = samplingGammaDist(innerEpoch)
      val thisStepSize = stepSize
      val pSize = data.partitions.length
      val sampledData = if (samplingFraction == 1D) {
        data
      } else {
        data.sample(withReplacement = false, samplingFraction, innerEpoch + 17)
      }.mapPartitionsWithIndex { case (pid, iter) =>
        val rand = new XORShiftRandom((innerEpoch + 119) * (pSize + 1) + pid)
        iter.map(t => (rand.nextInt(), t))
      }.sortByKey().map(_._2)
      val costSum = sampledData.mapPartitionsWithIndex { case (pid, iter) =>
        val psClient = new PSClient(new PSConf(true))
        val matrixReader = new MatrixReader(psClient, weightName)
        val rand = new XORShiftRandom(innerEpoch * (pSize + 13) + pid)
        var innerIter = 0
        val newIter = iter.grouped(batchSize).map { samples =>
          var costSum = 0D
          val bias = getBias(psClient)
          val sampledSize = samples.length
          val featureIds = samples.map(_.features.toSparse).flatMap(_.indices).distinct.sorted.toArray
          val features = rowData2Array(matrixReader.read(featureIds))
          val f2i = featureIds.zipWithIndex.toMap
          val grad = new Array[VD](featureIds.length)
          var gradBias = 0D
          samples.foreach { sample =>
            val ssv = sample.features.toSparse
            val indices = ssv.indices
            val values = ssv.values
            val arr = forwardSample(indices, values, f2i, features)
            val (multi, loss) = multiplier(bias, arr, sample.label)
            gradBias += multi.head
            costSum += loss
            backwardSample(indices, values, multi, f2i, features, grad)
          }

          matrixReader.clear()
          grad.foreach(g => g.indices.foreach(i => g(i) /= sampledSize))
          gradBias /= sampledSize
          gradBias = l2(gammaDist, regDist, rand, bias, featureIds, features, grad, gradBias)

          innerIter += 1
          updateWeight(gradBias, grad, features, featureIds, psClient, rand, thisStepSize, innerIter)
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

  private[ml] def getBias(psClient: PSClient = null): Double = {
    val newPsClient = if (psClient == null) Some(new PSClient(new PSConf(true))) else None
    val bias = newPsClient.getOrElse(psClient).getVector(biasName, Array(0)).
      asInstanceOf[DoubleArray].getValues.head
    newPsClient.foreach(_.close())
    bias
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
      val factors = features(fo)
      if (grad(fo) == null) grad(fo) = new Array[Double](rank + 2)
      backward(rank, value, multi, multi.head, factors, grad(fo))
      i += 1
    }
  }

  private def genRegDist(): VD = {
    val (regParam0, regParam1, regParam2) = regParam
    val beta = 1D / 300D
    var alpha = regParam2 * 300D
    var regRand = new GammaDistribution(new Well19937c(Utils.random.nextLong()), alpha, beta)
    val regDist = regRand.sample(rank + 2)
    alpha = regParam1 * 300D
    regRand = new GammaDistribution(new Well19937c(Utils.random.nextLong()), alpha, beta)
    regDist(1) = regRand.sample()
    alpha = regParam0 * 300D
    regRand = new GammaDistribution(new Well19937c(Utils.random.nextLong()), alpha, beta)
    regDist(0) = regRand.sample()
    regDist
  }

  private def l2(
    gammaDist: Array[Double],
    regDist: Array[Double],
    rand: JavaRandom,
    bias: Double,
    featureIds: Array[Int],
    features: Array[VD],
    grad: Array[VD],
    gradBias: Double): Double = {
    val rankIndices = 0 until (rank + 1)
    features.indices.foreach { i =>
      val w = features(i)
      val g = grad(i)
      val deg = g.last
      rankIndices.foreach { offset =>
        assert(!(g(offset).isNaN || g(offset).isInfinity))
        val reg = regDist(offset + 1)
        val gamma = gammaDist(offset)
        g(offset) += deg * (reg + rand.nextGaussian() * gamma) * w(offset)
        // g(offset) += deg * (reg * w(offset) + rand.nextGaussian() * gamma)
        // g(offset) += deg * reg * w(offset)
      }
    }
    gradBias + regDist.head * bias
  }

  private def forwardSample(
    indices: Array[Int],
    values: Array[Double],
    fId2Offset: Map[Int, Int],
    features: Array[VD]): Array[Double] = {
    val arr = new Array[Double](rank * 2 + 1)
    var i = 0
    while (i < indices.length) {
      val featureId = indices(i)
      val value = values(i)
      val fo = fId2Offset(featureId)
      val w = features(fo)
      forward(rank, arr, value, w)
      i += 1
    }
    arr
  }

  def updateWeight(
    gradBias: Double,
    grad: Array[VD],
    features: Array[VD],
    featuresIds: Array[Int],
    psClient: PSClient,
    rand: JavaRandom,
    stepSize: Double,
    iter: Int): Unit = {
    val rankIndices = 0 until (rank + 1)
    val newGrad = Array.fill(featuresIds.length)(new VD(rank + 1))
    var newGradBias = 0D
    if (useAdaGrad) {
      val (gBias, bias2Sum, g2Sum) = adaGrad(gradBias, grad, featuresIds, psClient)
      // val nuEpsilon = stepSize * math.pow(iter + 48D, -0.51)
      // val nuEpsilon = stepSize / (math.sqrt(iter + 15D) * math.log(iter + 14D))
      // val nuEpsilon = 2D * stepSize / numSamples
      val nuEpsilon = stepSize * eta
      newGradBias = -stepSize * gBias + rand.nextGaussian() * math.sqrt(nuEpsilon / bias2Sum)
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
    } else {
      val epsilon = stepSize * math.pow(iter + 48D, -0.51)
      // val epsilon = stepSize / (math.sqrt(iter + 15D) * math.log(iter + 14D))
      val nuEpsilon = epsilon * eta
      newGradBias = -stepSize * gradBias + rand.nextGaussian() * math.sqrt(nuEpsilon)
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
    psClient.add2Vector(biasName, Array(0), new DoubleArray(Array(newGradBias)))
    psClient.add2Matrix(weightName, array2RowData(newGrad, featuresIds))
  }

  def adaGrad(
    gradBias: Double,
    grad: Array[VD],
    featuresIds: Array[Int],
    psClient: PSClient): (Double, Double, Array[VD]) = {
    val rankIndices = 0 until (rank + 1)
    val t2Sum = grad.map { g =>
      val t2 = new Array[Double](rank + 1)
      rankIndices.foreach { i =>
        t2(i) = g(i) * g(i)
      }
      t2
    }
    val grad2Bias = gradBias * gradBias

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
    var bias2Sum = psClient.getVector(biasName, Array(1)).asInstanceOf[DoubleArray].getValues.head
    bias2Sum = 1E-6 + math.sqrt(bias2Sum + grad2Bias)

    psClient.add2Vector(biasName, Array(1), new DoubleArray(Array(grad2Bias)))
    psClient.add2Matrix(gardSumName, array2RowData(t2Sum, featuresIds))
    (gradBias / bias2Sum, bias2Sum, g2Sum)
  }

  private def samplingGammaDist(innerEpoch: Int): Array[Double] = {
    val rankIndices = 0 until (rank + 1)
    val dist = features.aggregate(new Array[Double](rank + 2))((arr, a) => {
      val (_, weight) = a
      arr(rank + 1) += 1
      for (offset <- rankIndices) {
        arr(offset) += math.pow(weight(offset), 2)
      }
      arr
    }, reduceInterval)
    val gamma = new Array[Double](rank + 1)
    val alpha = 1D
    val beta = 1D
    val rand = new Well19937c(innerEpoch)
    val shape = alpha + dist.last / 2D
    for (offset <- rankIndices) {
      val scale = beta + dist(offset) / 2D
      val rng = new GammaDistribution(rand, shape, scale)
      gamma(offset) = math.sqrt(1D / rng.sample())
    }
    gamma
  }

  def saveModel(): FMModel

  def predict(bias: Double, arr: Array[Double]): Double

  def forward(rank: Int, arr: Array[Double], z: ED, w: VD): Array[Double] = {
    forwardInterval(rank, arr, z, w)
  }

  def multiplier(bias: Double, arr: Array[Double], label: Double): (Array[Double], Double)

  def lossSum(loss: Double, numSamples: Long): Double

  def backward(rank: Int, x: ED, arr: VD, multi: Double, factors: VD, grad: VD): Array[Double] = {
    backwardInterval(rank, x, arr, multi, factors, grad)
  }

}

class FMRegression(
  @transient override val data: RDD[LabeledPoint],
  override val rank: Int,
  override val stepSize: Double,
  override val regParam: (Double, Double, Double),
  override val batchSize: Int,
  override val useAdaGrad: Boolean,
  override val samplingFraction: Double = 1D,
  override val eta: Double = 1E-6) extends FM(data, rank) {

  import FM._

  require(samplingFraction > 0 && samplingFraction <= 1,
    s"Sampling fraction ($samplingFraction) must be > 0 and <= 1")
  val max = data.map(_.label).max
  val min = data.map(_.label).min

  override def saveModel(): FMModel = {
    new FMModel(rank, getBias(), false, features.map(t => (t._1.toLong, t._2)), max, min)
  }

  override def predict(bias: Double, arr: Array[Double]): Double = {
    var result = predictInterval(rank, bias, arr)
    result = Math.max(result, min)
    result = Math.min(result, max)
    result
  }

  override def lossSum(loss: Double, numSamples: Long): Double = {
    math.sqrt(loss / numSamples)
  }

  override def multiplier(bias: Double, arr: Array[Double], label: Double): (Array[Double], Double) = {
    val multi = sumInterval(rank, arr)
    val sum = predict(bias, arr)
    val diff = sum - label
    multi(0) = diff * 2.0
    (multi, diff * diff)
  }
}

class FMClassification(
  @transient override val data: RDD[LabeledPoint],
  override val rank: Int,
  override val stepSize: Double,
  override val regParam: (Double, Double, Double),
  override val batchSize: Int,
  override val useAdaGrad: Boolean,
  override val samplingFraction: Double = 1D,
  override val eta: Double = 1E-6) extends FM(data, rank) {

  import FM._

  require(samplingFraction > 0 && samplingFraction <= 1,
    s"Sampling fraction ($samplingFraction) must be > 0 and <= 1")

  override def saveModel(): FMModel = {
    new FMModel(rank, getBias(), true, features.map(t => (t._1.toLong, t._2)), 1D, 0D)
  }

  override def predict(bias: Double, arr: Array[Double]): Double = {
    val result = predictInterval(rank, bias, arr)
    sigmoid(result)
  }

  @inline private def sigmoid(x: Double): Double = {
    1D / (1D + math.exp(-x))
  }

  override def lossSum(loss: Double, numSamples: Long): Double = {
    loss / numSamples
  }

  override def multiplier(bias: Double, arr: Array[Double], label: Double): (Array[Double], Double) = {
    val z = predict(bias, arr)
    val multi = sumInterval(rank, arr)
    val diff = sigmoid(z) - label
    multi(0) = diff
    (multi, Utils.log1pExp(if (label > 0D) -z else z))
  }
}

object FM {
  private[ml] type ED = Double
  private[ml] type VD = Array[Double]

  def trainRegression(
    input: RDD[LabeledPoint],
    rank: Int,
    numIterations: Int,
    stepSize: Double,
    regParam: (Double, Double, Double),
    miniBatch: Int = 100,
    useAdaGrad: Boolean = false,
    samplingFraction: Double = 1D,
    eta: Double = 1E-4): FMModel = {
    val mvm = new FMRegression(input, rank, stepSize, regParam, miniBatch,
      useAdaGrad, samplingFraction, eta)
    mvm.run(numIterations)
    mvm.saveModel()
  }

  def trainClassification(
    input: RDD[LabeledPoint],
    rank: Int,
    numIterations: Int,
    stepSize: Double,
    regParam: (Double, Double, Double),
    miniBatch: Int = 100,
    useAdaGrad: Boolean = false,
    samplingFraction: Double = 1D,
    eta: Double = 1E-4): FMModel = {
    val data = input.map { case LabeledPoint(label, features) =>
      LabeledPoint(if (label > 0D) 1D else 0D, features)
    }
    val mvm = new FMClassification(data, rank, stepSize, regParam, miniBatch,
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

  /**
    * arr[0] = \sum_{j=1}^{n}w_{j}x_{i}
    * arr[f] = \sum_{i=1}^{n}v_{i,f}x_{i} f属于 [1,rank]
    * arr[k] = \sum_{i=1}^{n} v_{i,k}^{2}x_{i}^{2} k属于 (rank,rank * 2 + 1]
    */
  private[ml] def predictInterval(rank: Int, bias: Double, arr: VD): ED = {
    var sum = 0D
    var i = 1
    while (i <= rank) {
      sum += math.pow(arr(i), 2) - arr(rank + i)
      i += 1
    }
    bias + arr(0) + 0.5 * sum
  }


  /**
    * arr[0] = \sum_{j=1}^{n}w_{j}x_{i}
    * arr[f] = \sum_{i=1}^{n}v_{i,f}x_{i}, f belongs to  [1,rank]
    * arr[k] = \sum_{i=1}^{n} v_{i,k}^{2}x_{i}^{2}, k belongs to  (rank,rank * 2 + 1]
    */
  private[ml] def forwardInterval(rank: Int, arr: Array[Double], x: ED, w: VD): VD = {
    arr(0) += x * w(0)
    var i = 1
    while (i <= rank) {
      arr(i) += x * w(i)
      arr(rank + i) += math.pow(x, 2) * math.pow(w(i), 2)
      i += 1
    }
    arr
  }

  /**
    * sumM = \sum_{j=1}^{n}v_{j,f}x_{j}
    */
  private[ml] def backwardInterval(
    rank: Int,
    x: ED,
    sumM: VD,
    multi: ED,
    factors: VD,
    m: VD): VD = {
    m(0) += x * multi
    var i = 1
    while (i <= rank) {
      val grad = sumM(i) * x - factors(i) * math.pow(x, 2)
      m(i) += grad * multi
      i += 1
    }
    m(m.length - 1) += 1
    m
  }

  private[ml] def sumInterval(rank: Int, arr: Array[Double]): VD = {
    val result = new Array[Double](rank + 1)
    var i = 1
    while (i <= rank) {
      result(i) = arr(i)
      i += 1
    }
    result
  }
}
