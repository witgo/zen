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
import org.parameterserver.client.{VectorClient, MatrixClient, PSClient}
import org.parameterserver.protocol.matrix.{Row, RowData}
import org.parameterserver.protocol.vector.{DenseVector => PDV, SparseVector => PSV, Vector => PV}
import org.parameterserver.protocol.{Array => PSArray, DataType, DoubleArray}
import org.parameterserver.{Configuration => PSConf}

private[ml] abstract class FM(
  val data: RDD[LabeledPoint],
  val rank: Int) extends Serializable with Logging {

  import FM._

  def stepSize: Double

  def regParam: (Double, Double, Double)

  def batchSize: Int

  def samplingFraction: Double = 1D

  private var innerEpoch: Int = 1
  var storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK

  val numSamples: Long = data.count()
  val numFeature = data.first().features.size
  val biasId = numFeature + 1
  @transient lazy val featureIds: RDD[Int] = {
    data.map(_.features.toSparse).flatMap(_.indices).distinct.
      union(data.sparkContext.makeRDD(Seq(biasId))).persist(storageLevel)
  }

  protected val weightName = {
    val psClient = new PSClient(new PSConf(true))
    val maxId = featureIds.max()
    val wName = UUID.randomUUID().toString
    psClient.createMatrix(wName, false, true, maxId + 2, rank + 1, DataType.Double)
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

  protected val gardSumName: String = {
    val psClient = new PSClient(new PSConf(true))
    val gName = UUID.randomUUID().toString
    psClient.createMatrix(gName, weightName)
    // psClient.matrixAxpby(gName, 1D, weightName, 0D)
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
        val matrixReader = new MatrixClient(psClient, weightName)
        val rand = new XORShiftRandom(innerEpoch * (pSize + 13) + pid)
        var innerIter = 0
        val newIter = iter.grouped(batchSize).map { samples =>
          var costSum = 0D
          val sampledSize = samples.length
          val featureIds = (samples.flatMap(_.features.toSparse.indices).distinct :+ biasId).toArray
          val features = rowData2Array(matrixReader.read(featureIds))
          val bias = getBias(featureIds, features)
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

          grad(grad.length - 1) = new Array[Double](rank + 2)
          grad(grad.length - 1)(0) = gradBias
          matrixReader.clear()
          grad.foreach(g => g.indices.foreach(i => g(i) /= sampledSize))
          l2(featureIds, features, grad)

          innerIter += 1
          updateWeight(grad, features, featureIds, psClient, rand, thisStepSize, innerIter)
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

  private[ml] def getBias(featureIds: Array[Int], features: Array[VD]): Double = {
    assert(featureIds.last == biasId)
    return features.last.head
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

  private def l2(
    featureIds: Array[Int],
    features: Array[VD],
    grad: Array[VD]): Unit = {
    val bias = getBias(featureIds, features)
    val (regParam0, regParam1, regParam2) = regParam
    val rankIndices = 0 until (rank + 1)
    for (i <- 0 until featureIds.length - 1) {
      val w = features(i)
      val g = grad(i)
      val deg = g.last
      rankIndices.foreach { offset =>
        assert(!(g(offset).isNaN || g(offset).isInfinity))
        val reg = if (offset == 0) regParam1 else regParam2
        g(offset) += deg * reg * w(offset)
      }
    }
    grad(grad.length - 1)(0) += regParam0 * bias
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

  protected def getWeight(): (Double, RDD[(Long, VD)]) = {
    val weights = features.map(t => (t._1.toLong, t._2)).persist(storageLevel)
    (weights.filter(_._1 == biasId).first()._2.head,
      weights.filter(_._1 < biasId))
  }

  def updateWeight(
    grad: Array[VD],
    features: Array[VD],
    featuresIds: Array[Int],
    psClient: PSClient,
    rand: JavaRandom,
    stepSize: Double,
    iter: Int): Unit = {
    val rankIndices = 0 until (rank + 1)
    val newGrad = Array.fill(featuresIds.length)(new VD(rank + 1))
    val g2Sum = adaGrad(grad, featuresIds, psClient)
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
    psClient.add2Matrix(weightName, array2RowData(newGrad, featuresIds))
  }

  def adaGrad(
    grad: Array[VD],
    featuresIds: Array[Int],
    psClient: PSClient): Array[VD] = {
    val rankIndices = 0 until (rank + 1)
    val t2Sum = grad.map { g =>
      val t2 = new Array[Double](rank + 1)
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
        sum(offset) = math.sqrt(sum(offset) + t2(offset) + 1E-8)
        g(offset) /= sum(offset)
      }
    }

    psClient.add2Matrix(gardSumName, array2RowData(t2Sum, featuresIds))
    g2Sum
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
  override val samplingFraction: Double = 1D,
  override val eta: Double = 1E-6) extends FM(data, rank) {

  import FM._

  require(samplingFraction > 0 && samplingFraction <= 1,
    s"Sampling fraction ($samplingFraction) must be > 0 and <= 1")
  val max = data.map(_.label).max
  val min = data.map(_.label).min

  override def saveModel(): FMModel = {
    val (bias, features) = getWeight()
    new FMModel(rank, bias, false, features, max, min)
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
  override val samplingFraction: Double = 1D,
  override val eta: Double = 1E-6) extends FM(data, rank) {

  import FM._

  require(samplingFraction > 0 && samplingFraction <= 1,
    s"Sampling fraction ($samplingFraction) must be > 0 and <= 1")

  override def saveModel(): FMModel = {
    val (bias, features) = getWeight()
    new FMModel(rank, bias, true, features, 1D, 0D)
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
    val diff = z - label
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
    samplingFraction: Double = 1D,
    eta: Double = 1E-4): FMModel = {
    val mvm = new FMRegression(input, rank, stepSize, regParam, miniBatch, samplingFraction, eta)
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
    samplingFraction: Double = 1D,
    eta: Double = 1E-4): FMModel = {
    val data = input.map { case LabeledPoint(label, features) =>
      LabeledPoint(if (label > 0D) 1D else 0D, features)
    }
    val mvm = new FMClassification(data, rank, stepSize, regParam, miniBatch, samplingFraction, eta)
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
