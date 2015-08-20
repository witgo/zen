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

import java.util.{Random => JavaRandom}

import breeze.optimize.{BacktrackingLineSearch, StepSizeUnderflow, StrongWolfeLineSearch, DiffFunction}
import breeze.linalg.{DenseMatrix => BDM}
import com.github.cloudml.zen.ml.DBHPartitioner
import com.github.cloudml.zen.ml.recommendation.MVM._
import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.ml.util.Utils
import org.apache.spark.Logging
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.{EdgeRDDImpl, GraphImpl}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.math._

/**
 * Multi-view Machines :
 * \hat{y}(x) :=\sum_{i_1 =1}^{I_i +1} ...\sum_{i_m =1}^{I_m +1}
 * (\prod_{v=1}^{m} z_{i_v}^{(v)})(\sum_{f=1}^{k}\prod_{v=1}^{m}a_{i_{v,j}}^{(v)})
 * :=  \sum_{f}^{k}(\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
 * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
 *
 * derivative of the model :
 * \frac{\partial \hat{y}(x|\Theta )}{\partial\theta} :=z_{i_{v}}^{(v)}
 * (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ...
 * (\sum_{i_{v-1} =1}^{I_{v-1}+1}z_{i_{v-1}}^{({v-1})}a_{i_{v-1},j}^{({v-1})})
 * (\sum_{i_{v+1} =1}^{I_{v+1}+1}z_{i_{v+1}}^{({v+1})}a_{i_{v+1},j}^{({v+1})}) ...
 * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
 */
private[ml] abstract class MVM extends Serializable with Logging {

  protected val checkpointInterval = 10
  protected var numFeatures: Long = 0
  protected var numSamples: Long = 0

  // SGD
  @transient protected var dataSet: Graph[VD, ED] = null
  @transient protected var multi: VertexRDD[VD] = null
  @transient protected var previousGrad: VertexRDD[VD] = null
  @transient protected var previousVertices: VertexRDD[VD] = null
  @transient protected var vertices: VertexRDD[VD] = null
  @transient protected var edges: EdgeRDD[ED] = null
  @transient protected var delta: VertexRDD[(IndexedSeq[VD], IndexedSeq[VD])] = null
  @transient protected var previousDelta: VertexRDD[(IndexedSeq[VD], IndexedSeq[VD])] = null
  @transient private var innerIter: Int = 1

  var stepSize: Double
  lazy val rankIndices = 0 until rank

  def setDataSet(data: Graph[VD, ED]): this.type = {
    vertices = data.vertices
    edges = data.edges.asInstanceOf[EdgeRDDImpl[ED, _]].mapEdgePartitions { (pid, part) =>
      part.withoutVertexAttributes[VD]
    }.setName("edges").persist(storageLevel)
    if (vertices.getStorageLevel == StorageLevel.NONE) {
      vertices.persist(storageLevel)
    }
    if (edges.sparkContext.getCheckpointDir.isDefined) {
      edges.checkpoint()
      edges.count()
    }
    data.edges.unpersist(blocking = false)
    dataSet = GraphImpl.fromExistingRDDs(vertices, edges)
    numFeatures = features.count()
    numSamples = samples.count()
    logInfo(s"$numFeatures features, $numSamples samples in the data")
    this
  }

  def views: Array[Long]

  def regParam: Double

  def rank: Int

  def useWeightedLambda: Boolean

  def storageLevel: StorageLevel

  def samples: VertexRDD[VD] = {
    dataSet.vertices.filter(t => isSampleId(t._1))
  }

  def features: VertexRDD[VD] = {
    dataSet.vertices.filter(t => !isSampleId(t._1))
  }

  def run(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      logInfo(s"Start train (Iteration $iter/$iterations)")
      val startedAt = System.nanoTime()
      val margin = forward(dataSet, innerIter)
      val (_, rmse, thisMulti) = multiplier(margin, iter)
      multi = thisMulti
      val gradient = backward(dataSet, multi, innerIter)
      assert(gradient.count == numFeatures)
      Option(previousDelta).foreach(_.unpersist(blocking = false))
      previousDelta = delta
      delta = updateDelta(features, previousVertices, gradient, previousGrad, delta, 10, innerIter)
      delta.count()
      Option(previousGrad).foreach(_.unpersist(blocking = false))
      previousGrad = gradient

      val dir = twoLoopRecursion(gradient, delta, innerIter)
      stepSize = determineStepSize(dir, innerIter)
      Option(previousVertices).foreach(_.unpersist(blocking = false))
      previousVertices = vertices
      vertices = updateWeight(dataSet, dir, stepSize, innerIter)
      checkpointVertices(vertices)
      vertices.count()
      dataSet = GraphImpl.fromExistingRDDs(vertices, edges)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      println(s"(Iteration $iter/$iterations) stepSize: $stepSize")
      println(f"(Iteration $iter/$iterations) loss:                     $rmse%1.6G")
      logInfo(s"End  train (Iteration $iter/$iterations) takes:         $elapsedSeconds")

      gradient.unpersist(blocking = false)
      dir.unpersist(blocking = false)
      margin.unpersist(blocking = false)
      multi.unpersist(blocking = false)
      innerIter += 1
    }
  }

  def saveModel(): MVMModel = {
    new MVMModel(rank, views, false, features.mapValues(arr => arr.slice(0, arr.length - 1)))
  }

  protected[ml] def forward(dataSet: Graph[VD, ED], iter: Int): VertexRDD[Array[Double]] = {
    dataSet.aggregateMessages[Array[Double]](ctx => {
      val sampleId = ctx.dstId
      val featureId = ctx.srcId
      val viewId = featureId2viewId(featureId, views)
      val result = forwardInterval(rank, views.length, viewId, ctx.attr, ctx.srcAttr)
      ctx.sendToDst(result)
    }, reduceInterval, TripletFields.Src).setName(s"margin-$iter").persist(storageLevel)
  }

  protected def predict(arr: Array[Double]): Double

  protected def multiplier(q: VertexRDD[VD], iter: Int): (Long, Double, VertexRDD[VD])

  protected def backward(
    dataSet: Graph[VD, ED],
    multi: VertexRDD[VD],
    iter: Int): VertexRDD[VD] = {
    GraphImpl.fromExistingRDDs(multi, edges).aggregateMessages[VD](ctx => {
      // val sampleId = ctx.dstId
      val featureId = ctx.srcId
      val x = ctx.attr
      val arr = ctx.dstAttr
      val viewId = featureId2viewId(featureId, views)
      val m = backwardInterval(rank, viewId, x, arr, arr.last)
      ctx.sendToSrc(m)
    }, reduceInterval, TripletFields.Dst).mapValues { a =>
      //  val deg = a.last
      //  a.slice(0, rank).map(_ / deg)
      a.slice(0, rank).map(_ / numSamples)
    }.innerJoin(dataSet.vertices) { case (_, g, w) =>
      val wd = if (useWeightedLambda) w.last / (numSamples + 1.0) else 1.0
      rankIndices.foreach { i =>
        g(i) += wd * regParam * w(i)
      }
      g
    }.setName(s"gradient-$iter").persist(storageLevel)
  }

  // Updater for elastic net regularized problems
  protected def updateWeight(
    dataSet: Graph[VD, ED],
    gradient: VertexRDD[Array[Double]],
    stepSize: Double,
    iter: Int): VertexRDD[VD] = {
    dataSet.vertices.leftJoin(gradient) { (_, attr, gradient) =>
      gradient match {
        case Some(grad) =>
          val weight = attr.clone()
          rankIndices.foreach { i =>
            weight(i) += stepSize * grad(i)
          }
          weight
        case None => attr
      }
    }.setName(s"vertices-$iter").persist(storageLevel)
  }

  def updateDelta(
    vertices: VertexRDD[VD],
    previousVertices: VertexRDD[VD],
    grad: VertexRDD[VD],
    previousGrad: VertexRDD[VD],
    delta: VertexRDD[(IndexedSeq[VD], IndexedSeq[VD])],
    m: Int, iter: Int): VertexRDD[(IndexedSeq[VD], IndexedSeq[VD])] = {
    val newDelta = Option(delta).getOrElse(features.mapValues(_ => (IndexedSeq.empty[VD], IndexedSeq.empty[VD])))
    if (previousVertices == null) return newDelta
    val s = vertices.filter(t => !isSampleId(t._1)).innerJoin(previousVertices) { (id, s_k1, s_k) =>
      rankIndices.map { i =>
        val diff = s_k1(i) - s_k(i)
        assert(diff != 0.0)
        diff
      }.toArray
    }
    val y = grad.innerJoin(previousGrad) { (id, y_k1, y_k) =>
      rankIndices.map { i =>
        val diff = y_k1(i) - y_k(i)
        assert(diff != 0.0)
        diff
      }.toArray
    }
    newDelta.innerJoin(s.join(y)) { case (_, (memStep, memGradDelta), (s, y)) =>
      ((s +: memStep).take(m), (y +: memGradDelta).take(m))
    }.setName(s"delta-$iter").persist(storageLevel)
  }

  // Algorithm 3 in the paper: Large-scale L-BFGS using MapReduce
  def twoLoopRecursion(
    grad: VertexRDD[VD],
    delta: VertexRDD[(IndexedSeq[VD], IndexedSeq[VD])],
    iter: Int): VertexRDD[VD] = {
    val m = delta.first()._2._2.size
    val rankIndices = 0 until rank
    val mIndices = 0 until m
    val mPlusIndices = 0 until 2 * m + 1
    val bVectors = grad.innerJoin(delta) { case (_, g, (s, y)) =>
      s.reverse ++ y.reverse :+ g
    }
    val z = bVectors.aggregate(BDM.zeros[Double](2 * m + 1, 2 * m + 1))(seqOp = (arr, v) => {
      val b = v._2
      mPlusIndices.foreach { i =>
        mPlusIndices.foreach { j =>
          rankIndices.foreach { rankId =>
            arr(i, j) += b(i)(rankId) * b(j)(rankId)
          }
        }
      }
      arr
    }, _ :+= _)

    val d = new Array[Double](2 * m + 1)
    val a = new Array[Double](m)

    d(2 * m) = -1D

    mIndices.reverse.foreach { j =>
      var sum = 0D
      mPlusIndices.foreach { l =>
        sum += d(l) * z(l, j)
      }
      a(j) = sum / z(j, m + j)
      d(m + j) -= a(j)
    }

    mPlusIndices.foreach { j =>
      d(j) *= z(m - 1, 2 * m - 1) / z(2 * m - 1, 2 * m - 1)
    }

    mIndices.foreach { j =>
      var sum = 0D
      mPlusIndices.foreach { l =>
        sum += d(l) * z(m + j, l)
      }
      val beta = sum / z(j, m + j)
      d(j) += a(j) - beta
    }
    // println(d.mkString(" "))

    val direction = bVectors.mapValues { (_, b) =>
      val dir = new Array[Double](rank)
      rankIndices.foreach { rankId =>
        mPlusIndices.foreach { l =>
          dir(rankId) += b(l)(rankId) * d(l)
        }
      }
      dir
    }

    direction.setName(s"direction-$iter").persist(storageLevel)
  }

  def functionFromSearchDirection(
    direction: VertexRDD[VD],
    vertices: VertexRDD[VD],
    edges: EdgeRDD[ED]): DiffFunction[Double] = new DiffFunction[Double] {
    /** calculates the value at a point */
    override def valueAt(alpha: Double): Double = {
      calculate(alpha)._1
    }

    /** calculates the gradient at a point */
    override def gradientAt(alpha: Double): Double = {
      calculate(alpha)._2
    }

    /** Calculates both the value and the gradient at a point */
    def calculate(alpha: Double): (Double, Double) = {
      val rankIndices = 0 until rank
      val newVertices = vertices.leftJoin(direction) { case (vid, x, dir) =>
        dir match {
          case Some(d) =>
            val newX = x.clone()
            val d = dir.get
            rankIndices.foreach { rankId =>
              newX(rankId) = x(rankId) + alpha * d(rankId)
            }
            newX
          case _ =>
            x
        }
      }
      direction.foreach(_._2.foreach(t => assert(!t.isNaN)))
      newVertices.foreach(_._2.foreach(t => assert(!t.isNaN)))

      val wl = useWeightedLambda
      val r = regParam
      val ns = numSamples
      val regVal = newVertices.aggregate(0.0)(seqOp = (a, b) => {
        if (isSampleId(b._1)) {
          a
        } else {
          val w = b._2
          val wd = if (wl) w.last / (ns + 1.0) else 1.0
          var s = 0.0
          rankIndices.foreach { i =>
            s += pow(w(i), 2.0)
          }
          a + 0.5 * wd * r * s
        }
      }, _ + _)

      println(f"regVal: $regVal%1.6f")
      if (regParam != 0.0) assert(regParam != 0.0)

      val dataSet = GraphImpl.fromExistingRDDs(newVertices, edges)
      val margin = forward(dataSet, innerIter)

      margin.foreach(_._2.foreach(t => assert(!t.isNaN)))

      val (_, rmse, thisMulti) = multiplier(margin, innerIter)
      val gradient = backward(dataSet, thisMulti, innerIter)

      thisMulti.foreach(_._2.foreach(t => assert(!t.isNaN)))

      val dd = features.join(gradient).join(direction).map { case (_, ((w, g), d)) =>
        var s = 0.0
        rankIndices.foreach { rankId =>
          s += d(rankId) * g(rankId)
        }
        s
      }.aggregate(0.0)(_ + _, _ + _)

      margin.unpersist(blocking = false)
      thisMulti.unpersist(blocking = false)
      gradient.unpersist(blocking = false)
      println(f"calculate: $alpha%1.6G -> $rmse%1.6G -> ${rmse + regVal}%1.6G -> $dd%1.6G")
      (rmse + regVal, dd)
    }
  }

  def determineStepSize(dir: VertexRDD[VD], iter: Int): Double = {
    val ff = functionFromSearchDirection(dir, vertices, edges)
    val init = if (innerIter < 3) 0.1 else 1.0
    val search = new BacktrackingLineSearch(ff.valueAt(init), enforceWolfeConditions = false,
      enforceStrongWolfeConditions = false)
    //    val search = new StrongWolfeLineSearch(maxZoomIter = 10, maxLineSearchIter = 10)
    val alpha = search.minimize(ff, init)
    alpha
    // 0.1 / sqrt(iter)
  }

  protected def checkpointGradientSum(delta: VertexRDD[(Array[Double], Array[Double])]): Unit = {
    val sc = delta.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      delta.checkpoint()
    }
  }

  protected def checkpointVertices(vertices: VertexRDD[VD]): Unit = {
    val sc = vertices.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      vertices.checkpoint()
    }
  }
}

class MVMClassification(
  @transient _dataSet: Graph[VD, ED],
  override var stepSize: Double,
  val views: Array[Long],
  val regParam: Double,
  val elasticNetParam: Double,
  val rank: Int,
  val useAdaGrad: Boolean,
  val useWeightedLambda: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends MVM {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    views: Array[Long],
    regParam: Double = 1e-2,
    elasticNetParam: Double = 0,
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    useWeightedLambda: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, views, rank, storageLevel), stepSize, views, regParam,
      elasticNetParam, rank, useAdaGrad, useWeightedLambda, miniBatchFraction, storageLevel)
  }

  setDataSet(_dataSet)

  // assert(rank > 1, s"rank $rank less than 2")

  override protected def predict(arr: Array[Double]): Double = {
    val result = predictInterval(rank, arr)
    sigmoid(result)
  }

  @inline private def sigmoid(x: Double): Double = {
    1d / (1d + math.exp(-x))
  }

  override def saveModel(): MVMModel = {
    new MVMModel(rank, views, true, features.mapValues(arr => arr.slice(0, arr.length - 1)))
  }

  override protected def multiplier(q: VertexRDD[VD], iter: Int): (Long, Double, VertexRDD[VD]) = {
    val accNumSamples = q.sparkContext.accumulator(1L)
    val accLossSum = q.sparkContext.accumulator(0.0)
    val multi = dataSet.vertices.leftJoin(q) { (vid, data, deg) =>
      deg match {
        case Some(m) =>
          val y = data.head
          // val diff = predict(m) - y
          val arr = sumInterval(rank, m)
          val z = arr.last
          val diff = sigmoid(z) - y
          accNumSamples += 1L
          accLossSum += Utils.log1pExp(if (y > 0.0) -z else z)
          arr(arr.length - 1) = diff
          arr
        case _ => data
      }
    }.setName(s"multiplier-$iter").persist(storageLevel)
    multi.count()
    val numSamples = accNumSamples.value
    val lossSum = accLossSum.value
    (numSamples, lossSum / numSamples, multi)
  }

}

class MVMRegression(
  @transient _dataSet: Graph[VD, ED],
  override var stepSize: Double,
  val views: Array[Long],
  val regParam: Double,
  val elasticNetParam: Double,
  val rank: Int,
  val useAdaGrad: Boolean,
  val useWeightedLambda: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends MVM {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    stepSize: Double = 1e-2,
    views: Array[Long],
    regParam: Double = 1e-2,
    elasticNetParam: Double = 0,
    rank: Int = 20,
    useAdaGrad: Boolean = true,
    useWeightedLambda: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, views, rank, storageLevel), stepSize, views, regParam, elasticNetParam, rank,
      useAdaGrad, useWeightedLambda, miniBatchFraction, storageLevel)
  }

  setDataSet(_dataSet)

  // assert(rank > 1, s"rank $rank less than 2")

  // val max = samples.map(_._2.head).max
  // val min = samples.map(_._2.head).min

  override protected def predict(arr: Array[Double]): Double = {
    var result = predictInterval(rank, arr)
    // result = Math.max(result, min)
    // result = Math.min(result, max)
    result
  }

  override protected def multiplier(q: VertexRDD[VD], iter: Int): (Long, Double, VertexRDD[VD]) = {
    val accNumSamples = q.sparkContext.accumulator(1L)
    val accLossSum = q.sparkContext.accumulator(0.0)
    val multi = dataSet.vertices.leftJoin(q) { (vid, data, deg) =>
      deg match {
        case Some(m) =>
          val y = data.head
          val arr = sumInterval(rank, m)
          val diff = arr.last - y
          accLossSum += pow(diff, 2)
          accNumSamples += 1L
          arr(arr.length - 1) = diff * 2.0
          arr
        case _ => data
      }
    }.setName(s"multiplier-$iter").persist(storageLevel)
    multi.count()
    val numSamples = accNumSamples.value
    val costSum = accLossSum.value
    (numSamples, sqrt(costSum / numSamples), multi)
  }
}

object MVM {
  private[ml] type ED = Double
  private[ml] type VD = Array[Double]

  /**
   * MVM Clustering
   * @param input train data
   * @param numIterations
   * @param stepSize  we recommend the step size: 1e-2 - 1e-1
   * @param regParam  elastic net regularization
   * @param elasticNetParam  we recommend 0
   * @param rank   we recommend the rank of eigenvector: 10-20
   * @param useAdaGrad use AdaGrad to train
   * @param miniBatchFraction
   * @param storageLevel  cache storage level
   * @return
   */
  def trainClassification(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    stepSize: Double,
    views: Array[Long],
    regParam: Double,
    elasticNetParam: Double,
    rank: Int,
    useAdaGrad: Boolean = true,
    useWeightedLambda: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): MVMModel = {
    val data = input.map { case (id, LabeledPoint(label, features)) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      val newLabel = if (label > 0.0) 1.0 else 0.0
      (id, LabeledPoint(newLabel, features))
    }
    val lfm = new MVMClassification(data, stepSize, views, regParam, elasticNetParam, rank,
      useAdaGrad, useWeightedLambda, miniBatchFraction, storageLevel)
    lfm.run(numIterations)
    val model = lfm.saveModel()
    model
  }

  /**
   * MVM Regression
   * @param input train data
   * @param numIterations
   * @param stepSize  we recommend the step size: 1e-2 - 1e-1
   * @param regParam    elastic net regularization
   * @param elasticNetParam  we recommend 0
   * @param rank   we recommend the rank of eigenvector: 10-20
   * @param useAdaGrad use AdaGrad to train
   * @param miniBatchFraction
   * @param storageLevel  cache storage level
   * @return
   */

  def trainRegression(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    stepSize: Double,
    views: Array[Long],
    regParam: Double,
    elasticNetParam: Double,
    rank: Int,
    useAdaGrad: Boolean = true,
    useWeightedLambda: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): MVMModel = {
    val data = input.map { case (id, labeledPoint) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      (id, labeledPoint)
    }
    val lfm = new MVMRegression(data, stepSize, views, regParam, elasticNetParam,
      rank, useAdaGrad, useWeightedLambda, miniBatchFraction, storageLevel)
    lfm.run(numIterations)
    val model = lfm.saveModel()
    model
  }

  private[ml] def initializeDataSet(
    input: RDD[(VertexId, LabeledPoint)],
    views: Array[Long],
    rank: Int,
    storageLevel: StorageLevel): Graph[VD, ED] = {
    val numFeatures = input.first()._2.features.size
    assert(numFeatures == views.last)

    val edges = input.flatMap { case (sampleId, labelPoint) =>
      // sample id
      val newId = newSampleId(sampleId)
      labelPoint.features.activeIterator.filter(_._2 != 0.0).map { case (featureId, value) =>
        assert(featureId < numFeatures)
        Edge(featureId, newId, value)
      } ++ views.indices.map { i => Edge(numFeatures + i, newId, 1D) }
    }.persist(storageLevel)
    edges.count()

    val inDegrees = edges.map(e => (e.srcId, 1L)).reduceByKey(_ + _).map {
      case (featureId, deg) =>
        (featureId, deg)
    }

    val features = edges.map(_.srcId).distinct().map { featureId =>
      // parameter point
      val parms = Array.fill(rank + 1) {
        Utils.random.nextGaussian() * 1e-1
      }
      (featureId, parms)
    }.join(inDegrees).map { case (featureId, (parms, deg)) =>
      parms(parms.length - 1) = deg
      (featureId, parms)
    }

    val vertices = (input.map { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      val label = Array(labelPoint.label)
      // label point
      (newId, label)
    } ++ features).repartition(input.partitions.length)
    vertices.persist(storageLevel)
    vertices.count()

    val dataSet = GraphImpl(vertices, edges, null.asInstanceOf[VD], storageLevel, storageLevel)
    val newDataSet = DBHPartitioner.partitionByDBH(dataSet, storageLevel)
    edges.unpersist()
    vertices.unpersist()
    newDataSet
  }

  @inline private[ml] def featureId2viewId(featureId: Long, views: Array[Long]): Int = {
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

  @inline private[ml] def newSampleId(id: Long): VertexId = {
    -(id + 1L)
  }

  @inline private[ml] def isSampleId(id: Long): Boolean = {
    id < 0
  }

  @inline private[ml] def isSampleId[T](id: (Long, T)): Boolean = {
    id._1 < 0
  }


  @inline private[ml] def isSampled(
    random: JavaRandom,
    seed: Long,
    sampleId: Long,
    iter: Int,
    mod: Int): Boolean = {
    random.setSeed(seed * sampleId)
    random.nextInt(mod) == iter % mod
  }

  @inline private[ml] def genRandom(mod: Int, iter: Int): JavaRandom = {
    val random: JavaRandom = new JavaRandom()
    random.setSeed(17425170 - iter / mod)
    random
  }

  /**
   * arr(k + v * rank)= (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)})
   * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
   */
  private[ml] def predictInterval(rank: Int, arr: VD): ED = {
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

  private[ml] def reduceInterval(a: VD, b: VD): VD = {
    var i = 0
    while (i < a.length) {
      a(i) += b(i)
      i += 1
    }
    a
  }

  private[ml] def forwardInterval(rank: Int, viewSize: Int, viewId: Int, x: ED, w: VD): VD = {
    val arr = new Array[Double](rank * viewSize)
    forwardInterval(rank, viewId, arr, x, w)
  }

  /**
   * arr = rank * viewSize
   * when f belongs to [viewId * rank ,(viewId +1) * rank)
   * arr[f] = z_{i_v}^{(1)}a_{i_{i},j}^{(1)}
   */
  private[ml] def forwardInterval(rank: Int, viewId: Int, arr: Array[Double], z: ED, w: VD): VD = {
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
    arr: VD,
    multi: ED): VD = {
    val m = new Array[Double](rank + 1)
    var i = 0
    while (i < rank) {
      m(i) = multi * x * arr(i + viewId * rank)
      i += 1
    }
    m(rank) = 1.0
    m
  }

  /**
   * arr(k + v * rank)= (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
   * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
   */
  private[ml] def sumInterval(rank: Int, arr: Array[Double]): VD = {
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
