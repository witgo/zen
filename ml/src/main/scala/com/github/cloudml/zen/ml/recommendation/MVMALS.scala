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

import com.github.cloudml.zen.ml.DBHPartitioner
import com.github.cloudml.zen.ml.recommendation.MVMALS._
import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.ml.util.{Utils, XORShiftRandom}
import org.apache.spark.Logging
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.{EdgeRDDImpl, GraphImpl}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import scala.math._

/**
 * Multi-view Machines 公式定义:
 * \hat{y}(x) :=\sum_{i_1 =1}^{I_i +1} ...\sum_{i_m =1}^{I_m +1}
 * (\prod_{v=1}^{m} z_{i_v}^{(v)})(\sum_{f=1}^{k}\prod_{v=1}^{m}a_{i_{v,j}}^{(v)})
 * :=  \sum_{f}^{k}(\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
 * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
 *
 * 其导数是:
 * \frac{\partial \hat{y}(x|\Theta )}{\partial\theta} :=z_{i_{v}}^{(v)}
 * (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ...
 * (\sum_{i_{v-1} =1}^{I_{v-1}+1}z_{i_{v-1}}^{({v-1})}a_{i_{v-1},j}^{({v-1})})
 * (\sum_{i_{v+1} =1}^{I_{v+1}+1}z_{i_{v+1}}^{({v+1})}a_{i_{v+1},j}^{({v+1})}) ...
 * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
 */
private[ml] abstract class MVMALS extends Serializable with Logging {

  protected val checkpointInterval = 30
  protected var numFeatures: Long = 0
  protected var numSamples: Long = 0

  // ALS
  @transient protected var dataSet: Graph[VD, ED] = null
  @transient protected var multi: VertexRDD[Array[Double]] = null
  @transient protected var vertices: VertexRDD[VD] = null
  @transient protected var edges: EdgeRDD[ED] = null
  @transient private var innerIter = 1

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

  def lambda: Double

  def rank: Int

  def storageLevel: StorageLevel

  def miniBatchFraction: Double

  def useWeightedLambda: Boolean

  protected[ml] def mask: Int = {
    max(1 / miniBatchFraction, 1).toInt
  }

  def samples: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 < 0)
  }

  def features: VertexRDD[VD] = {
    dataSet.vertices.filter(t => t._1 >= 0)
  }

  def run(iterations: Int): Unit = {
    for (iter <- 1 to iterations) {
      logInfo(s"Start train (Iteration $iter/$iterations)")
      val startedAt = System.nanoTime()
      val previousVertices = vertices
      val margin = forward(iter)
      val (_, costSum, thisMulti) = multiplier(margin, iter)
      multi = thisMulti
      val delta = backward(multi, iter)
      vertices = updateWeight(delta, iter)
      checkpointVertices()
      vertices.count()
      dataSet = GraphImpl.fromExistingRDDs(vertices, edges)
      val elapsedSeconds = (System.nanoTime() - startedAt) / 1e9
      println(s"(Iteration $iter/$iterations) RMSE:                     $costSum")
      logInfo(s"End  train (Iteration $iter/$iterations) takes:         $elapsedSeconds")

      previousVertices.unpersist(blocking = false)
      margin.unpersist(blocking = false)
      multi.unpersist(blocking = false)
      delta.unpersist(blocking = false)
      innerIter += 1
    }
  }

  def saveModel(): MVMModel = {
    new MVMModel(rank, views, false, features.mapValues(arr => arr.slice(0, arr.length - 1)))
  }

  protected[ml] def forward(iter: Int): VertexRDD[Array[Double]] = {
    val mod = mask
    val random = genRandom(mod, iter)
    val seed = random.nextLong()
    dataSet.aggregateMessages[Array[Double]](ctx => {
      val sampleId = ctx.dstId
      val featureId = ctx.srcId
      val viewId = featureId2viewId(featureId, views)
      if (mod == 1 || isSampled(random, seed, sampleId, iter, mod)) {
        val result = forwardInterval(rank, views.length, viewId, ctx.attr, ctx.srcAttr)
        ctx.sendToDst(result)
      }
    }, reduceInterval, TripletFields.Src).setName(s"margin-$iter").persist(storageLevel)
  }

  protected def predict(arr: Array[Double]): Double

  protected def multiplier(q: VertexRDD[VD], iter: Int): (Long, Double, VertexRDD[VD])

  protected def backward(
    multi: VertexRDD[VD],
    iter: Int): VertexRDD[VD] = {
    val mod = mask
    val random = genRandom(mod, iter)
    val seed = random.nextLong()
    GraphImpl.fromExistingRDDs(multi, edges).aggregateMessages[VD](ctx => {
      val sampleId = ctx.dstId
      val featureId = ctx.srcId
      if (mod == 1 || isSampled(random, seed, sampleId, iter, mod)) {
        val x = ctx.attr
        val arr = ctx.dstAttr
        val viewId = featureId2viewId(featureId, views)
        val m = backwardInterval(rank, viewId, x, arr, arr.last)
        ctx.sendToSrc(m)
      }
    }, reduceInterval, TripletFields.Dst).setName(s"delta-$iter").persist(storageLevel)
  }

  // Updater for CD problems
  protected def updateWeight(delta: VertexRDD[Array[Double]], iter: Int): VertexRDD[VD] = {
    // val thisViewId = Utils.random.nextInt(views.length)
    // val thisViewId = (iter / rank) % views.length
    val thisViewId = iter % views.length
    dataSet.vertices.leftJoin(delta) { (featureId, attr, gradient) =>
      gradient match {
        case Some(grad) =>
          val weight = attr
          assert(weight.length == rank + 1)
          val viewId = featureId2viewId(featureId, views)
          var i = 0
          while (i < rank) {
            // if (i == iter % rank && viewId == thisViewId) {
            if (i == (iter / views.length) % rank && viewId == thisViewId) {
              val wd = if (useWeightedLambda) weight.last / (numSamples + 1.0) else 1.0
              val h2 = grad(i + rank)
              val he = grad(i)
              val w = weight(i)
              weight(i) = (w * h2 + he) / (h2 + wd * lambda)
              if (Utils.random.nextDouble() < 1e-4) {
                // println(s"he: $he => h2: $h2 => lambda:$lambda => old weight: $w => new weight: ${weight(i)} ")
              }
            }
            i += 1
          }
          weight
        case None => attr

      }
    }.setName(s"vertices-$iter").persist(storageLevel)
  }

  protected def checkpointVertices(): Unit = {
    val sc = vertices.sparkContext
    if (innerIter % checkpointInterval == 0 && sc.getCheckpointDir.isDefined) {
      vertices.checkpoint()
    }
  }

}

class MVMALSRegression(
  @transient _dataSet: Graph[VD, ED],
  val views: Array[Long],
  val rank: Int,
  val lambda: Double,
  val useWeightedLambda: Boolean,
  val miniBatchFraction: Double,
  val storageLevel: StorageLevel) extends MVMALS {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    views: Array[Long],
    rank: Int = 20,
    lambda: Double = 5e-2,
    useWeightedLambda: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, views, rank, storageLevel), views, rank, lambda,
      useWeightedLambda, miniBatchFraction, storageLevel)
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
          assert(data.length == 1)
          val y = data.head
          val arr = sumInterval(rank, m)
          // assert(abs(arr.last -  predict(m)) < 1e-6)
          val diff = y - arr.last
          accLossSum += pow(diff, 2)
          accNumSamples += 1L
          arr(arr.length - 1) = diff
          arr
        case _ => data
      }
    }.setName(s"multiplier-$iter").persist(storageLevel)
    multi.count()
    val numSamples = accNumSamples.value
    val costSum = accLossSum.value
    (numSamples.toLong, sqrt(costSum / numSamples), multi)
  }
}

object MVMALS {
  private[ml] type ED = Double
  private[ml] type VD = Array[Double]

  /**
   * MVMALS 回归
   * @param input 训练数据
   * @param numIterations 迭代次数
   * @param rank   特征分解向量的维度推荐 10-20
   * @param miniBatchFraction  每次迭代采样比例
   * @param storageLevel   缓存级别
   * @return
   */

  def trainRegression(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    views: Array[Long],
    rank: Int,
    lambda: Double,
    useWeightedLambda: Boolean = true,
    miniBatchFraction: Double = 1.0,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): MVMModel = {
    val data = input.map { case (id, labeledPoint) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      (id, labeledPoint)
    }
    val lfm = new MVMALSRegression(data, views, rank, lambda, useWeightedLambda, miniBatchFraction, storageLevel)
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
        Utils.random.nextGaussian() * 1e-2
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
    val random: JavaRandom = new XORShiftRandom()
    random.setSeed(17425170 - iter / mod)
    random
  }

  /**
   * arr(k + v * rank)= (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
   * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
   */
  private[ml] def predictInterval(rank: Int, arr: VD): ED = {
    val viewSize: Int = arr.length / rank
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
    assert(a.length == b.length)
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
   * arr的长度是rank * viewSize
   * f属于 [viewId * rank ,(viewId +1) * rank)时
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
   * arr的长度是rank * viewSize
   * 当 v=viewId , k属于[0,rank] 时
   * arr(k + v * rank) = \frac{\partial \hat{y}(x|\Theta )}{\partial\theta }
   * 返回 multi * \frac{\partial \hat{y}(x|\Theta )}{\partial\theta }
   * 分类: multi = 1/(1+ \exp(-\hat{y}(x|\Theta)) ) - y
   * 回归: multi = 2(\hat{y}(x|\Theta) -y)
   */
  private[ml] def backwardInterval(
    rank: Int,
    viewId: Int,
    x: ED,
    arr: VD,
    multi: ED): VD = {
    val m = new Array[Double](rank * 2)
    var i = 0
    while (i < rank) {
      val hx = x * arr(i + viewId * rank)
      m(i) += multi * hx
      m(i + rank) += pow(hx, 2)
      i += 1
    }
    m
  }

  /**
   * arr(k + v * rank)= (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
   * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
   */
  private[ml] def sumInterval(rank: Int, arr: Array[Double]): VD = {
    val viewSize: Int = arr.length / rank
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

