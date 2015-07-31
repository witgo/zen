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
import com.github.cloudml.zen.ml.util.{XORShiftRandom, Utils}
import org.apache.commons.math3.distribution.GammaDistribution
import org.apache.commons.math3.random.Well19937c
import org.apache.spark.{SparkContext, Logging}
import org.apache.spark.graphx._
import org.apache.spark.graphx.impl.{EdgeRDDImpl, GraphImpl}
import org.apache.spark.mllib.linalg.{Vector => SV}
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

  protected val alpha_0 = 1.0
  protected val gamma_0 = 1.0
  protected val beta_0 = 1.0
  protected val mu_0 = 0.0

  // new Array[Double](views.length * rank)
  protected val mu = Array.fill(views.length * rank) {
    Utils.random.nextDouble() * 1e-2
  }

  protected val lambda = Array.fill(views.length * rank) {
    Utils.random.nextDouble() * 1e-2
  }

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
    this
  }

  def views: Array[Long]

  def rank: Int

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
      drawLambda(iter)
      drawMu(iter)
      val previousVertices = vertices
      val margin = forward(iter)
      val (thisNumSamples, costSum, thisMulti) = multiplier(margin, iter)
      multi = thisMulti
      val alpha = drawAlpha(multi, iter)
      val delta = backward(multi, thisNumSamples, iter)
      val sigma = drawSigma(delta, alpha, iter)
      vertices = updateWeight(delta, sigma, alpha, iter)
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
    new MVMModel(rank, views, false, features)
  }

  protected[ml] def forward(iter: Int): VertexRDD[Array[Double]] = {

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
    multi: VertexRDD[VD],
    thisNumSamples: Long,
    iter: Int): VertexRDD[VD] = {
    GraphImpl.fromExistingRDDs(multi, edges).aggregateMessages[VD](ctx => {
      val sampleId = ctx.dstId
      val featureId = ctx.srcId
      val x = ctx.attr
      val arr = ctx.dstAttr
      val viewId = featureId2viewId(featureId, views)
      val m = backwardInterval(rank, viewId, x, arr, arr.last)
      ctx.sendToSrc(m)

    }, reduceInterval, TripletFields.Dst).setName(s"delta-$iter").persist(storageLevel)
  }

  // Updater for Efficient Gibbs Sampling
  protected def updateWeight(
    delta: VertexRDD[Array[Double]],
    sigma: VertexRDD[VD],
    alpha: Double,
    iter: Int): VertexRDD[VD] = {
    // val thisViewId = Utils.random.nextInt(views.length)
    // val thisViewId =  (iter / rank) % views.length
    val thisViewId = iter % views.length

    //  val thisRankId= iter % rank
    val thisRankId = (iter / views.length) % rank

    val rankIndices = 0 until rank
    val seed = Utils.random.nextInt()
    val rand = new Well19937c(seed * iter)
    dataSet.vertices.leftJoin(delta.join(sigma)) { (vid, attr, gradient) =>
      gradient match {
        case Some((grad, reg)) =>
          assert(!isSampleId(vid))
          rand.setSeed(Array(iter, vid.toInt, seed))
          val weight = attr
          val viewId = featureId2viewId(vid, views)
          for (rankId <- rankIndices) {
            if (rankId == thisRankId % rank && viewId == thisViewId) {
              val h2 = grad(rankId + rank)
              val he = grad(rankId)
              val w = weight(rankId)
              val lm = lambda(rankId + viewId * rank) * mu(rankId + viewId * rank)
              weight(rankId) = (alpha * (w * h2 + he) + lm) * reg(rankId)
              weight(rankId) += rand.nextGaussian() * sqrt(reg(rankId))
            }
          }
          weight
        case None => attr

      }
    }.setName(s"vertices-$iter").persist(storageLevel)
  }

  def drawLambda(iter: Int): Unit = {
    val viewsSize = views.length
    val rankIndices = 0 until rank
    val dist = features.aggregate(new Array[Double](viewsSize * (rank + 1)))({ case (arr, (featureId, weight)) =>
      val viewId = featureId2viewId(featureId, views)
      for (rankId <- rankIndices) {
        arr(rankId + viewId * rank) += pow(weight(rankId) - mu(rankId + viewId * rank), 2)
      }
      arr(viewsSize * rank + viewId) += 1
      arr
    }, reduceInterval)
    val seed = Utils.random.nextInt()
    val rand = new Well19937c(Array(seed, iter))
    for (viewId <- views.indices) {
      for (rankId <- rankIndices) {
        val shape = (alpha_0 + dist(viewsSize * rank + viewId) + 1.0) / 2.0
        val scale = (beta_0 * pow(mu(rankId + viewId * rank) - mu_0, 2) + gamma_0 + dist(rankId + viewId * rank)) / 2.0
        val rng = new GammaDistribution(rand, shape, scale)
        lambda(rankId + viewId * rank) = rng.sample()
      }
    }
  }

  def drawMu(iter: Int): Unit = {
    val viewsSize = views.length
    val rankIndices = 0 until rank
    val dist = features.aggregate(new Array[Double](viewsSize * (rank + 1)))({ case (arr, (featureId, weight)) =>
      val viewId = featureId2viewId(featureId, views)
      for (rankId <- rankIndices) {
        arr(rankId + viewId * rank) += weight(rankId)
      }
      arr(viewsSize * rank + viewId) += 1
      arr
    }, reduceInterval)
    for (viewId <- views.indices) {
      for (rankId <- rankIndices) {
        val mu_sigma = 1.0 / ((dist(viewsSize * rank + viewId) + beta_0) * lambda(rankId + viewId * rank))
        val mu_mean = (dist(rankId + viewId * rank) + beta_0 * mu_0) / (dist(viewsSize * rank + viewId) + beta_0)
        mu(rankId + viewId * rank) = mu_mean + Utils.random.nextGaussian() * sqrt(mu_sigma)
      }
    }
  }

  def drawSigma(
    hx2: VertexRDD[VD],
    alpha: Double,
    iter: Int): VertexRDD[VD] = {
    val rankIndices = 0 until rank
    hx2.mapValues { (vid, h2) =>
      val viewId = featureId2viewId(vid, views)
      val arr = new Array[Double](rank)
      for (rankId <- rankIndices) {
        arr(rankId) = 1.0 / (alpha * h2(rankId + rank) + lambda(rankId + viewId * rank))
      }
      arr
    }
  }

  def drawAlpha(
    multi: VertexRDD[VD],
    iter: Int): Double = {
    val e = multi.filter(t => isSampleId(t._1)).mapValues(_.last)
    val shape = (1 + numSamples) / 2
    val scale = e.map(t => pow(t._2, 2)).sum() / 2
    val seed = Utils.random.nextLong()
    val rand = new Well19937c(seed * iter)
    val rng = new GammaDistribution(rand, shape, scale)
    rng.sample()
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
  val storageLevel: StorageLevel) extends MVMALS {

  def this(
    input: RDD[(VertexId, LabeledPoint)],
    views: Array[Long],
    rank: Int = 20,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK) {
    this(initializeDataSet(input, views, rank, storageLevel), views, rank, storageLevel)
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
          val perd = arr.last
          // assert(abs(perd -  predict(m)) < 1e-6)
          val diff = y - perd
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
   * @param storageLevel   缓存级别
   * @return
   */

  def trainRegression(
    input: RDD[(Long, LabeledPoint)],
    numIterations: Int,
    views: Array[Long],
    rank: Int,
    storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK): MVMModel = {
    val data = input.map { case (id, labeledPoint) =>
      assert(id >= 0.0, s"sampleId $id less than 0")
      (id, labeledPoint)
    }
    val lfm = new MVMALSRegression(data, views, rank, storageLevel)
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

    val vertices = (input.map { case (sampleId, labelPoint) =>
      val newId = newSampleId(sampleId)
      val label = Array(labelPoint.label)
      // label point
      (newId, label)
    } ++ edges.map(_.srcId).distinct().map { featureId =>
      // parameter point
      val parms = Array.fill(rank) {
        Utils.random.nextDouble() * 1e-2
      }
      (featureId, parms)
    }).repartition(input.partitions.length)
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

  /**
   * arr(k + v * rank)= (\sum_{i_1 =1}^{I_1+1}z_{i_1}^{(1)}a_{i_1,j}^{(1)}) ..
   * (\sum_{i_m =1}^{I_m+1}z_{i_m}^{(m)}a_{i_m,j}^{(m)})
   */
  private[ml] def predictInterval(rank: Int, arr: VD): ED = {
    val viewSize: Int = arr.length / rank
    var sum = 0.0
    var rankId = 0
    while (rankId < rank) {
      var multi = 1.0
      var viewId = 0
      while (viewId < viewSize) {
        multi *= arr(rankId + viewId * rank)
        viewId += 1
      }
      sum += multi
      rankId += 1
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
    var rankId = 0
    while (rankId < rank) {
      arr(rankId + viewId * rank) += z * w(rankId)
      rankId += 1
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
    var rankId = 0
    while (rankId < rank) {
      val hx = x * arr(rankId + viewId * rank)
      m(rankId) += multi * hx
      m(rankId + rank) += pow(hx, 2)
      rankId += 1
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
    var rankId = 0
    while (rankId < rank) {
      var multi = 1.0
      var viewId = 0
      while (viewId < viewSize) {
        multi *= arr(rankId + viewId * rank)
        viewId += 1
      }
      m(rankId) += multi
      sum += multi
      rankId += 1
    }

    val ret = new Array[Double](rank * viewSize + 1)
    rankId = 0
    while (rankId < rank) {
      var viewId = 0
      while (viewId < viewSize) {
        val vm = arr(rankId + viewId * rank)
        ret(rankId + viewId * rank) = if (vm == 0.0) {
          0.0
        } else {
          m(rankId) / vm
        }
        viewId += 1
      }

      rankId += 1
    }
    ret(rank * viewSize) = sum
    ret
  }
}
