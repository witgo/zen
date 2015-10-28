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

package com.github.cloudml.zen.ml.parameterserver.optimization

import scala.collection.mutable.ArrayBuffer

import com.github.cloudml.zen.ml.linalg.BLAS
import org.apache.spark.annotation.{Experimental, DeveloperApi}
import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.{Vectors, Vector}

import org.parameterserver.client.{VectorReader, PSClient}
import org.parameterserver.collect.{IntHashSet, Int2DoubleMap}
import org.parameterserver.protocol.{DoubleArray, DataType}

import scala.util.Random

/**
 * Class used to solve an optimization problem using Gradient Descent.
 * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 */
@Experimental
class GradientDescent(private var gradient: Gradient, private var updater: Updater,
  private var masterSockAddr: String) extends Optimizer with Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var batchSize: Int = 1

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  /**
   * :: Experimental ::
   * Set the mini-batch size for SGD. default value 1.0
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  @Experimental
  def setMiniBatch(batchSize: Int): this.type = {
    this.batchSize = batchSize
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for SGD.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }

  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
   * Set the IP Socket Address for parameter server.
   * @param addr
   * @return
   */
  def setMasterSockAddr(addr: String): this.type = {
    this.masterSockAddr = addr
    this
  }

  /**
   * :: DeveloperApi ::
   * Runs gradient descent on the given training data.
   * @param data training data
   * @param initialWeights initial weights
   * @return solution vector
   */
  @DeveloperApi
  def optimize(data: RDD[(Vector, Vector)], initialWeights: Vector): Vector = {
    val (weights, _) = GradientDescent.runMiniBatchSGD(
      data,
      gradient,
      updater,
      masterSockAddr,
      stepSize,
      numIterations,
      regParam,
      batchSize,
      initialWeights)
    weights
  }

}

/**
 * :: DeveloperApi ::
 * Top-level method to run gradient descent.
 */
@DeveloperApi
object GradientDescent extends Logging {
  /**
   * Run stochastic gradient descent (SGD) in parallel using mini batches.
   * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
   * in order to compute a gradient estimate.
   * Sampling, and averaging the subgradients over this subset is performed using one standard
   * spark map-reduce in each iteration.
   *
   * @param data - Input data for SGD. RDD of the set of data examples, each of
   *             the form (label, [feature values]).
   * @param gradient - Gradient object (used to compute the gradient of the loss function of
   *                 one single data example)
   * @param updater - Updater function to actually perform a gradient step in a given direction.
   * @param stepSize - initial step size for the first step
   * @param numIterations - number of iterations that SGD should be run.
   * @param regParam - regularization parameter
   * @param batchSize -  mini-batch size default value 1.0.
   *
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the
   *         stochastic loss computed for every iteration.
   */
  def runMiniBatchSGD(
    data: RDD[(Vector, Vector)],
    gradient: Gradient,
    updater: Updater,
    masterSockAddr: String,
    stepSize: Double,
    numIterations: Int,
    regParam: Double,
    batchSize: Int,
    initialWeights: Vector): (Vector, Array[Double]) = {
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    val psClient = new PSClient(masterSockAddr)
    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val randStr = Random.nextLong().toString
    // weights vector name
    val wName = "w-" + randStr
    psClient.createVector(wName, true, weights.size, DataType.Double, true)
    // If initial weights are all 0, no need to update initial values
    psClient.updateVector(wName, new DoubleArray(weights.toArray))

    val partitionsSize = data.partitions.length
    for (i <- 1 to numIterations) {
      // psClient.setEpoch(i)
      val (countSum, lossSum) = data.sortBy(t => Random.nextLong()).mapPartitionsWithIndex { case (pid, iter) =>
        val rand = new Random(pid + i * partitionsSize + 17)
        val gName = s"g-$wName-${rand.nextLong().toString}"
        psClient.createVector(gName, wName)

        var innerIter = 1
        var loss = 0D
        var count = 0L
        iter.grouped(batchSize).foreach { seq =>
          val w = Vectors.dense(psClient.getVector(wName).getValues.asInstanceOf[DoubleArray].getValues)
          val g = Vectors.dense(initialWeights.toArray)
          val l = gradient.compute(seq.toIterator, w, g)
          val (updatedGrad, _) = updater.compute(w, g, stepSize, innerIter, regParam)
          psClient.updateVector(gName, new DoubleArray(updatedGrad.toArray))
          psClient.vectorAxpby(wName, 1, gName, 1)
          loss += l._2
          count += l._1
          innerIter += 1
        }
        psClient.removeVector(gName)
        Iterator((count, loss))
      }.reduce((c1, c2) => {
        // c: (count, loss)
        (c1._1 + c2._1, c1._2 + c2._2)
      })
      // println(s"$i ${lossSum / countSum}")
      stochasticLossHistory.append(lossSum / countSum)
    }

    weights = Vectors.dense(psClient.getVector(wName).getValues.asInstanceOf[DoubleArray].getValues)
    psClient.removeVector(wName)
    psClient.close()

    logInfo("GradientDescent.runMiniBatchSGD finished. Last 10 stochastic losses %s".format(
      stochasticLossHistory.takeRight(10).mkString(", ")))

    (weights, stochasticLossHistory.toArray)
  }
}