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

import org.apache.spark.Logging
import org.apache.spark.annotation.{DeveloperApi, Experimental}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV, Vectors}
import org.apache.spark.rdd.RDD
import org.parameterserver.client.PSClient
import org.parameterserver.protocol.{DataType, DoubleArray}
import org.parameterserver.{Configuration => PSConf}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Class used to solve an optimization problem using Gradient Descent.
  * @param gradient Gradient function to be used.
  * @param updater Updater to be used to update weights after every iteration.
  */
@Experimental
class GradientDescent(private var gradient: Gradient, private var updater: Updater) extends Optimizer with Logging {

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
    * :: DeveloperApi ::
    * Runs gradient descent on the given training data.
    * @param data training data
    * @param initialWeights initial weights
    * @return solution vector
    */
  @DeveloperApi
  def optimize(data: RDD[(SV, SV)], initialWeights: SV): SV = {
    val (weights, _) = GradientDescent.runMiniBatchSGD(
      data,
      gradient,
      updater,
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
    data: RDD[(SV, SV)],
    gradient: Gradient,
    updater: Updater,
    stepSize: Double,
    numIterations: Int,
    regParam: Double,
    batchSize: Int,
    initialWeights: SV): (SV, Array[Double]) = {
    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)
    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    // Initialize weights as a column vector
    var weights = Vectors.dense(initialWeights.toArray)
    val numFeatures = initialWeights.size
    val randStr = Random.nextLong().toString

    val psNamespace = s"n-$randStr"
    val psClient = new PSClient(new PSConf(true))
    psClient.setContext(psNamespace)

    // weights vector name
    val wName = s"w-$randStr"
    psClient.createVector(wName, true, weights.size, DataType.Double, true)
    // If initial weights are all 0, no need to update initial values
    psClient.updateVector(wName, new DoubleArray(weights.toArray))

    for (epoch <- 1 to numIterations) {
      // psClient.setEpoch(i)
      // .sortBy(t => Random.nextLong())
      val (countSum, lossSum) = data.mapPartitions { iter =>
        psClient.setContext(psNamespace)
        var sdvIndices: Array[Int] = null
        var innerIter = 1
        var loss = 0D
        var count = 0L
        iter.grouped(batchSize).foreach { seq =>
          val w = Vectors.dense(psClient.getVector(wName).getValues.asInstanceOf[DoubleArray].getValues)
          val g = Vectors.dense(initialWeights.toArray)
          val l = gradient.compute(seq.toIterator, w, g)
          val (updatedGrad, _) = updater.compute(w, g, stepSize, innerIter, regParam)
          updatedGrad match {
            case SDV(values) =>
              if (sdvIndices == null) sdvIndices = (0 until numFeatures).toArray
              psClient.add2Vector(wName, sdvIndices, new DoubleArray(values))
            case SSV(size, indices, values) =>
              psClient.add2Vector(wName, indices, new DoubleArray(values))
          }
          loss += l._2
          count += l._1
          innerIter += 1
        }
        psClient.close()
        Iterator((count, loss))
      }.reduce((c1, c2) => {
        // c: (count, loss)
        (c1._1 + c2._1, c1._2 + c2._2)
      })
      // println(f"epoch: $epoch, loss: ${lossSum / countSum}%1.8f")
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
