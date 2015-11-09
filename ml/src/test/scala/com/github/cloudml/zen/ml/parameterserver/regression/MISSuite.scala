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

package com.github.cloudml.zen.ml.parameterserver.regression

import com.github.cloudml.zen.ml.parameterserver.regression.{LogisticRegressionMIS => NLR}
import com.github.cloudml.zen.ml.regression.{LogisticRegressionMIS => OLR}
import com.github.cloudml.zen.ml.util.SparkUtils._
import com.github.cloudml.zen.ml.util._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.scalatest.{FunSuite, Matchers}

class MISSuite extends FunSuite with SharedSparkContext with Matchers {
  //  ignore("LogisticRegression MIS") {
  //    val zenHome = sys.props.getOrElse("zen.test.home", fail("zen.test.home is not set!"))
  //    val dataSetFile = s"$zenHome/data/binary_classification_data.txt"
  //    val psMaster = "witgo-pro:10010"
  //    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile)
  //    val max = dataSet.map(_.features.activeValuesIterator.map(_.abs).sum + 1L).max
  //
  //    val maxIter = 100
  //    val stepSize = 1 / (2 * max)
  //    val useAdaGrad = false
  //    val regParam = 0.0
  //    val trainDataSet = dataSet.zipWithUniqueId().map { case (LabeledPoint(label, features), id) =>
  //      val newLabel = if (label > 0.0) 1.0 else -1.0
  //      (id, LabeledPoint(newLabel, features))
  //    }.persist()
  //    trainDataSet.count()
  //    val nlr = new NLR(trainDataSet, psMaster, stepSize, regParam, useAdaGrad)
  //    val olr = new OLR(trainDataSet, stepSize, regParam, useAdaGrad)
  //
  //    nlr.run(1)
  //    olr.run(1)
  //
  //    val nv = nlr.features.map(t => (t._1 / 2, t._2))
  //    val ov = olr.features
  //    val numFeatures = nv.count()
  //    println(numFeatures)
  //    assert(nv.count() == ov.count())
  //    val onv = ov.join(nv)
  //    assert(onv.count() == numFeatures)
  //    onv.collect().foreach { case (_, (o, n)) =>
  //      assert((o - n).abs < 1E-6)
  //    }
  //
  //    val nm = nlr.forward(1)
  //    val om = olr.forward(1)
  //
  //    val numSamples = nm.count()
  //    println(numSamples)
  //    assert(nm.count() == om.count())
  //    val collect = om.join(nm.map(t => (-(t._1 / 2 + 1), t._2))).collect()
  //    assert(collect.length == 1605)
  //    collect.foreach { case (_, (o, n)) =>
  //      assert((o - n).abs < 1E-6)
  //    }
  //
  //    val nb = nlr.backward(nm, 1)
  //    val ob = olr.backward(om, 1)
  //    assert(nb.count() === numFeatures)
  //    assert(nb.count === ob.count)
  //    val onb = ob.join(nb.map(t => (t._1 / 2, t._2)))
  //    assert(onb.count() == numFeatures)
  //    onb.collect().foreach { case (_, (o, n)) =>
  //      assert((o - n).abs < 1E-6)
  //    }
  //
  //
  //    nlr.updateWeight(nb, 1, 0.01, 0)
  //    val nv2 = nlr.features.map(t => (t._1 / 2, t._2))
  //    val ov2 = olr.updateWeight(ob, 1, 0.01, 0).filter(t => t._1 >= 0)
  //
  //    assert(ov2.count() == nv2.count())
  //    val onv2 = ov2.join(nv2)
  //    assert(onv2.count() == numFeatures)
  //    onv2.collect().foreach { case (_, (o, n)) =>
  //      println((o - n).abs)
  //      assert((o - n).abs < 1E-6)
  //    }
  //
  //  }

  test("MIS") {
    val zenHome = sys.props.getOrElse("zen.test.home", fail("zen.test.home is not set!"))
    val dataSetFile = s"$zenHome/data/binary_classification_data.txt"
    val psMaster = "witgo-pro:10010"
    val dataSet = MLUtils.loadLibSVMFile(sc, dataSetFile)
    val max = dataSet.map(_.features.activeValuesIterator.map(_.abs).sum + 1L).max

    val maxIter = 100
    val stepSize = 1 / (2 * max)
    val useAdaGrad = false
    val regParam = 0.0
    val trainDataSet = dataSet.zipWithUniqueId().map { case (LabeledPoint(label, features), id) =>
      val newLabel = if (label > 0.0) 1.0 else -1.0
      (id, LabeledPoint(newLabel, features))
    }.persist()
    trainDataSet.count()
    val nlr = new NLR(trainDataSet, psMaster, stepSize, regParam, useAdaGrad)

    nlr.run(100)

  }
}
