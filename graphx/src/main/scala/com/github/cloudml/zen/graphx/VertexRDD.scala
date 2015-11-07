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

package com.github.cloudml.zen.graphx

import org.apache.spark.rdd.RDD
import org.apache.spark.{Dependency, Partition, SparkContext}
import org.parameterserver.client.PSClient

import scala.reflect.ClassTag

abstract class VertexRDD[VD](
  sc: SparkContext,
  deps: Seq[Dependency[_]]) extends RDD[(VertexId, VD)](sc, deps) {

  private[graphx] def psClient: PSClient

  private[graphx] def psName: String

  private[graphx] def isDense: Boolean

  private[graphx] def rowSize: Long

  private[graphx] def colSize: Long

  private[graphx] def partitionsRDD: RDD[VertexId]

  override protected def getPartitions: Array[Partition] = partitionsRDD.partitions

  def updateValues(data: RDD[(VertexId, VD)]): Unit
}
