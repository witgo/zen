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

package com.github.cloudml.zen.graphx.impl

import java.util.UUID

import com.github.cloudml.zen.graphx.{VertexRDD, VertexId}
import org.parameterserver.client.PSClient
import org.parameterserver.protocol.matrix.{Column, Row}
import org.parameterserver.protocol.{DoubleArray, IntArray}

import scala.reflect.ClassTag

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import com.github.cloudml.zen.graphx.util.{PSUtils => GPSUtils, CompletionIterator}

class VertexRDDImpl[VD: ClassTag] private[graphx](
  @transient override val partitionsRDD: RDD[VertexId],
  val masterSockAddr: String,
  override val psName: String,
  override val isDense: Boolean,
  override val rowSize: Long,
  override val colSize: Long,
  val targetStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK)
  extends VertexRDD[VD](partitionsRDD.context, List(new OneToOneDependency(partitionsRDD))) {
  @transient protected lazy val vdTag: ClassTag[VD] = implicitly[ClassTag[VD]]
  private[graphx] var batchSize: Int = 1000

  @transient protected[graphx] def psClient: PSClient = new PSClient(masterSockAddr)

  override protected def getPartitions: Array[Partition] = partitionsRDD.partitions

  /**
   * Provides the `RDD[(VertexId, VD)]` equivalent output.
   */
  override def compute(part: Partition, context: TaskContext): Iterator[(VertexId, VD)] = {
    val client = psClient
    val newIter = firstParent[VertexId].iterator(part, context).grouped(batchSize).map { ids =>
      val values = GPSUtils.get[VD](psClient, psName, ids.map(_.toInt).toArray)
      ids.zip(values).toIterator
    }.flatten
    CompletionIterator[(VertexId, VD),Iterator[(VertexId, VD)]](newIter, client.close())
  }

  override def updateValues(data: RDD[(VertexId, VD)]): this.type = {
    data.foreachPartition { iter =>
      val client = psClient
      iter.grouped(batchSize).foreach { p =>
        val (indices, values) = p.unzip
        GPSUtils.batchUpadte(client, psName, indices.map(_.toInt).toArray, values.toArray)
      }
      client.close()
    }
    this
  }

  override def copy(withValues: Boolean): this.type = {
    val newName = UUID.randomUUID().toString
    val vdClass = vdTag.runtimeClass
    val client = psClient
    if (vdClass == classOf[SV] || vdClass.isAssignableFrom(classOf[SV])) {
      client.createMatrix(newName, psName)
      if (withValues) client.matrixAdd(newName, psName)

    } else {
      client.createVector(newName, psName)
      if (withValues) client.vectorAxpby(newName, 0, psName, 1)
    }
    client.close()
    this
  }
}
