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

import com.github.cloudml.zen.graphx.{VertexRDD, VertexId}
import org.parameterserver.client.PSClient
import org.parameterserver.protocol.matrix.{Column, Row}
import org.parameterserver.protocol.{DoubleArray, IntArray}

import scala.reflect.ClassTag

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.mllib.linalg.{DenseVector => SDV, SparseVector => SSV, Vector => SV}
import com.github.cloudml.zen.graphx.util.{PSUtils => GPSUtils}

class VertexRDDImpl[VD: ClassTag] private[graphx] (
  @transient override val partitionsRDD: RDD[VertexId],
  val masterSockAddr: String,
  override val psName: String,
  override val isDense: Boolean,
  override val rowSize: Long,
  override val colSize: Long,
  val targetStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK)
  extends VertexRDD[VD](partitionsRDD.context, List(new OneToOneDependency(partitionsRDD))) {
  override protected def getPartitions: Array[Partition] = partitionsRDD.partitions

  private[graphx] override def psClient: PSClient = new PSClient(masterSockAddr)

  @transient protected lazy val vdTag: ClassTag[VD] = implicitly[ClassTag[VD]]

  /**
   * Provides the `RDD[(VertexId, VD)]` equivalent output.
   */
  override def compute(part: Partition, context: TaskContext): Iterator[(VertexId, VD)] = {
    val client = psClient
    val vdClass = vdTag.runtimeClass
    if (vdClass == java.lang.Integer.TYPE) {
      firstParent[VertexId].iterator(part, context).map { vid =>
        val value = client.getVector(psName, Array(vid.toInt)).asInstanceOf[IntArray].getValues.head
        (vid, value.asInstanceOf[VD])
      }
    } else if (vdClass == java.lang.Double.TYPE) {
      firstParent[VertexId].iterator(part, context).map { vid =>
        val value = client.getVector(psName, Array(vid.toInt)).asInstanceOf[DoubleArray].getValues.head
        (vid, value.asInstanceOf[VD])
      }
    } else if (vdClass == classOf[SV] || vdClass.isAssignableFrom(classOf[SV])) {
      firstParent[VertexId].iterator(part, context).map { vid =>
        val rowData = client.getMatrix(psName, Array(new Row(vid.toInt))).head
        val value = if (rowData.getColumns != null) {
          new SSV(Int.MaxValue, rowData.getColumns, rowData.getData.asInstanceOf[DoubleArray].getValues)
        } else {
          new SDV(rowData.getData.asInstanceOf[DoubleArray].getValues)
        }
        (vid, value.asInstanceOf[VD])
      }
    } else {
      throw new IllegalArgumentException(s"Unsupported type: $vdClass")
    }
  }

  override def updateValues(data: RDD[(VertexId, VD)]): Unit = {
    updateValues(data, 1000)
  }

  def updateValues(data: RDD[(VertexId, VD)], batchSize: Int): Unit = {
    data.foreachPartition { iter =>
      iter.grouped(batchSize).foreach { p =>
        val (indices, values) = p.unzip
        GPSUtils.batchUpadte(psClient, psName, indices.map(_.toInt).toArray, values.toArray)
      }
      psClient.close()
    }
  }
}
